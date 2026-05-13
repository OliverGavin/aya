//! In-memory BPF map store for testing.
//!
//! Provides a thread-local map store that intercepts BPF syscalls and implements
//! map operations in userspace. This allows tests to use real aya map types
//! (e.g. `Array`, `HashMap`, `PerCpuArray`) without a kernel.
//!
//! # Example
//!
//! ```ignore
//! use aya::maps::{Map, MapData, PerCpuArray};
//! use aya::sys::test_utils::map_store;
//!
//! // Install the syscall handler
//! map_store::setup();
//!
//! // Create a real PerCpuArray backed by the in-memory store
//! let map_data = map_store::new_map_data::<u32, u64>(
//!     bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY, 16,
//! );
//! let mut array: PerCpuArray<_, u64> =
//!     PerCpuArray::try_from(Map::PerCpuArray(map_data)).unwrap();
//!
//! // Use it exactly like a real map
//! array.set(0, PerCpuValues::try_from(vec![42u64; nr_cpus])?, 0)?;
//! let values = array.get(&0, 0)?;
//! ```

use std::cell::RefCell;
use std::collections::BTreeMap;

use aya_obj::generated::{bpf_cmd, bpf_map_type};
pub use aya_obj::generated::bpf_map_type as MapType;
use aya_obj::{EbpfSectionKind, maps::LegacyMap};

use crate::maps::MapData;
use crate::sys::fake::override_syscall;
use crate::sys::{SysResult, Syscall};
use crate::util::nr_cpus;
use crate::{MockableFd, Pod, bpf_map_def};

/// Metadata for a created map.
#[derive(Clone, Debug)]
struct MapMeta {
    map_type: u32,
    key_size: u32,
    value_size: u32,
    max_entries: u32,
}

impl MapMeta {
    /// The size of the value buffer for syscall operations.
    /// For per-CPU maps, this is `nr_cpus * value_size` (aligned to 8).
    fn syscall_value_size(&self) -> usize {
        if self.is_per_cpu() {
            let aligned = (self.value_size as usize).next_multiple_of(8);
            nr_cpus().unwrap() * aligned
        } else {
            self.value_size as usize
        }
    }

    fn is_per_cpu(&self) -> bool {
        matches!(
            bpf_map_type::try_from(self.map_type),
            Ok(bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY
                | bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH
                | bpf_map_type::BPF_MAP_TYPE_LRU_PERCPU_HASH)
        )
    }

    fn is_array(&self) -> bool {
        matches!(
            bpf_map_type::try_from(self.map_type),
            Ok(bpf_map_type::BPF_MAP_TYPE_ARRAY | bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY)
        )
    }
}

/// Thread-local in-memory map store.
struct Store {
    next_fd: i64,
    maps: BTreeMap<u32, MapMeta>,
    data: BTreeMap<u32, BTreeMap<Vec<u8>, Vec<u8>>>,
}

impl Store {
    fn new() -> Self {
        Self {
            next_fd: MockableFd::mock_signed_fd() as i64,
            maps: BTreeMap::new(),
            data: BTreeMap::new(),
        }
    }

    fn create_map(&mut self, meta: MapMeta) -> i64 {
        let fd = self.next_fd;
        self.next_fd += 1;
        let fd_u32 = fd as u32;
        // Pre-populate array maps with zeroed entries
        if meta.is_array() {
            let entries = self.data.entry(fd_u32).or_default();
            let value_size = meta.syscall_value_size();
            for i in 0..meta.max_entries {
                let key = i.to_ne_bytes().to_vec();
                entries.insert(key, vec![0u8; value_size]);
            }
        }
        self.maps.insert(fd_u32, meta);
        self.data.entry(fd_u32).or_default();
        fd
    }

    fn lookup(&self, fd: u32, key: &[u8]) -> Option<&Vec<u8>> {
        self.data.get(&fd)?.get(key)
    }

    fn update(&mut self, fd: u32, key: Vec<u8>, value: Vec<u8>) -> Result<(), i32> {
        let meta = self.maps.get(&fd).ok_or(libc::EBADF)?;
        if meta.is_array() {
            // Array maps: key must be within bounds
            if key.len() == 4 {
                let index = u32::from_ne_bytes(key[..4].try_into().unwrap());
                if index >= meta.max_entries {
                    return Err(libc::E2BIG);
                }
            }
        }
        let entries = self.data.entry(fd).or_default();
        entries.insert(key, value);
        Ok(())
    }

    fn delete(&mut self, fd: u32, key: &[u8]) -> Result<(), i32> {
        let meta = self.maps.get(&fd).ok_or(libc::EBADF)?;
        if meta.is_array() {
            return Err(libc::EINVAL); // Can't delete from arrays
        }
        let entries = self.data.get_mut(&fd).ok_or(libc::EBADF)?;
        entries.remove(key).ok_or(libc::ENOENT)?;
        Ok(())
    }

    fn get_next_key(&self, fd: u32, key: Option<&[u8]>) -> Option<Vec<u8>> {
        let entries = self.data.get(&fd)?;
        match key {
            None => entries.keys().next().cloned(),
            Some(k) => {
                use std::ops::Bound;
                let mut range = entries.range((Bound::Excluded(k.to_vec()), Bound::Unbounded));
                range.next().map(|(k, _)| k.clone())
            }
        }
    }
}

thread_local! {
    static STORE: RefCell<Store> = RefCell::new(Store::new());
}

/// Install the syscall override that routes BPF map operations to the
/// in-memory store.
///
/// Call this once at the start of your test. All subsequent map operations
/// (create, lookup, update, delete, get_next_key) will be handled in-memory.
pub fn setup() {
    STORE.with(|s| *s.borrow_mut() = Store::new());
    override_syscall(|call| handle_syscall(call));
}

/// Reset the map store, clearing all maps and data.
pub fn reset() {
    STORE.with(|s| *s.borrow_mut() = Store::new());
}

fn handle_syscall(call: Syscall<'_>) -> SysResult {
    match call {
        Syscall::Ebpf { cmd: bpf_cmd::BPF_MAP_CREATE, attr } => {
            let u = unsafe { &attr.__bindgen_anon_1 };
            let meta = MapMeta {
                map_type: u.map_type,
                key_size: u.key_size,
                value_size: u.value_size,
                max_entries: u.max_entries,
            };
            let fd = STORE.with(|s| s.borrow_mut().create_map(meta));
            Ok(fd)
        }
        Syscall::Ebpf { cmd: bpf_cmd::BPF_MAP_LOOKUP_ELEM, attr } => {
            let u = unsafe { &attr.__bindgen_anon_2 };
            let fd = u.map_fd;
            let meta = STORE.with(|s| s.borrow().maps.get(&fd).cloned());
            let meta = meta.ok_or((-1, std::io::Error::from_raw_os_error(libc::EBADF)))?;
            let key_size = meta.key_size as usize;
            let value_size = meta.syscall_value_size();
            let key = unsafe { std::slice::from_raw_parts(u.key as *const u8, key_size) };
            let value_ptr = unsafe { u.__bindgen_anon_1.value } as *mut u8;

            STORE.with(|s| {
                let store = s.borrow();
                match store.lookup(fd, key) {
                    Some(data) => {
                        let copy_len = data.len().min(value_size);
                        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), value_ptr, copy_len) };
                        Ok(0)
                    }
                    None => Err((-1, std::io::Error::from_raw_os_error(libc::ENOENT))),
                }
            })
        }
        Syscall::Ebpf { cmd: bpf_cmd::BPF_MAP_UPDATE_ELEM, attr } => {
            let u = unsafe { &attr.__bindgen_anon_2 };
            let fd = u.map_fd;
            let meta = STORE.with(|s| s.borrow().maps.get(&fd).cloned());
            let meta = meta.ok_or((-1, std::io::Error::from_raw_os_error(libc::EBADF)))?;
            let key_size = meta.key_size as usize;
            let value_size = meta.syscall_value_size();
            let key = unsafe { std::slice::from_raw_parts(u.key as *const u8, key_size) }.to_vec();
            let value_ptr = unsafe { u.__bindgen_anon_1.value } as *const u8;
            let value = unsafe { std::slice::from_raw_parts(value_ptr, value_size) }.to_vec();

            STORE.with(|s| {
                s.borrow_mut().update(fd, key, value)
                    .map(|()| 0)
                    .map_err(|e| (-1, std::io::Error::from_raw_os_error(e)))
            })
        }
        Syscall::Ebpf { cmd: bpf_cmd::BPF_MAP_DELETE_ELEM, attr } => {
            let u = unsafe { &attr.__bindgen_anon_2 };
            let fd = u.map_fd;
            let meta = STORE.with(|s| s.borrow().maps.get(&fd).cloned());
            let meta = meta.ok_or((-1, std::io::Error::from_raw_os_error(libc::EBADF)))?;
            let key_size = meta.key_size as usize;
            let key = unsafe { std::slice::from_raw_parts(u.key as *const u8, key_size) };

            STORE.with(|s| {
                s.borrow_mut().delete(fd, key)
                    .map(|()| 0)
                    .map_err(|e| (-1, std::io::Error::from_raw_os_error(e)))
            })
        }
        Syscall::Ebpf { cmd: bpf_cmd::BPF_MAP_GET_NEXT_KEY, attr } => {
            let u = unsafe { &attr.__bindgen_anon_2 };
            let fd = u.map_fd;
            let meta = STORE.with(|s| s.borrow().maps.get(&fd).cloned());
            let meta = meta.ok_or((-1, std::io::Error::from_raw_os_error(libc::EBADF)))?;
            let key_size = meta.key_size as usize;
            let key = if u.key == 0 {
                None
            } else {
                Some(unsafe { std::slice::from_raw_parts(u.key as *const u8, key_size) })
            };
            let next_key_ptr = unsafe { u.__bindgen_anon_1.next_key } as *mut u8;

            STORE.with(|s| {
                let store = s.borrow();
                match store.get_next_key(fd, key) {
                    Some(next) => {
                        unsafe { std::ptr::copy_nonoverlapping(next.as_ptr(), next_key_ptr, key_size) };
                        Ok(0)
                    }
                    None => Err((-1, std::io::Error::from_raw_os_error(libc::ENOENT))),
                }
            })
        }
        // Pass through other BPF commands (e.g. PROG_LOAD)
        Syscall::Ebpf { cmd: bpf_cmd::BPF_PROG_LOAD, .. } => {
            Ok(MockableFd::mock_signed_fd() as i64)
        }
        Syscall::Ebpf { .. } => Ok(0),
        _ => Ok(0),
    }
}

/// Create a [`MapData`] instance backed by the in-memory store.
///
/// This is the primary way to create testable map instances. The returned
/// `MapData` can be wrapped in [`Map`](crate::maps::Map) and converted to
/// any typed map (e.g. `Array`, `HashMap`, `PerCpuArray`).
///
/// # Example
///
/// ```ignore
/// use aya::maps::{Map, Array, HashMap, PerCpuArray};
/// use aya::sys::test_utils::map_store;
/// use aya_obj::generated::bpf_map_type;
///
/// map_store::setup();
///
/// // Array
/// let data = map_store::new_map_data::<u32, u64>(bpf_map_type::BPF_MAP_TYPE_ARRAY, 16);
/// let mut array: Array<_, u64> = Array::try_from(Map::Array(data)).unwrap();
///
/// // HashMap
/// let data = map_store::new_map_data::<u32, u64>(bpf_map_type::BPF_MAP_TYPE_HASH, 1024);
/// let mut map: HashMap<_, u32, u64> = HashMap::try_from(Map::HashMap(data)).unwrap();
///
/// // PerCpuArray
/// let data = map_store::new_map_data::<u32, u64>(bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY, 16);
/// let mut pca: PerCpuArray<_, u64> = PerCpuArray::try_from(Map::PerCpuArray(data)).unwrap();
/// ```
pub fn new_map_data<K: Pod, V: Pod>(map_type: bpf_map_type, max_entries: u32) -> MapData {
    let obj = aya_obj::Map::Legacy(LegacyMap {
        def: bpf_map_def {
            map_type: map_type as u32,
            key_size: size_of::<K>() as u32,
            value_size: size_of::<V>() as u32,
            max_entries,
            ..Default::default()
        },
        section_index: 0,
        section_kind: EbpfSectionKind::Maps,
        data: Vec::new(),
        symbol_index: None,
    });
    MapData::create(obj, "test_map", None).unwrap()
}
