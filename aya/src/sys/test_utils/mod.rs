//! Test utilities for mocking BPF syscalls.
//!
//! This module is only available when the `test-utils` feature is enabled.
//! It provides tools to intercept BPF syscalls in tests, allowing userspace
//! code to be tested without a real kernel.
//!
//! # Quick Start
//!
//! ```ignore
//! use aya::maps::{Map, PerCpuArray, PerCpuValues};
//! use aya::sys::test_utils::map_store;
//! use aya::util::nr_cpus;
//! use aya_obj::generated::bpf_map_type;
//!
//! // Set up the in-memory map store
//! map_store::setup();
//!
//! // Create a real PerCpuArray backed by the store
//! let data = map_store::new_map_data::<u32, u64>(
//!     bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY, 16,
//! );
//! let mut array: PerCpuArray<_, u64> =
//!     PerCpuArray::try_from(Map::PerCpuArray(data)).unwrap();
//!
//! // Use it like a real map
//! let nr_cpus = nr_cpus().unwrap();
//! array.set(0, PerCpuValues::try_from(vec![42u64; nr_cpus]).unwrap(), 0).unwrap();
//! let values = array.get(&0, 0).unwrap();
//! assert_eq!(values[0], 42);
//! ```

pub use aya_obj::generated::{bpf_attr, bpf_cmd};

pub use super::fake::override_syscall;
pub use super::{PerfEventIoctlRequest, SysResult, Syscall};

pub mod map_store;

/// Returns the fake file descriptor value used internally by aya's test
/// infrastructure. Return this from your `override_syscall` handler for
/// syscalls that create FDs (e.g. `BPF_MAP_CREATE`).
pub const fn mock_fd() -> i64 {
    crate::MockableFd::mock_signed_fd() as i64
}
