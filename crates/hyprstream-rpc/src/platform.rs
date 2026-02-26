//! Platform-adaptive concurrency primitives.
//!
//! Provides type aliases for cross-platform concurrency primitives:
//! - Native: parking_lot (multi-threaded)
//! - WASM: RefCell/Cell (single-threaded worker)
//!
//! Designed for future migration to SharedArrayBuffer when Rust wasm atomics stabilize.

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;
#[cfg(target_arch = "wasm32")]
pub use wasm::*;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    pub type Lock<T> = parking_lot::RwLock<T>;
    pub type Counter = std::sync::atomic::AtomicU64;

    pub fn counter_next(c: &Counter) -> u64 {
        c.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::cell::{Cell, RefCell};

    /// RefCell-based lock for single-threaded WASM workers.
    ///
    /// Each Web Worker runs one WASM instance — no concurrent access.
    pub type Lock<T> = RefCell<T>;

    /// Cell-based counter for single-threaded WASM workers.
    pub type Counter = Cell<u64>;

    pub fn counter_next(c: &Counter) -> u64 {
        let v = c.get();
        c.set(v + 1);
        v
    }
}
