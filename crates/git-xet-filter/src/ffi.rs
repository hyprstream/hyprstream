//! Safe FFI bindings to libgit2's filter system
//!
//! Provides type-safe wrappers around git2/sys/filter.h

use libc::{c_char, c_int, c_uint, c_void, size_t};
use std::marker::PhantomData;

// Opaque types for type safety (not just c_void)
// Note: These are intentionally zero-sized for compile-time type safety.
// The FFI-safety warnings are expected and suppressed below.
#[repr(transparent)]
pub struct OpaqueGitFilter {
    _private: [u8; 0],
    _phantom: PhantomData<*mut c_void>,
}

#[repr(transparent)]
pub struct OpaqueGitFilterSource {
    _private: [u8; 0],
    _phantom: PhantomData<*mut c_void>,
}

#[repr(transparent)]
pub struct OpaqueGitWriteStream {
    _private: [u8; 0],
    _phantom: PhantomData<*mut c_void>,
}

// Filter mode constants
/// Filter mode for cleaning (workdir → ODB)
pub const GIT_FILTER_CLEAN: c_int = 0;
/// Filter mode for smudging (ODB → workdir)
pub const GIT_FILTER_SMUDGE: c_int = 1;

// Filter source flags
/// Flag indicating operation is towards worktree (checkout operation)
pub const GIT_FILTER_TO_WORKTREE: u32 = 1 << 0;

// Filter return codes
/// Filter passthrough - filter declines to process this file
pub const GIT_PASSTHROUGH: c_int = -30;

// Filter version
/// Current filter structure version for ABI compatibility
pub const GIT_FILTER_VERSION: c_uint = 1;

// libgit2 configuration
/// Memory window size for large file operations (8 MB)
pub const LIBGIT2_MWINDOW_SIZE: usize = 8 * 1024 * 1024;

// Callback function types
pub type FilterInitializeFn = extern "C" fn(*mut OpaqueGitFilter) -> c_int;
pub type FilterShutdownFn = extern "C" fn(*mut OpaqueGitFilter);
pub type FilterCheckFn = extern "C" fn(
    *mut OpaqueGitFilter,
    *mut *mut c_void,
    *const OpaqueGitFilterSource,
    *const *const c_char,
) -> c_int;
pub type FilterStreamFn = extern "C" fn(
    *mut *mut OpaqueGitWriteStream,  // out (pointer to pointer)
    *mut OpaqueGitFilter,             // self
    *mut *mut c_void,                 // payload
    *const OpaqueGitFilterSource,    // src
    *mut OpaqueGitWriteStream,        // next
) -> c_int;
pub type FilterCleanupFn = extern "C" fn(*mut OpaqueGitFilter, *mut c_void);

/// Writestream callback types
pub type WriteStreamWriteFn = extern "C" fn(
    stream: *mut GitWriteStreamBase,
    buffer: *const c_char,
    len: size_t,
) -> c_int;

pub type WriteStreamCloseFn = extern "C" fn(
    stream: *mut GitWriteStreamBase,
) -> c_int;

pub type WriteStreamFreeFn = extern "C" fn(
    stream: *mut GitWriteStreamBase,
);

/// Base writestream structure (matches libgit2's git_writestream)
/// Custom writestreams should embed this at the start of their struct
#[repr(C)]
pub struct GitWriteStreamBase {
    pub write: WriteStreamWriteFn,
    pub close: WriteStreamCloseFn,
    pub free: WriteStreamFreeFn,
}

/// Filter structure (must match libgit2's git_filter)
#[repr(C)]
pub struct GitFilter {
    pub version: c_uint,
    pub attributes: *const c_char,
    pub initialize: Option<FilterInitializeFn>,
    pub shutdown: Option<FilterShutdownFn>,
    pub check: Option<FilterCheckFn>,
    /// Reserved field (or apply callback in older versions)
    /// This field exists for ABI compatibility with libgit2
    pub reserved: *mut c_void,
    pub stream: Option<FilterStreamFn>,
    pub cleanup: Option<FilterCleanupFn>,
}

// Safety: GitFilter has no interior pointers and matches C layout
unsafe impl Send for GitFilter {}
unsafe impl Sync for GitFilter {}

// These functions are part of libgit2's filter API
// They're available via libgit2-sys's vendored libgit2
#[allow(improper_ctypes)] // Opaque types are intentionally zero-sized for type safety
extern "C" {
    /// Initialize the global filter registry (internal libgit2 function)
    pub fn git_filter_global_init() -> c_int;

    /// Initialize a git_filter with default values
    pub fn git_filter_init(filter: *mut GitFilter, version: c_uint) -> c_int;

    /// Register a filter with libgit2
    pub fn git_filter_register(
        name: *const c_char,
        filter: *mut GitFilter,
        priority: c_int,
    ) -> c_int;

    /// Unregister a filter
    pub fn git_filter_unregister(name: *const c_char) -> c_int;

    /// Get filter source mode (clean or smudge)
    pub fn git_filter_source_mode(src: *const OpaqueGitFilterSource) -> c_int;

    /// Get filter source flags (e.g., GIT_FILTER_TO_WORKTREE)
    pub fn git_filter_source_flags(src: *const OpaqueGitFilterSource) -> u32;

    /// Get filter source repo
    pub fn git_filter_source_repo(src: *const OpaqueGitFilterSource) -> *mut c_void;

    /// Get filter source path
    pub fn git_filter_source_path(src: *const OpaqueGitFilterSource) -> *const c_char;

}

// Safety: GitWriteStreamBase is a simple C struct with function pointers
unsafe impl Send for GitWriteStreamBase {}
unsafe impl Sync for GitWriteStreamBase {}
