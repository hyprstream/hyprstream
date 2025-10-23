//! Safe FFI bindings to libgit2's filter system
//!
//! Provides type-safe wrappers around git2/sys/filter.h

use libc::{c_char, c_int, c_uint, c_void, size_t};
use std::marker::PhantomData;

// Re-export libgit2-sys types we need
pub use libgit2_sys::{git_buf, git_buf_dispose};

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
pub const GIT_FILTER_CLEAN: c_int = 0;
pub const GIT_FILTER_SMUDGE: c_int = 1;

// Filter return codes
pub const GIT_PASSTHROUGH: c_int = -30; // Filter doesn't want to process this file

// Filter version
pub const GIT_FILTER_VERSION: c_uint = 1;

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

/// Apply function for use with git_filter_buffered_stream_new
pub type FilterApplyFn = extern "C" fn(
    *mut GitBuf,                    // to (output)
    *const GitBuf,                  // from (input)
    *const OpaqueGitFilterSource,  // src (source info)
    *mut *mut c_void,              // payload (user data)
) -> c_int;

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

// Re-export git_buf from libgit2-sys as GitBuf for compatibility
pub type GitBuf = git_buf;

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

    /// Get filter source buffer (input data to filter)
    pub fn git_filter_source_buffer(src: *const OpaqueGitFilterSource) -> *const git_buf;

    /// Get filter source path
    pub fn git_filter_source_path(src: *const OpaqueGitFilterSource) -> *const c_char;

    /// Write to an ODB stream (substitute for writestream)
    /// This is a similar API available in libgit2
    pub fn git_odb_stream_write(
        stream: *mut c_void,
        buffer: *const c_char,
        len: size_t,
    ) -> c_int;

    /// Finalize an ODB stream (substitute for writestream_close)
    pub fn git_odb_stream_finalize_write(
        out: *mut git2::Oid,
        stream: *mut c_void,
    ) -> c_int;

    /// Free an ODB stream (substitute for writestream_free)
    pub fn git_odb_stream_free(stream: *mut c_void);

    /// Create a buffered stream for filter processing
    /// This is a helper function that handles the streaming API for us
    pub fn git_filter_buffered_stream_new(
        out: *mut *mut OpaqueGitWriteStream,
        filter: *mut OpaqueGitFilter,
        apply_fn: FilterApplyFn,
        temp_buf: *mut GitBuf,  // Pass NULL for auto allocation
        payload: *mut *mut c_void,
        src: *const OpaqueGitFilterSource,
        next: *mut OpaqueGitWriteStream,
    ) -> c_int;

    /// Set git_buf from bytes
    pub fn git_buf_set(
        buf: *mut GitBuf,
        data: *const c_void,
        len: size_t,
    ) -> c_int;
}

/// Helper functions for git_buf (implemented as inline accessors)
/// These are not actual libgit2 functions but inline macros in C
#[inline]
pub unsafe fn git_buf_ptr(buf: *const GitBuf) -> *const c_char {
    if buf.is_null() {
        std::ptr::null()
    } else {
        (*buf).ptr
    }
}

#[inline]
pub unsafe fn git_buf_len(buf: *const GitBuf) -> size_t {
    if buf.is_null() {
        0
    } else {
        (*buf).size
    }
}

/// RAII wrapper for git writestream
pub struct GitWriteStream {
    raw: *mut OpaqueGitWriteStream,
    owned: bool,
}

impl GitWriteStream {
    /// Create from raw pointer (does not take ownership)
    ///
    /// # Safety
    /// Pointer must be valid for the lifetime of this wrapper
    pub unsafe fn from_raw(raw: *mut OpaqueGitWriteStream) -> Self {
        Self { raw, owned: false }
    }

    /// Create from raw pointer (takes ownership)
    ///
    /// # Safety
    /// Pointer must be valid and caller must not use it after this call
    pub unsafe fn from_raw_owned(raw: *mut OpaqueGitWriteStream) -> Self {
        Self { raw, owned: true }
    }

    /// Write data to stream
    pub fn write(&mut self, data: &[u8]) -> Result<(), git2::Error> {
        if self.raw.is_null() {
            return Err(git2::Error::from_str("Null stream"));
        }

        unsafe {
            // Cast OpaqueGitWriteStream to c_void for ODB stream functions
            let result = git_odb_stream_write(
                self.raw as *mut c_void,
                data.as_ptr() as *const c_char,
                data.len() as size_t,
            );
            if result < 0 {
                Err(git2::Error::from_str("Write failed"))
            } else {
                Ok(())
            }
        }
    }

    /// Close the stream
    pub fn close(mut self) -> Result<(), git2::Error> {
        if self.raw.is_null() {
            return Ok(());
        }

        unsafe {
            let mut oid = std::mem::zeroed();
            let result = git_odb_stream_finalize_write(
                &mut oid,
                self.raw as *mut c_void,
            );
            self.raw = std::ptr::null_mut(); // Prevent double-close
            if result < 0 {
                Err(git2::Error::from_str("Close failed"))
            } else {
                Ok(())
            }
        }
    }
}

impl Drop for GitWriteStream {
    fn drop(&mut self) {
        if self.owned && !self.raw.is_null() {
            unsafe {
                git_odb_stream_free(self.raw as *mut c_void);
            }
        }
    }
}
