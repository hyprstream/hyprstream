//! C callback implementations for libgit2 filter

#[cfg(feature = "xet-storage")]
use dashmap::DashMap;

#[cfg(feature = "xet-storage")]
use once_cell::sync::Lazy;

#[cfg(feature = "xet-storage")]
use std::sync::Arc;

#[cfg(feature = "xet-storage")]
use std::ffi::CStr;

#[cfg(feature = "xet-storage")]
use libc::c_int;

#[cfg(feature = "xet-storage")]
use crate::ffi::*;

#[cfg(feature = "xet-storage")]
use crate::filter::XetFilterPayload;

#[cfg(feature = "xet-storage")]
use crate::error::{FfiResult, XetError, XetErrorKind};

/// Global payload registry
///
/// Maps filter instance pointer -> payload. Used because libgit2 doesn't provide
/// a way to pass user data to all callbacks. Using instance address instead of
/// name allows multiple filter instances with different configurations.
#[cfg(feature = "xet-storage")]
pub(crate) static PAYLOAD_REGISTRY: Lazy<DashMap<usize, Arc<XetFilterPayload>>> =
    Lazy::new(DashMap::new);

/// Register payload for a filter instance
#[cfg(feature = "xet-storage")]
pub(crate) fn register_payload(filter_ptr: *const OpaqueGitFilter, payload: XetFilterPayload) {
    let key = filter_ptr as usize;
    PAYLOAD_REGISTRY.insert(key, Arc::new(payload));
    tracing::debug!("Registered payload for filter instance {:x}", key);
}

/// Unregister and return payload
///
/// Returns None if the payload was not found (e.g., double-unregister or corrupted registry)
#[cfg(feature = "xet-storage")]
pub(crate) fn unregister_payload(filter_ptr: *const OpaqueGitFilter) -> Option<Arc<XetFilterPayload>> {
    let key = filter_ptr as usize;
    tracing::debug!("Unregistering payload for filter instance {:x}", key);
    PAYLOAD_REGISTRY.remove(&key).map(|(_, p)| p)
}

/// Get payload by filter instance
#[cfg(feature = "xet-storage")]
fn get_payload(filter_ptr: *const OpaqueGitFilter) -> Option<Arc<XetFilterPayload>> {
    let key = filter_ptr as usize;
    PAYLOAD_REGISTRY.get(&key).map(|entry| entry.value().clone())
}

// Implement Clone for XetFilterPayload
#[cfg(feature = "xet-storage")]
impl Clone for XetFilterPayload {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            runtime: self.runtime.clone(),
        }
    }
}

/// Initialize callback (called once when filter is first used)
#[cfg(feature = "xet-storage")]
pub extern "C" fn xet_filter_initialize(_filter: *mut OpaqueGitFilter) -> c_int {
    tracing::debug!("XET filter initialized");
    0
}

/// Shutdown callback (called when filter is unregistered)
#[cfg(feature = "xet-storage")]
pub extern "C" fn xet_filter_shutdown(_filter: *mut OpaqueGitFilter) {
    tracing::debug!("XET filter shutdown");
}

/// Check callback (determines if filter should apply)
///
/// # Safety
/// This function is called by libgit2 with valid pointers. The `source` pointer
/// must be valid for the duration of this call. This is guaranteed by libgit2's
/// callback contract. The raw pointer dereferences are safe within this context.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[cfg(feature = "xet-storage")]
pub extern "C" fn xet_filter_check(
    _filter: *mut OpaqueGitFilter,
    _payload_out: *mut *mut libc::c_void,
    source: *const OpaqueGitFilterSource,
    _attr_values: *const *const libc::c_char,
) -> c_int {
    // SAFETY: libgit2 guarantees source is valid for this callback
    let mode = unsafe { git_filter_source_mode(source) };

    // CRITICAL: During clone/checkout, libgit2 should ONLY call SMUDGE (ODB → workdir)
    // NEVER CLEAN (workdir → ODB). If CLEAN is being called during checkout, something
    // is wrong with how libgit2 is invoking the filter.
    //
    // Refuse to CLEAN during checkout to prevent uploading files during clone.
    if mode == GIT_FILTER_CLEAN {
        // Check if this is a checkout operation (TO_WORKDIR flag)
        // SAFETY: libgit2 guarantees source is valid for this callback
        let flags = unsafe { git_filter_source_flags(source) };
        const GIT_FILTER_TO_WORKTREE: u32 = 1 << 0;

        if (flags & GIT_FILTER_TO_WORKTREE) != 0 {
            tracing::warn!(
                "XET filter: Refusing CLEAN during checkout (this indicates a libgit2 bug)"
            );
            // Return GIT_PASSTHROUGH to skip this filter
            return GIT_PASSTHROUGH;
        }
    }

    // Apply filter if libgit2 matched our attributes
    0
}

/// Stream callback (performs actual filtering)
///
/// # Safety
/// This function is called by libgit2 with valid pointers. All pointer parameters
/// must be valid for the duration of this call. This is guaranteed by libgit2's
/// callback contract. The raw pointer dereferences are safe within this context.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[cfg(feature = "xet-storage")]
pub extern "C" fn xet_filter_stream(
    _out: *mut *mut OpaqueGitWriteStream,
    filter: *mut OpaqueGitFilter,
    _payload_ptr: *mut *mut libc::c_void,
    source: *const OpaqueGitFilterSource,
    next: *mut OpaqueGitWriteStream,
) -> c_int {
    // SAFETY: libgit2 guarantees source is valid for this callback
    let mode = unsafe { git_filter_source_mode(source) };

    // Get file path for logging
    // SAFETY: libgit2 guarantees source is valid for this callback
    let path = unsafe {
        let path_ptr = git_filter_source_path(source);
        if path_ptr.is_null() {
            return XetError::new(XetErrorKind::IoError, "Null path").to_ffi_code();
        }
        match CStr::from_ptr(path_ptr).to_str() {
            Ok(s) => s,
            Err(_) => {
                return XetError::new(XetErrorKind::IoError, "Invalid UTF-8 in path").to_ffi_code()
            }
        }
    };

    // Get payload using filter instance pointer
    let payload = match get_payload(filter) {
        Some(p) => p,
        None => {
            return XetError::new(
                XetErrorKind::StorageNotInitialized,
                format!("XET storage not initialized for filter {:p}", filter),
            )
            .to_ffi_code();
        }
    };

    match mode {
        GIT_FILTER_CLEAN => {
            tracing::debug!("XET clean: {}", path);
            xet_clean_stream(&payload, source, next, path).to_ffi_code()
        }
        GIT_FILTER_SMUDGE => {
            tracing::debug!("XET smudge: {}", path);
            xet_smudge_stream(&payload, source, next, path).to_ffi_code()
        }
        _ => XetError::new(XetErrorKind::RuntimeError, "Unknown filter mode").to_ffi_code(),
    }
}

/// Cleanup callback (releases per-source resources)
#[cfg(feature = "xet-storage")]
pub extern "C" fn xet_filter_cleanup(_filter: *mut OpaqueGitFilter, payload: *mut libc::c_void) {
    if !payload.is_null() {
        // Currently we don't use per-source payload
    }
}

/// Clean stream: file → XET pointer
/// Reads file content from git's internal buffer, uploads to CAS, writes pointer
#[cfg(feature = "xet-storage")]
fn xet_clean_stream(
    payload: &XetFilterPayload,
    source: *const OpaqueGitFilterSource,
    output: *mut OpaqueGitWriteStream,
    path: &str,
) -> crate::error::Result<()> {

    // Read input data from git's buffer
    let input_data = unsafe {
        let buf_ptr = crate::ffi::git_filter_source_buffer(source);
        if buf_ptr.is_null() {
            return Err(XetError::new(
                XetErrorKind::IoError,
                "Failed to read filter source buffer",
            ));
        }
        let buf = &*buf_ptr;
        if buf.ptr.is_null() || buf.size == 0 {
            // Empty file - write empty pointer
            let mut output_wrapper = GitWriteStream::from_raw(output);
            output_wrapper.write(b"").map_err(|e| {
                XetError::new(XetErrorKind::IoError, format!("Write failed: {}", e))
            })?;
            return Ok(());
        }
        std::slice::from_raw_parts(buf.ptr as *const u8, buf.size)
    };

    // Write to temporary file for upload
    let temp_file = tempfile::NamedTempFile::new().map_err(|e| {
        XetError::new(
            XetErrorKind::IoError,
            format!("Failed to create temp file: {}", e),
        )
    })?;

    std::fs::write(temp_file.path(), input_data).map_err(|e| {
        XetError::new(
            XetErrorKind::IoError,
            format!("Failed to write temp file: {}", e),
        )
    })?;

    // Upload to XET CAS
    let pointer = payload
        .runtime
        .block_on(payload.storage.clean_file(temp_file.path()))??;

    tracing::debug!("XET clean {}: {} bytes -> pointer", path, input_data.len());

    // Write pointer to output
    let mut output_wrapper = unsafe { GitWriteStream::from_raw(output) };
    output_wrapper
        .write(pointer.as_bytes())
        .map_err(|e| XetError::new(XetErrorKind::IoError, format!("Write failed: {}", e)))?;

    Ok(())
}

/// Smudge stream: XET pointer → file
/// Reads pointer from git's internal buffer, downloads from CAS, writes file content
#[cfg(feature = "xet-storage")]
fn xet_smudge_stream(
    payload: &XetFilterPayload,
    source: *const OpaqueGitFilterSource,
    output: *mut OpaqueGitWriteStream,
    path: &str,
) -> crate::error::Result<()> {
    // Read pointer from git's buffer
    let pointer_data = unsafe {
        let buf_ptr = crate::ffi::git_filter_source_buffer(source);
        if buf_ptr.is_null() {
            return Err(XetError::new(
                XetErrorKind::IoError,
                "Failed to read filter source buffer",
            ));
        }
        let buf = &*buf_ptr;
        if buf.ptr.is_null() || buf.size == 0 {
            // Empty file - pass through
            let mut output_wrapper = GitWriteStream::from_raw(output);
            output_wrapper.write(b"").map_err(|e| {
                XetError::new(XetErrorKind::IoError, format!("Write failed: {}", e))
            })?;
            return Ok(());
        }
        std::slice::from_raw_parts(buf.ptr as *const u8, buf.size)
    };

    // Convert to string to check if it's a pointer
    let pointer_str = std::str::from_utf8(pointer_data).map_err(|e| {
        XetError::new(
            XetErrorKind::InvalidPointer,
            format!("Pointer data is not valid UTF-8: {}", e),
        )
    })?;

    // Check if it's actually a pointer
    if !payload.storage.is_pointer(pointer_str) {
        // Not a pointer, pass through as-is
        tracing::debug!("XET smudge {}: not a pointer, passing through", path);
        let mut output_wrapper = unsafe { GitWriteStream::from_raw(output) };
        output_wrapper.write(pointer_data).map_err(|e| {
            XetError::new(XetErrorKind::IoError, format!("Write failed: {}", e))
        })?;
        return Ok(());
    }

    // Download from XET CAS
    let content = payload
        .runtime
        .block_on(payload.storage.smudge_pointer(pointer_str))??;

    tracing::debug!(
        "XET smudge {}: pointer -> {} bytes",
        path,
        content.len()
    );

    // Write content to output
    let mut output_wrapper = unsafe { GitWriteStream::from_raw(output) };
    output_wrapper
        .write(&content)
        .map_err(|e| XetError::new(XetErrorKind::IoError, format!("Write failed: {}", e)))?;

    Ok(())
}
