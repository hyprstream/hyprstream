//! C callback implementations for libgit2 filter
//!
//! Implements a streaming filter using git_writestream.
//! Data flows: source -> our writestream -> process -> next writestream

#[cfg(feature = "xet-storage")]
use dashmap::DashMap;

#[cfg(feature = "xet-storage")]
use once_cell::sync::Lazy;

#[cfg(feature = "xet-storage")]
use std::sync::Arc;

#[cfg(feature = "xet-storage")]
use std::ffi::CStr;

#[cfg(feature = "xet-storage")]
use libc::{c_char, c_int, size_t};

#[cfg(feature = "xet-storage")]
use crate::ffi::*;

#[cfg(feature = "xet-storage")]
use crate::filter::XetFilterPayload;

#[cfg(feature = "xet-storage")]
use crate::error::{XetError, XetErrorKind};

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

// ============================================================================
// Custom Writestream Implementation
// ============================================================================

/// XET writestream - buffers data and processes on close
#[cfg(feature = "xet-storage")]
#[repr(C)]
struct XetWriteStream {
    /// Base writestream (must be first field for C compatibility)
    base: GitWriteStreamBase,
    /// Buffered input data
    buffer: Vec<u8>,
    /// Next stream to write to
    next: *mut OpaqueGitWriteStream,
    /// Filter mode (clean or smudge)
    mode: c_int,
    /// File path being filtered
    path: String,
    /// Payload with storage backend
    payload: Arc<XetFilterPayload>,
}

/// Write callback - buffers incoming data
#[cfg(feature = "xet-storage")]
extern "C" fn xet_stream_write(
    stream: *mut GitWriteStreamBase,
    buffer: *const c_char,
    len: size_t,
) -> c_int {
    // SAFETY: stream is our XetWriteStream, buffer is valid for len bytes
    unsafe {
        let xet_stream = stream as *mut XetWriteStream;
        if xet_stream.is_null() {
            return -1;
        }

        let data = std::slice::from_raw_parts(buffer as *const u8, len);
        (*xet_stream).buffer.extend_from_slice(data);
    }
    0
}

/// Close callback - processes buffered data and writes to next stream
#[cfg(feature = "xet-storage")]
extern "C" fn xet_stream_close(stream: *mut GitWriteStreamBase) -> c_int {
    // SAFETY: stream is our XetWriteStream
    let result = unsafe {
        let xet_stream = stream as *mut XetWriteStream;
        if xet_stream.is_null() {
            return -1;
        }

        let stream_ref = &mut *xet_stream;
        process_and_write(stream_ref)
    };

    match result {
        Ok(()) => 0,
        Err(e) => {
            tracing::error!("XET stream close error: {}", e);
            -1
        }
    }
}

/// Process buffered data and write to next stream
#[cfg(feature = "xet-storage")]
fn process_and_write(stream: &mut XetWriteStream) -> Result<(), XetError> {
    let input = &stream.buffer;

    // Process based on mode
    let output = if stream.mode == GIT_FILTER_SMUDGE {
        // Smudge: XET pointer -> file content
        smudge_data(&stream.payload, input, &stream.path)?
    } else {
        // Clean: file content -> XET pointer
        clean_data(&stream.payload, input, &stream.path)?
    };

    // Write output to next stream
    write_to_next(stream.next, &output)?;

    // Close next stream
    close_next(stream.next)?;

    Ok(())
}

/// Smudge: convert XET pointer to file content
#[cfg(feature = "xet-storage")]
fn smudge_data(payload: &XetFilterPayload, input: &[u8], path: &str) -> Result<Vec<u8>, XetError> {
    // Convert to string to check if it's a pointer
    let pointer_str = match std::str::from_utf8(input) {
        Ok(s) => s,
        Err(_) => {
            // Not valid UTF-8, pass through as-is
            tracing::debug!("XET smudge {}: not UTF-8, passing through", path);
            return Ok(input.to_vec());
        }
    };

    // Check if it's actually a pointer
    if !payload.storage.is_pointer(pointer_str) {
        // Not a pointer, pass through as-is
        tracing::debug!("XET smudge {}: not a pointer, passing through", path);
        return Ok(input.to_vec());
    }

    // Download from XET CAS
    let content = payload
        .runtime
        .block_on(payload.storage.smudge_pointer(pointer_str))
        .map_err(|e| XetError::new(XetErrorKind::RuntimeError, format!("Runtime error: {}", e)))?
        .map_err(|e| XetError::new(XetErrorKind::DownloadFailed, format!("Download failed: {}", e)))?;

    tracing::debug!("XET smudge {}: pointer -> {} bytes", path, content.len());
    Ok(content)
}

/// Clean: convert file content to XET pointer
#[cfg(feature = "xet-storage")]
fn clean_data(payload: &XetFilterPayload, input: &[u8], path: &str) -> Result<Vec<u8>, XetError> {
    if input.is_empty() {
        // Empty file - pass through
        return Ok(Vec::new());
    }

    // Write to temporary file for upload
    let temp_file = tempfile::NamedTempFile::new().map_err(|e| {
        XetError::new(XetErrorKind::IoError, format!("Failed to create temp file: {}", e))
    })?;

    std::fs::write(temp_file.path(), input).map_err(|e| {
        XetError::new(XetErrorKind::IoError, format!("Failed to write temp file: {}", e))
    })?;

    // Upload to XET CAS
    let pointer = payload
        .runtime
        .block_on(payload.storage.clean_file(temp_file.path()))
        .map_err(|e| XetError::new(XetErrorKind::RuntimeError, format!("Runtime error: {}", e)))?
        .map_err(|e| XetError::new(XetErrorKind::UploadFailed, format!("Upload failed: {}", e)))?;

    tracing::debug!("XET clean {}: {} bytes -> pointer", path, input.len());
    Ok(pointer.into_bytes())
}

/// Write data to next stream using the writestream vtable
#[cfg(feature = "xet-storage")]
fn write_to_next(next: *mut OpaqueGitWriteStream, data: &[u8]) -> Result<(), XetError> {
    if next.is_null() || data.is_empty() {
        return Ok(());
    }

    // SAFETY: next is a valid git_writestream from libgit2
    // The git_writestream struct has write/close/free function pointers
    // We need to call the write function through the vtable
    unsafe {
        // Cast to our base struct to access vtable
        let ws = next as *mut GitWriteStreamBase;
        let write_fn = (*ws).write;
        let result = write_fn(ws, data.as_ptr() as *const c_char, data.len());
        if result < 0 {
            return Err(XetError::new(XetErrorKind::IoError, "Write to next stream failed"));
        }
    }
    Ok(())
}

/// Close the next stream
#[cfg(feature = "xet-storage")]
fn close_next(next: *mut OpaqueGitWriteStream) -> Result<(), XetError> {
    if next.is_null() {
        return Ok(());
    }

    // SAFETY: next is a valid git_writestream from libgit2
    unsafe {
        let ws = next as *mut GitWriteStreamBase;
        let close_fn = (*ws).close;
        let result = close_fn(ws);
        if result < 0 {
            return Err(XetError::new(XetErrorKind::IoError, "Close next stream failed"));
        }
    }
    Ok(())
}

/// Free callback - deallocates the stream
#[cfg(feature = "xet-storage")]
extern "C" fn xet_stream_free(stream: *mut GitWriteStreamBase) {
    if stream.is_null() {
        return;
    }

    // SAFETY: stream was allocated by Box::into_raw in xet_filter_stream
    unsafe {
        let xet_stream = stream as *mut XetWriteStream;
        drop(Box::from_raw(xet_stream));
    }
}

/// Stream callback (creates our custom writestream)
///
/// # Safety
/// This function is called by libgit2 with valid pointers. All pointer parameters
/// must be valid for the duration of this call. This is guaranteed by libgit2's
/// callback contract. The raw pointer dereferences are safe within this context.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[cfg(feature = "xet-storage")]
pub extern "C" fn xet_filter_stream(
    out: *mut *mut OpaqueGitWriteStream,
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
            tracing::error!("XET filter: null path");
            return -1;
        }
        match CStr::from_ptr(path_ptr).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                tracing::error!("XET filter: invalid UTF-8 in path");
                return -1;
            }
        }
    };

    // Get payload using filter instance pointer
    let payload = match get_payload(filter) {
        Some(p) => p,
        None => {
            tracing::error!("XET storage not initialized for filter {:p}", filter);
            return -1;
        }
    };

    tracing::debug!(
        "XET filter stream: mode={}, path={}",
        if mode == GIT_FILTER_SMUDGE { "smudge" } else { "clean" },
        path
    );

    // Create our custom writestream
    let xet_stream = Box::new(XetWriteStream {
        base: GitWriteStreamBase {
            write: xet_stream_write,
            close: xet_stream_close,
            free: xet_stream_free,
        },
        buffer: Vec::new(),
        next,
        mode,
        path,
        payload,
    });

    // Convert to raw pointer and return
    // SAFETY: Box::into_raw creates a stable pointer
    // libgit2 will call free() when done
    unsafe {
        *out = Box::into_raw(xet_stream) as *mut OpaqueGitWriteStream;
    }

    0
}

/// Cleanup callback (releases per-source resources)
#[cfg(feature = "xet-storage")]
pub extern "C" fn xet_filter_cleanup(_filter: *mut OpaqueGitFilter, _payload: *mut libc::c_void) {
    // Currently we don't use per-source payload
}
