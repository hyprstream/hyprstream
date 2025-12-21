//! XET filter implementation with type state pattern

use std::ffi::{CStr, CString};

#[cfg(feature = "xet-storage")]
use std::marker::PhantomData;

#[cfg(feature = "xet-storage")]
use std::pin::Pin;

#[cfg(feature = "xet-storage")]
use std::sync::Arc;

#[cfg(feature = "xet-storage")]
use libc::c_char;

#[cfg(feature = "xet-storage")]
use crate::ffi::GitFilter;

#[cfg(feature = "xet-storage")]
use crate::storage::{StorageBackend, XetStorage};

#[cfg(feature = "xet-storage")]
use crate::runtime::XetRuntime;

#[cfg(feature = "xet-storage")]
use crate::error::{Result, XetError, XetErrorKind};

// Type state markers
#[cfg(feature = "xet-storage")]
pub struct Unregistered;

#[cfg(feature = "xet-storage")]
pub struct Registered;

/// XET filter with type state
#[cfg(feature = "xet-storage")]
pub struct XetFilter<State = Unregistered> {
    inner: Pin<Box<GitFilter>>,
    payload: Option<Box<XetFilterPayload>>,
    name: String,
    attributes_cstr: Option<CString>,
    _state: PhantomData<State>,
}

/// Filter payload stored in libgit2's filter->payload field
#[cfg(feature = "xet-storage")]
#[derive(Clone)]
pub struct XetFilterPayload {
    pub storage: Arc<dyn StorageBackend>,
    pub runtime: Arc<XetRuntime>,
}

#[cfg(feature = "xet-storage")]
impl XetFilter<Unregistered> {
    /// Create a new unregistered filter
    pub async fn new(config: crate::config::XetConfig) -> Result<Self> {
        // Create runtime in a separate thread to avoid nested runtime errors
        // We use std::thread::spawn instead of tokio::spawn_blocking because
        // tokio's blocking thread pool has access to the runtime handle
        let (runtime, storage) = std::thread::spawn(move || {
            let runtime = XetRuntime::new()?;
            let storage = runtime.block_on_unchecked(async { XetStorage::new(&config).await })?;
            Ok::<_, XetError>((Arc::new(runtime), Arc::new(storage) as Arc<dyn StorageBackend>))
        })
        .join()
        .map_err(|e| XetError::new(XetErrorKind::RuntimeError, format!("Thread join error: {:?}", e)))??;

        // Create payload
        let payload = Box::new(XetFilterPayload { storage, runtime });

        // Creating a CString for attributes
        // Format: "filter=xet" matches .gitattributes entries like "*.safetensors filter=xet"
        let attributes_cstr = CString::new("filter=xet").unwrap();
        let attributes_ptr = attributes_cstr.as_ptr() as *const c_char;

        // Create filter structure
        let filter = Box::pin(GitFilter {
            version: crate::ffi::GIT_FILTER_VERSION,
            attributes: attributes_ptr,
            initialize: Some(crate::callbacks::xet_filter_initialize),
            shutdown: Some(crate::callbacks::xet_filter_shutdown),
            check: Some(crate::callbacks::xet_filter_check),
            reserved: std::ptr::null_mut(),
            stream: Some(crate::callbacks::xet_filter_stream),
            cleanup: Some(crate::callbacks::xet_filter_cleanup),
        });

        Ok(Self {
            inner: filter,
            payload: Some(payload),
            name: "xet".to_string(),
            attributes_cstr: Some(attributes_cstr),
            _state: PhantomData,
        })
    }

    /// Register this filter with libgit2
    ///
    /// Transitions to Registered state on success
    pub fn register(mut self, priority: i32) -> Result<XetFilter<Registered>> {
        // Ensure libgit2 is fully initialized
        // Calling any git2 function triggers initialization
        unsafe { git2::opts::set_mwindow_size(8 * 1024 * 1024).ok() };

        let name_cstr = CString::new(self.name.as_str())
            .map_err(|_| XetError::new(XetErrorKind::RuntimeError, "Invalid filter name"))?;

        // SAFETY: We're passing a stable pointer to a Pin<Box<GitFilter>>
        // which will not move. The payload is also boxed and won't move.
        unsafe {
            // Try to unregister first in case it's already registered
            let unregister_result = crate::ffi::git_filter_unregister(name_cstr.as_ptr());
            if unregister_result == 0 {
                tracing::debug!("Unregistered existing '{}' filter before re-registering", self.name);
            }

            let filter_ptr = Pin::as_mut(&mut self.inner).get_unchecked_mut() as *mut GitFilter;

            // Initialize the filter structure with libgit2's expected defaults
            let init_result = crate::ffi::git_filter_init(filter_ptr, crate::ffi::GIT_FILTER_VERSION);
            if init_result < 0 {
                tracing::warn!("git_filter_init returned {} (continuing anyway)", init_result);
            }

            // Restore our callbacks after git_filter_init (it may have cleared them)
            (*filter_ptr).attributes = self.attributes_cstr.as_ref().unwrap().as_ptr();
            (*filter_ptr).initialize = Some(crate::callbacks::xet_filter_initialize);
            (*filter_ptr).shutdown = Some(crate::callbacks::xet_filter_shutdown);
            (*filter_ptr).check = Some(crate::callbacks::xet_filter_check);
            (*filter_ptr).stream = Some(crate::callbacks::xet_filter_stream);
            (*filter_ptr).cleanup = Some(crate::callbacks::xet_filter_cleanup);

            // Store payload in global registry using filter instance pointer
            // Cast to OpaqueGitFilter for the registry
            crate::callbacks::register_payload(
                filter_ptr as *const crate::ffi::OpaqueGitFilter,
                *self.payload.take().unwrap(),
            );

            let result = crate::ffi::git_filter_register(
                name_cstr.as_ptr(),
                filter_ptr,
                priority,
            );

            if result < 0 {
                // Get detailed error from libgit2
                let git_error = git2::Error::last_error(result);

                // Check if it's GIT_EEXISTS (-4) which means filter already registered
                // In that case, we'll consider it a success
                const GIT_EEXISTS: i32 = -4;
                if result == GIT_EEXISTS {
                    tracing::warn!("XET filter was already registered, continuing anyway");
                    // Don't clean up payload - keep it registered
                } else {
                    // Clean up payload on other failures
                    let _ = crate::callbacks::unregister_payload(
                        filter_ptr as *const crate::ffi::OpaqueGitFilter
                    );

                    // Log for debugging
                    tracing::error!(
                        "git_filter_register failed: result={}, error={:?}, version={}, attributes={:?}",
                        result,
                        git_error,
                        (*filter_ptr).version,
                        CStr::from_ptr((*filter_ptr).attributes)
                    );

                    return Err(XetError::new(
                        XetErrorKind::RuntimeError,
                        format!("git_filter_register failed (code {}): {:?}", result, git_error),
                    ));
                }
            }
        }

        // Transition to Registered state
        Ok(XetFilter {
            inner: self.inner,
            payload: None, // Ownership transferred to global registry
            name: self.name,
            attributes_cstr: self.attributes_cstr,
            _state: PhantomData,
        })
    }
}

#[cfg(feature = "xet-storage")]
impl XetFilter<Registered> {
    /// Unregister this filter
    ///
    /// Transitions back to Unregistered state
    pub fn unregister(mut self) -> Result<XetFilter<Unregistered>> {
        let name_cstr = CString::new(self.name.as_str())
            .map_err(|_| XetError::new(XetErrorKind::RuntimeError, "Invalid filter name"))?;

        unsafe {
            let filter_ptr = Pin::as_mut(&mut self.inner).get_unchecked_mut() as *mut GitFilter;

            let result = crate::ffi::git_filter_unregister(name_cstr.as_ptr());

            if result < 0 {
                return Err(XetError::new(
                    XetErrorKind::RuntimeError,
                    "git_filter_unregister failed",
                ));
            }

            // Retrieve payload from global registry using filter instance pointer
            // Cast to OpaqueGitFilter for the registry
            let payload = crate::callbacks::unregister_payload(
                filter_ptr as *const crate::ffi::OpaqueGitFilter
            ).ok_or_else(|| XetError::new(
                XetErrorKind::RuntimeError,
                "Payload not found in registry - filter may have been double-unregistered"
            ))?;

            // Convert Arc back to Box for ownership transfer
            let payload_box = Box::new((*payload).clone());

            Ok(XetFilter {
                inner: self.inner,
                payload: Some(payload_box),
                name: self.name,
                attributes_cstr: self.attributes_cstr,
                _state: PhantomData,
            })
        }
    }
}

// Safety: Filter can be sent across threads when unregistered
#[cfg(feature = "xet-storage")]
unsafe impl Send for XetFilter<Unregistered> {}

#[cfg(feature = "xet-storage")]
unsafe impl Sync for XetFilter<Unregistered> {}

// Safety: Registered filter is stored in static and never accessed directly
// across threads. The actual filter operations are handled by libgit2's C API.
#[cfg(feature = "xet-storage")]
unsafe impl Send for XetFilter<Registered> {}

#[cfg(feature = "xet-storage")]
unsafe impl Sync for XetFilter<Registered> {}

// Note: No Drop implementation for XetFilter<Registered>
//
// We intentionally do NOT implement Drop for the Registered state because:
//
// 1. **C Library Integration Safety**: libgit2 may still have references to the
//    filter structure. Auto-unregistering on drop could cause use-after-free.
//
// 2. **Explicit Lifecycle**: Users must explicitly call `unregister()` to get back
//    the Unregistered state. This makes the lifecycle predictable and intentional.
//
// 3. **Global Registration**: The filter is registered globally with libgit2, not
//    scoped to this Rust object. Automatic cleanup on drop would be surprising.
//
// 4. **Payload Cleanup**: The payload remains in PAYLOAD_REGISTRY until explicit
//    unregister. If the XetFilter is dropped without unregister, the payload leaks
//    (intentionally) rather than causing a crash.
//
// Users should follow this pattern:
// ```rust
// let filter = XetFilter::new(config).await?;
// let registered = filter.register(100)?;
// // ... use filter ...
// let unregistered = registered.unregister()?; // Explicit cleanup
// ```

