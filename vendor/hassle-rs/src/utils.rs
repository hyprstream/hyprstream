use std::ffi::CStr;
use std::path::PathBuf;

use crate::os::{SysFreeString, SysStringLen, BSTR, HRESULT, LPCSTR, LPCWSTR, WCHAR};
use crate::wrapper::*;
use thiserror::Error;

pub(crate) fn to_wide(msg: &str) -> Vec<WCHAR> {
    widestring::WideCString::from_str(msg)
        .unwrap()
        .into_vec_with_nul()
}

pub(crate) fn from_wide(wide: LPCWSTR) -> String {
    unsafe { widestring::WideCStr::from_ptr_str(wide) }
        .to_string()
        .expect("widestring decode failed")
}

pub(crate) fn from_bstr(string: BSTR) -> String {
    let len = unsafe { SysStringLen(string) } as usize;

    let result = unsafe { widestring::WideStr::from_ptr(string, len) }
        .to_string()
        .expect("widestring decode failed");

    unsafe { SysFreeString(string) };
    result
}

pub(crate) fn from_lpstr(string: LPCSTR) -> String {
    unsafe { CStr::from_ptr(string) }
        .to_str()
        .unwrap()
        .to_owned()
}

struct DefaultIncludeHandler {}

impl DxcIncludeHandler for DefaultIncludeHandler {
    fn load_source(&mut self, filename: String) -> Option<String> {
        use std::io::Read;
        match std::fs::File::open(filename) {
            Ok(mut f) => {
                let mut content = String::new();
                f.read_to_string(&mut content).ok()?;
                Some(content)
            }
            Err(_) => None,
        }
    }
}

#[derive(Error, Debug)]
pub enum HassleError {
    #[error("Win32 error: {0:x}")]
    Win32Error(HRESULT),
    #[error("{0}")]
    CompileError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Failed to load library {filename:?}: {inner:?}")]
    LoadLibraryError {
        filename: PathBuf,
        #[source]
        inner: libloading::Error,
    },
    #[error("LibLoading error: {0:?}")]
    LibLoadingError(#[from] libloading::Error),
    #[error("Windows only")]
    WindowsOnly(String),
}

pub type Result<T, E = HassleError> = std::result::Result<T, E>;

impl HRESULT {
    /// Turns an [`HRESULT`] from the COM [`crate::ffi`] API declaration
    /// into a [`Result`] containing [`HassleError`].
    pub fn result(self) -> Result<()> {
        self.result_with_success(())
    }

    /// Turns an [`HRESULT`] from the COM [`crate::ffi`] API declaration
    /// into a [`Result`] containing [`HassleError`], with the desired value.
    ///
    /// Note that `v` is passed by value and is not a closure that is executed
    /// lazily.  Use the short-circuiting `?` operator for such cases:
    /// ```no_run
    /// let mut blob: ComPtr<IDxcBlob> = ComPtr::new();
    /// unsafe { self.inner.get_result(blob.as_mut_ptr()) }.result()?;
    /// Ok(DxcBlob::new(blob))
    /// ```
    pub fn result_with_success<T>(self, v: T) -> Result<T> {
        if self.is_err() {
            Err(HassleError::Win32Error(self))
        } else {
            Ok(v)
        }
    }
}

/// Helper function to directly compile a HLSL shader to an intermediate language,
/// this function expects `dxcompiler.dll` to be available in the current
/// executable environment.
///
/// Specify -spirv as one of the `args` to compile to SPIR-V
/// `dxc_path` can point to a library directly or the directory containing the library,
/// in which case the appended filename depends on the platform.
pub fn compile_hlsl(
    source_name: &str,
    shader_text: &str,
    entry_point: &str,
    target_profile: &str,
    args: &[&str],
    defines: &[(&str, Option<&str>)],
) -> Result<Vec<u8>> {
    let dxc = Dxc::new(None)?;

    let compiler = dxc.create_compiler()?;
    let library = dxc.create_library()?;

    let blob = library.create_blob_with_encoding_from_str(shader_text)?;

    let result = compiler.compile(
        &blob,
        source_name,
        entry_point,
        target_profile,
        args,
        Some(&mut DefaultIncludeHandler {}),
        defines,
    );

    match result {
        Err(result) => {
            let error_blob = result.0.get_error_buffer()?;
            Err(HassleError::CompileError(
                library.get_blob_as_string(&error_blob.into())?,
            ))
        }
        Ok(result) => {
            let result_blob = result.get_result()?;

            Ok(result_blob.to_vec())
        }
    }
}

/// Helper function to validate a DXIL binary independent from the compilation process,
/// this function expects `dxcompiler.dll` and `dxil.dll` to be available in the current
/// execution environment.
///
/// `dxil.dll` is only available on Windows.
pub fn validate_dxil(data: &[u8]) -> Result<Vec<u8>, HassleError> {
    let dxc = Dxc::new(None)?;
    let dxil = Dxil::new(None)?;

    let validator = dxil.create_validator()?;
    let library = dxc.create_library()?;

    let blob_encoding = library.create_blob_with_encoding(data)?;

    match validator.validate(blob_encoding.into()) {
        Ok(blob) => Ok(blob.to_vec()),
        Err(result) => {
            let error_blob = result.0.get_error_buffer()?;
            Err(HassleError::ValidationError(
                library.get_blob_as_string(&error_blob.into())?,
            ))
        }
    }
}

pub use crate::fake_sign::fake_sign_dxil_in_place;
