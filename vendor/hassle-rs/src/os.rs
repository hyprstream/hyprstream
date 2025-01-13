// Allow uppercase names to match Windows API:
#![allow(clippy::upper_case_acronyms)]

#[cfg(windows)]
mod os_defs {
    pub use winapi::shared::{
        ntdef::{HRESULT, LPCSTR, LPCWSTR, LPSTR, LPWSTR, WCHAR},
        wtypes::BSTR,
    };

    pub use winapi::um::combaseapi::CoTaskMemFree;
    pub use winapi::um::oleauto::{SysFreeString, SysStringLen};
}

#[cfg(not(windows))]
mod os_defs {
    pub type CHAR = std::os::raw::c_char;
    pub type UINT = u32;
    pub type WCHAR = widestring::WideChar;
    pub type OLECHAR = WCHAR;
    pub type LPSTR = *mut CHAR;
    pub type LPWSTR = *mut WCHAR;
    pub type LPCSTR = *const CHAR;
    pub type LPCWSTR = *const WCHAR;
    pub type BSTR = *mut OLECHAR;
    pub type LPBSTR = *mut BSTR;
    pub type HRESULT = i32;

    /// Returns a mutable pointer to the length prefix of the string
    fn len_ptr(p: BSTR) -> *mut UINT {
        // The first four bytes before the pointer contain the length prefix:
        // https://docs.microsoft.com/en-us/previous-versions/windows/desktop/automat/bstr#remarks
        unsafe { p.cast::<UINT>().offset(-1) }
    }

    #[allow(non_snake_case)]
    /// # Safety
    /// `p` must be a valid pointer to an allocation made with `malloc`, or null.
    pub unsafe fn CoTaskMemFree(p: *mut libc::c_void) {
        // https://github.com/microsoft/DirectXShaderCompiler/blob/56e22b30c5e43632f56a1f97865f37108ec35463/include/dxc/Support/WinAdapter.h#L46
        if !p.is_null() {
            libc::free(p)
        }
    }

    #[allow(non_snake_case)]
    /// # Safety
    /// `p` must be a valid pointer to an allocation made with `malloc`, or null.
    pub unsafe fn SysFreeString(p: BSTR) {
        // https://github.com/microsoft/DirectXShaderCompiler/blob/56e22b30c5e43632f56a1f97865f37108ec35463/lib/DxcSupport/WinAdapter.cpp#L50-L53
        if !p.is_null() {
            libc::free(len_ptr(p).cast::<_>())
        }
    }

    #[allow(non_snake_case)]
    /// Returns the size of `p` in bytes, excluding terminating NULL character
    ///
    /// # Safety
    /// `p` must be a valid pointer to a [`BSTR`] with size-prefix in the `4` leading bytes, or null.
    ///
    /// https://docs.microsoft.com/en-us/previous-versions/windows/desktop/automat/bstr#remarks
    pub unsafe fn SysStringByteLen(p: BSTR) -> UINT {
        if p.is_null() {
            0
        } else {
            *len_ptr(p)
        }
    }

    #[allow(non_snake_case)]
    /// Returns the size of `p` in characters, excluding terminating NULL character
    ///
    /// # Safety
    /// `p` must be a valid pointer to a [`BSTR`] with size-prefix in the `4` leading bytes, or null.
    ///
    /// https://docs.microsoft.com/en-us/previous-versions/windows/desktop/automat/bstr#remarks
    pub unsafe fn SysStringLen(p: BSTR) -> UINT {
        SysStringByteLen(p) / std::mem::size_of::<OLECHAR>() as UINT
    }
}

pub use os_defs::*;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
#[must_use]
pub struct HRESULT(pub os_defs::HRESULT);
impl HRESULT {
    pub fn is_err(&self) -> bool {
        self.0 < 0
    }
}

impl From<i32> for HRESULT {
    fn from(v: i32) -> Self {
        Self(v)
    }
}

impl std::fmt::Debug for HRESULT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as std::fmt::Display>::fmt(self, f)
    }
}

impl std::fmt::Display for HRESULT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:#x}", self))
    }
}

impl std::fmt::LowerHex for HRESULT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = if f.alternate() { "0x" } else { "" };
        let bare_hex = format!("{:x}", self.0.abs());
        // https://stackoverflow.com/a/44712309
        f.pad_integral(self.0 >= 0, prefix, &bare_hex)
        // <i32 as std::fmt::LowerHex>::fmt(&self.0, f)
    }
}
