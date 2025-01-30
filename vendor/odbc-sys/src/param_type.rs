/// Used by `SQLBindParameter`.
#[repr(i16)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ParamType {
    Unknown = 0,
    Input = 1,
    InputOutput = 2,
    ResultCol = 3,
    Output = 4,
    ReturnValue = 5,
    #[cfg(feature = "odbc_version_3_80")]
    InputOutputStream = 8,
    #[cfg(feature = "odbc_version_3_80")]
    OutputStream = 16,
}
