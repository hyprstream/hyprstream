/// Indicates the overall success or failure of the function
///
/// Each function in ODBC returns a code, known as its return code, which indicates the overall
/// success or failure of the function. Program logic is generally based on return codes.
/// See [ODBC reference](https://docs.microsoft.com/en-us/sql/odbc/reference/develop-app/return-codes-odbc)
#[must_use]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[repr(C)]
pub struct SqlReturn(pub i16);

impl SqlReturn {
    /// `SQL_INVALID_HANDLE`; Function failed due to an invalid environment, connection, statement,
    /// or descriptor handle.
    ///
    /// This indicates a programming error. No additional information is available from
    /// `SQLGetDiagRec` or `SQLGetDiagField`. This code is returned only when the handle is a null
    /// pointer or is the wrong type, such as when a statement handle is passed for an argument a
    /// connection handle.
    pub const INVALID_HANDLE: SqlReturn = SqlReturn(-2);

    /// Function failed
    ///
    /// The application calls `SQLGetDiagRec` or `SQLGetDiagField` to retrieve additional
    /// information. The contents of any output arguments to the function are undefined.
    pub const ERROR: SqlReturn = SqlReturn(-1);

    /// Function completed successfully
    ///
    /// The application calls `SQLGetDiagField` to retrieve additional information from the header
    /// record.
    pub const SUCCESS: SqlReturn = SqlReturn(0);

    /// Function completed successfully, possibly with a nonfatal error (warning)
    ///
    /// The application calls `SQLGetDiagRec` or `SQLGetDiagField` to retrieve additional
    /// information.
    pub const SUCCESS_WITH_INFO: SqlReturn = SqlReturn(1);

    /// A function that was started asynchronously is still executing
    ///
    /// The application `SQLGetDiagRec` or `SQLGetDiagField` to retrieve additional information if
    /// any.
    pub const STILL_EXECUTING: SqlReturn = SqlReturn(2);

    /// More data is needed
    ///
    /// ,such as when a parameter data is sent at execution time or additional connection
    /// information is required. The application calls `SQLGetDiagRec` or `SQLGetDiagField` to
    /// retrieve additional information, if any.
    pub const NEED_DATA: SqlReturn = SqlReturn(99);

    /// No more data was available
    ///
    /// The application calls `SQLGetDiagRec` or `SQLGetDiagField` to retrieve additional
    /// information. One or more driver-defined status records in class 02xxx may be returned.
    pub const NO_DATA: SqlReturn = SqlReturn(100);

    #[cfg(feature = "odbc_version_3_80")]
    pub const PARAM_DATA_AVAILABLE: SqlReturn = SqlReturn(101);
}
