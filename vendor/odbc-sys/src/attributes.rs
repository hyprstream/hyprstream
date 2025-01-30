use crate::Pointer;

/// Governs behaviour of EnvironmentAttribute
#[repr(i32)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum EnvironmentAttribute {
    OdbcVersion = 200,
    ConnectionPooling = 201,
    CpMatch = 202,
    // This attribute was commented out because there is no mention of it in the ODBC specification
    // nor does this attribute exist in unixODBC or iODBC implementations. This attribute exists in
    // Microsoft implementation only and it's usage is unclear.
    // For private driver manager
    // SQL_ATTR_APPLICATION_KEY = 203,
    OutputNts = 10001,
}

/// ODBC verions
///
/// Possible values for `OdbcVersion` attribute set with `SQLSetEnvAttr` to declare ODBC version
#[repr(i32)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum AttrOdbcVersion {
    // Not supported by this crate
    // SQL_OV_ODBC2 = 2,
    Odbc3 = 3,
    #[cfg(feature = "odbc_version_3_80")]
    Odbc3_80 = 380,
    #[cfg(feature = "odbc_version_4")]
    Odbc4 = 400,
}

impl From<AttrOdbcVersion> for Pointer {
    fn from(source: AttrOdbcVersion) -> Pointer {
        source as i32 as Pointer
    }
}
/// Connection pool configuration
///
/// Possible values for `ConnectionPooling` attribute set with `SQLSetEnvAttr` to define which
/// pooling scheme will be used.
///
/// See: <https://docs.microsoft.com/en-us/sql/odbc/reference/syntax/sqlsetenvattr-function>
#[repr(u32)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum AttrConnectionPooling {
    /// Connection pooling is turned off. This is the default.
    Off = 0,
    /// A single connection pool is supported for each driver. Every connection in a pool is
    /// associated with one driver.
    OnePerDriver = 1,
    /// A single connection pool is supported for each environment. Every connection in a pool is
    /// associated with one environment.
    OnePerHenv = 2,
    /// Use the connection-pool awareness feature of the driver, if it is available. If the driver
    /// does not support connection-pool awareness, `DriverAware` is ignored and `OnePerHenv` is
    /// used.
    DriverAware = 3,
}

/// Connection pool default configuration
impl Default for AttrConnectionPooling {
    fn default() -> Self {
        AttrConnectionPooling::Off
    }
}

impl From<AttrConnectionPooling> for Pointer {
    fn from(source: AttrConnectionPooling) -> Pointer {
        source as u32 as Pointer
    }
}

/// Determines how a connection is chosen from a connection pool.
///
/// Possible values for `CpMatch` attribute set with [`crate::SQLSetEnvAttr`] to define which connection
/// attributes must match for a connection returned from the pool
#[repr(u32)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum AttrCpMatch {
    /// Only connections that exactly match the connection options in the call and the connection
    /// attributes set by the application are reused. This is the default.
    Strict = 0,
    /// Connections with matching connection string keywords can be used. Keywords must match, but
    /// not all connection attributes must match.
    Relaxed = 1,
}

/// Default matching for connections returned from the pool
impl Default for AttrCpMatch {
    fn default() -> Self {
        AttrCpMatch::Strict
    }
}

impl From<AttrCpMatch> for Pointer {
    fn from(source: AttrCpMatch) -> Pointer {
        source as u32 as Pointer
    }
}

/// Statement attributes are characteristics of the statement. For example, whether to use bookmarks
/// and what kind of cursor to use with the statement's result set are statement attributes.
///
/// Statement attributes are set with `SQLSetStmtAttr` and their current settings retrieved with
/// `SQLGetStmtAttr`. There is no requirement that an application set any statement attributes; all
/// statement attributes have defaults, some of which are driver-specific.
/// When a statement attribute can be set depends on the attribute itself. The
/// `Concurrency`, `CursorType, `SimulateCursor`, and `UseBookmars` statement attributes must be set
/// before the statement is executed. The `AsyncEnable` and `NoScan` statement attributes can be set
/// at any time but are not applied until the statement is used again. `MaxLength`, `MaxRows`, and
/// `QueryTimeout` statement attributes can be set at any time, but it is driver-specific whether
/// they are applied before the statement is used again. The remaining statement attributes can be
/// set at any time.
#[repr(i32)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum StatementAttribute {
    /// SQL_ATTR_APP_ROW_DESC
    AppRowDesc = 10010,
    /// SQL_ATTR_APP_PARAM_DESC
    AppParamDesc = 10011,
    /// SQL_ATTR_IMP_ROW_DESC
    ImpRowDesc = 10012,
    /// SQL_ATTR_IMP_PARAM_DESC
    ImpParamDesc = 10013,
    /// SQL_ATTR_CURSOR_SCROLLABLE
    CursorScrollable = -1,
    /// SQL_ATTR_CURSOR_SENSITIVITY
    CursorSensitivity = -2,

    // Extensions
    /// SQL_ATTR_ASYNC_ENABLE
    AsyncEnable = 4,
    /// SQL_ATTR_CONCURRENCY
    Concurrency = 7,
    /// SQL_ATTR_CURSOR_TYPE
    CursorType = 6,
    /// SQL_ATTR_ENABLE_AUTO_IPD
    EnableAutoIpd = 15,
    /// SQL_ATTR_FETCH_BOOKMARK_PTR
    FetchBookmarkPtr = 16,
    /// SQL_ATTR_KEYSET_SIZE
    KeysetSize = 8,
    /// SQL_ATTR_MAX_LENGTH
    MaxLength = 3,
    /// SQL_ATTR_MAX_ROWS
    MaxRows = 1,
    /// SQL_ATTR_NOSCAN
    NoScan = 2,
    /// SQL_ATTR_PARAM_BIND_OFFSET_PTR
    ParamBindOffsetPtr = 17,
    /// SQL_ATTR_PARAM_BIND_TYPE
    ParamBindType = 18,
    /// SQL_ATTR_PARAM_OPERATION_PTR
    ParamOpterationPtr = 19,
    /// SQL_ATTR_PARAM_STATUS_PTR
    ParamStatusPtr = 20,
    /// SQL_ATTR_PARAMS_PROCESSED_PTR
    ParamsProcessedPtr = 21,
    // SQL_ATTR_PARAMSET_SIZE
    ParamsetSize = 22,
    /// SQL_ATTR_QUERY_TIMEOUT
    QueryTimeout = 0,
    /// SQL_ATTR_RETRIEVE_DATA
    RetrieveData = 11,
    /// SQL_ATTR_ROW_BIND_OFFSET_PTR
    RowBindOffsetPtr = 23,
    /// SQL_ATTR_ROW_BIND_TYPE
    RowBindType = 5,
    /// SQL_ATTR_ROW_NUMBER `GetStmtAttr`
    RowNumber = 14,
    /// SQL_ATTR_ROW_OPERATION_PTR
    RowOperationPtr = 24,
    /// SQL_ATTR_ROW_STATUS_PTR
    RowStatusPtr = 25,
    /// SQL_ATTR_ROWS_FETCHED_PTR
    RowsFetchedPtr = 26,
    /// SQL_ATTR_ROW_ARRAY_SIZE
    RowArraySize = 27,
    /// SQL_ATTR_SIMULATE_CURSOR
    SimulateCursor = 10,
    /// SQL_ATTR_USE_BOOKMARKS
    UseBookmarks = 12,
    #[cfg(feature = "odbc_version_3_80")]
    /// SQL_ATTR_ASYNC_STMT_EVENT
    AsyncStmtEvent = 29,
    #[cfg(feature = "odbc_version_4")]
    /// SQL_ATTR_SAMPLE_SIZE
    SampleSize = 30,
    #[cfg(feature = "odbc_version_4")]
    /// SQL_ATTR_DYNAMIC_COLUMNS
    DynamicColumns = 31,
    #[cfg(feature = "odbc_version_4")]
    /// SQL_ATTR_TYPE_EXCEPTION_BEHAVIOR
    TypeExceptionBehaviour = 32,
    #[cfg(feature = "odbc_version_4")]
    /// SQL_ATTR_LENGTH_EXCEPTION_BEHAVIOR
    LengthExceptionBehaviour = 33,
    /// SQL_ATTR_METADATA_ID
    MetadataId = 10014,
}
