/// Information requested by SQLGetInfo
#[repr(u16)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum InfoType {
    MaxDriverConnections = 0,
    MaxConcurrentActivities = 1,
    DataSourceName = 2,
    // FetchDirection = 8, Deprecated in ODBC 3
    ServerName = 13,
    SearchPatternEscape = 14,
    DbmsName = 17,
    DbmsVer = 18,
    AccessibleTables = 19,
    AccessibleProcedures = 20,
    CursorCommitBehaviour = 23,
    DataSourceReadOnly = 25,
    DefaultTxnIsolation = 26,
    IdentifierCase = 28,
    IdentifierQuoteChar = 29,
    MaxColumnNameLen = 30,
    MaxCursorNameLen = 31,
    MaxSchemaNameLen = 32,
    MaxCatalogNameLen = 34,
    MaxTableNameLen = 35,
    // ScrollConcurrency = 43, deprecated in ODBC 3
    /// `SQL_SCROLL_OPTIONS` C-API places this in the extended header. Lists the supported cursor
    /// types (forward-only, static, keyset-driven, dynamic, or mixed). All data sources must
    /// support forward-only cursors. See:
    /// <https://learn.microsoft.com/sql/odbc/reference/develop-app/determining-cursor-capabilities>
    ScrollOptions = 44,
    TransactionCapable = 46,
    UserName = 47,
    TransactionIsolationProtocol = 72,
    Integrity = 73,
    GetDataExtensions = 81,
    NullCollation = 85,
    AlterTable = 86,
    OrderByColumnsInSelect = 90,
    SpecialCharacters = 94,
    MaxColumnsInGroupBy = 97,
    MaxColumnsInIndex = 98,
    MaxColumnsInOrderBy = 99,
    MaxColumnsInSelect = 100,
    MaxColumnsInTable = 101,
    MaxIndexSize = 102,
    MaxRowSize = 104,
    MaxStatementLen = 105,
    MaxTablesInSelect = 106,
    MaxUserNameLen = 107,
    OuterJoinCapabilities = 115,
    /// `SQL_ACTIVE_ENVIRONMENTS` C-API places this in the extended header. A `u16` value that
    /// specifies the maximum number of active environments that the driver can support. If there is
    /// no specified limit or the limit is unknown, this value is set to zero.
    ActiveEnvironments = 116,
    /// `SQL_DYNAMIC_CURSOR_ATTRIBUTES1` C-API places this in the extended header. Lists the fetch
    /// types supported by scrollable cursors. The bits in the return value correspond to the fetch
    /// types in SQLFetchScroll. See:
    /// <https://learn.microsoft.com/sql/odbc/reference/develop-app/determining-cursor-capabilities>
    DynamicCursorAttributes1 = 144,
    /// `SQL_DYNAMIC_CURSOR_ATTRIBUTES2` C-API places this in the extended header. Lists whether
    /// cursors can detect their own updates, deletes, and inserts. See:
    /// <https://learn.microsoft.com/sql/odbc/reference/develop-app/determining-cursor-capabilities>
    DynamicCursorAttributes2 = 145,
    /// `SQL_FORWARD_ONLY_CURSOR_ATTRIBUTES1` C-API places this in the extended header. Lists the
    /// fetch types supported by scrollable cursors. The bits in the return value correspond to the
    /// fetch types in SQLFetchScroll. See:
    /// <https://learn.microsoft.com/sql/odbc/reference/develop-app/determining-cursor-capabilities>
    ForwardOnlyCursorAttributes1 = 146,
    /// `SQL_FORWARD_ONLY_CURSOR_ATTRIBUTES2` C-API places this in the extended header. Lists
    /// whether cursors can detect their own updates, deletes, and inserts. See:
    /// <https://learn.microsoft.com/sql/odbc/reference/develop-app/determining-cursor-capabilities>
    ForwardOnlyCursorAttributes2 = 147,
    /// `SQL_KEYSET_CURSOR_ATTRIBUTES1` C-API places this in the extended header. Lists the fetch
    /// types supported by scrollable cursors. The bits in the return value correspond to the fetch
    /// types in SQLFetchScroll. See:
    /// <https://learn.microsoft.com/sql/odbc/reference/develop-app/determining-cursor-capabilities>
    KeysetCursorAttributes1 = 150,
    /// `SQL_KEYSET_CURSOR_ATTRIBUTES2` C-API places this in the extended header. Lists whether
    /// cursors can detect their own updates, deletes, and inserts. See:
    /// <https://learn.microsoft.com/sql/odbc/reference/develop-app/determining-cursor-capabilities>
    KeysetCursorAttributes2 = 151,
    /// `SQL_STATIC_CURSOR_ATTRIBUTES1` C-API places this in the extended header. Lists the fetch
    /// types supported by scrollable cursors. The bits in the return value correspond to the fetch
    /// types in SQLFetchScroll. See:
    /// <https://learn.microsoft.com/sql/odbc/reference/develop-app/determining-cursor-capabilities>
    StaticCursorAttributes1 = 167,
    /// `SQL_STATIC_CURSOR_ATTRIBUTES2` C-API places this in the extended header. Lists whether
    /// cursors can detect their own updates, deletes, and inserts. See:
    /// <https://learn.microsoft.com/sql/odbc/reference/develop-app/determining-cursor-capabilities>
    StaticCursorAttributes2 = 168,
    XopenCliYear = 10000,
    CursorSensitivity = 10001,
    DescribeParameter = 10002,
    CatalogName = 10003,
    CollationSeq = 10004,
    MaxIdentifierLen = 10005,
    AsyncMode = 10021,
    MaxAsyncConcurrentStatements = 10022,
    AsyncDbcFunctions = 10023,
    DriverAwarePoolingSupported = 10024,
    AsyncNotification = 10025,
}
