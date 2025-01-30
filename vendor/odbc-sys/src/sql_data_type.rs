#[repr(C)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct SqlDataType(pub i16);

impl SqlDataType {
    pub const UNKNOWN_TYPE: SqlDataType = SqlDataType(0);
    // also called SQL_VARIANT_TYPE since odbc 4.0
    pub const CHAR: SqlDataType = SqlDataType(1);
    pub const NUMERIC: SqlDataType = SqlDataType(2);
    pub const DECIMAL: SqlDataType = SqlDataType(3);
    /// Exact numeric value with precision 10 and scale 0 (signed: `-2[31] <= n <= 2[31] - 1`,
    /// unsigned: `0 <= n <= 2[32] - 1`).  An application uses `SQLGetTypeInfo` or `SQLColAttribute`
    /// to determine whether a particular data type or a particular column in a result set is
    /// unsigned.
    pub const INTEGER: SqlDataType = SqlDataType(4);
    pub const SMALLINT: SqlDataType = SqlDataType(5);
    pub const FLOAT: SqlDataType = SqlDataType(6);
    pub const REAL: SqlDataType = SqlDataType(7);
    /// Signed, approximate, numeric value with a binary precision 53 (zero or absolute value
    /// `10[-308]` to `10[308]`).
    pub const DOUBLE: SqlDataType = SqlDataType(8);
    pub const DATETIME: SqlDataType = SqlDataType(9);
    pub const VARCHAR: SqlDataType = SqlDataType(12);
    #[cfg(feature = "odbc_version_4")]
    pub const UDT: SqlDataType = SqlDataType(17);
    #[cfg(feature = "odbc_version_4")]
    pub const ROW: SqlDataType = SqlDataType(19);
    #[cfg(feature = "odbc_version_4")]
    pub const ARRAY: SqlDataType = SqlDataType(50);
    #[cfg(feature = "odbc_version_4")]
    pub const MULTISET: SqlDataType = SqlDataType(55);

    // one-parameter shortcuts for date/time data types
    pub const DATE: SqlDataType = SqlDataType(91);
    pub const TIME: SqlDataType = SqlDataType(92);
    /// Year, month, day, hour, minute, and second fields, with valid values as defined for the DATE
    /// and TIME data types.
    pub const TIMESTAMP: SqlDataType = SqlDataType(93);
    #[cfg(feature = "odbc_version_4")]
    pub const TIME_WITH_TIMEZONE: SqlDataType = SqlDataType(94);
    #[cfg(feature = "odbc_version_4")]
    pub const TIMESTAMP_WITH_TIMEZONE: SqlDataType = SqlDataType(95);

    // SQL extended datatypes:
    pub const EXT_TIME_OR_INTERVAL: SqlDataType = SqlDataType(10);
    pub const EXT_TIMESTAMP: SqlDataType = SqlDataType(11);
    pub const EXT_LONG_VARCHAR: SqlDataType = SqlDataType(-1);
    pub const EXT_BINARY: SqlDataType = SqlDataType(-2);
    pub const EXT_VAR_BINARY: SqlDataType = SqlDataType(-3);
    pub const EXT_LONG_VAR_BINARY: SqlDataType = SqlDataType(-4);
    pub const EXT_BIG_INT: SqlDataType = SqlDataType(-5);
    pub const EXT_TINY_INT: SqlDataType = SqlDataType(-6);
    pub const EXT_BIT: SqlDataType = SqlDataType(-7);
    pub const EXT_W_CHAR: SqlDataType = SqlDataType(-8);
    pub const EXT_W_VARCHAR: SqlDataType = SqlDataType(-9);
    pub const EXT_W_LONG_VARCHAR: SqlDataType = SqlDataType(-10);
    pub const EXT_GUID: SqlDataType = SqlDataType(-11);
}
