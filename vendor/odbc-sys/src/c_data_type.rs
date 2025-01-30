/// Extended C Types range 4000 and above. Range of -100 thru 200 is reserved by Driver Manager.
/// `SQL_C_TYPES_EXTENDED`.
pub const C_TYPES_EXTENDED: i16 = 0x04000;

/// The C data type is specified in the SQLBindCol and SQLGetData functions with the TargetType
/// argument and in the SQLBindParameter function with the ValueType argument.
#[repr(i16)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CDataType {
    /// SQL_ARD_TYPE
    Ard = -99,

    /// SQL_APD_TYPE
    Apd = -100,

    UTinyInt = -28,
    UBigInt = -27,
    STinyInt = -26,
    SBigInt = -25,

    ULong = -18,
    UShort = -17,
    SLong = -16,
    SShort = -15,

    #[cfg(feature = "odbc_version_3_50")]
    Guid = -11,

    WChar = -8,

    Bit = -7,
    // deprecated
    // SQL_C_TINYINT = -6,
    Binary = -2,
    /// `SQLCHAR` - CHAR, VARCHAR, DECIMAL, NUMERIC
    Char = 1,
    Numeric = 2,

    // deprecated
    // SQL_C_LONG = 4,
    // SQL_C_SHORT = 5,
    Float = 7,
    Double = 8,

    // Used in Odbc2.x Odbc3.x uses TypeDate instead.
    Date = 9,
    // Used in Odbc2.x Odbc3.x uses TypeTime instead.
    Time = 10,
    // Used in Odbc2.x Odbc3.x uses TypeTimeTimestamp instead.
    TimeStamp = 11,

    /// SQL_TYPE_DATE
    TypeDate = 91,
    /// SQL_TYPE_TIME
    TypeTime = 92,
    /// SQL_TYPE_TIMESTAMP
    TypeTimestamp = 93,
    #[cfg(feature = "odbc_version_4")]
    TypeTimeWithTimezone = 94,
    #[cfg(feature = "odbc_version_4")]
    TypeTimestampWithTimezone = 95,

    Default = 99,

    IntervalYear = 101,
    IntervalMonth = 102,
    IntervalDay = 103,
    IntervalHour = 104,
    IntervalMinute = 105,
    IntervalSecond = 106,
    IntervalYearToMonth = 107,
    IntervalDayToHour = 108,
    IntervalDayToMinute = 109,
    IntervalDayToSecond = 110,
    IntervalHourToMinute = 111,
    IntervalHourToSecond = 112,
    IntervalMinuteToSecond = 113,

    SsTime2 = C_TYPES_EXTENDED,
    SsTimestampOffset = C_TYPES_EXTENDED + 1,
}

#[cfg(windows)]
pub use CDataType::ULong as UBigInt;
#[cfg(not(windows))]
pub use CDataType::ULong as Bookmark;
