use std::{sync::Arc, thread};

use arrow::{
    array::{
        Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Date64Array, Decimal128Array,
        Decimal256Builder, FixedSizeBinaryArray, Float16Array, Float32Array, Int16Array,
        Int32Array, Int64Array, Int8Array, LargeStringArray, StringArray, Time32MillisecondArray,
        Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray,
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray, UInt8Array,
    },
    datatypes::{
        ArrowPrimitiveType, DataType, Decimal256Type, Field, Float16Type, Schema, SchemaRef,
        TimeUnit,
    },
    error::ArrowError,
    record_batch::{RecordBatch, RecordBatchReader},
};
use chrono::NaiveDate;
use float_eq::assert_float_eq;
use lazy_static::lazy_static;

/// This declaration is equivalent to `use half::f16`, yet it does have the benefit, that we do not
/// need to directly depend on the `half` crate and worry about version mismatches.
type F16 = <Float16Type as ArrowPrimitiveType>::Native;

use arrow_odbc::{
    arrow::array::Float64Array,
    arrow_schema_from, insert_into_table,
    odbc_api::{
        buffers::TextRowSet,
        sys::{AttrConnectionPooling, AttrCpMatch},
        Connection, ConnectionOptions, Cursor, CursorImpl, Environment, IntoParameter,
        StatementConnection,
    },
    ColumnFailure, Error, OdbcReaderBuilder, OdbcWriter, WriterError,
};

use stdext::function_name;

/// Connection string to our Microsoft SQL Database. Boot it up with docker-compose up
const MSSQL: &str = "Driver={ODBC Driver 17 for SQL Server};\
    Server=localhost;\
    UID=SA;\
    PWD=My@Test@Password1;";

// Rust by default executes tests in parallel. Yet only one environment is allowed at a time.
lazy_static! {
    static ref ENV: Environment = unsafe {
        // Enable connection pooling. Let driver decide wether the attributes of two connection
        // are similar enough to change the attributes of a pooled one, to fit the requested
        // connection, or if it is cheaper to create a new Connection from scratch.
        Environment::set_connection_pooling(AttrConnectionPooling::DriverAware).unwrap();
        let mut env = Environment::new().unwrap();
        // Strict is the default, and is set here to be explicit about it.
        env.set_connection_pooling_matching(AttrCpMatch::Strict).unwrap();
        env
    };
}

/// Fill a record batch with non nullable Integer 32 Bit directly from the datasource
#[test]
fn fetch_nullable_32bit_integer() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any = fetch_arrow_data(table_name, "INTEGER", "(1),(NULL),(3)").unwrap();

    let array_vals = array_any.as_any().downcast_ref::<Int32Array>().unwrap();
    assert!(array_vals.is_valid(0));
    assert!(array_vals.is_null(1));
    assert!(array_vals.is_valid(2));
    assert_eq!([1, 0, 3], *array_vals.values());
}

/// Fill a record batch with non nullable Integer 32 Bit directly from the datasource
#[test]
fn fetch_32bit_integer() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any = fetch_arrow_data(table_name, "INTEGER NOT NULL", "(1),(2),(3)").unwrap();

    let array_vals = array_any.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!([1, 2, 3], *array_vals.values());
}

/// Fill a record batch with non nullable Integer 16 Bit directly from the datasource
#[test]
fn fetch_16bit_integer() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any = fetch_arrow_data(table_name, "SMALLINT NOT NULL", "(1),(2),(3)").unwrap();

    let array_vals = array_any.as_any().downcast_ref::<Int16Array>().unwrap();
    assert_eq!([1, 2, 3], *array_vals.values());
}

/// Fill a record batch with non nullable Integer 8 Bit directly from the datasource. Remark:
/// Tinyint is unsigned for MSSQL
#[test]
fn fetch_unsigend_8bit_integer() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any = fetch_arrow_data(table_name, "TINYINT NOT NULL", "(1),(0),(255)").unwrap();

    let array_vals = array_any.as_any().downcast_ref::<UInt8Array>().unwrap();
    assert_eq!([1, 0, 255], *array_vals.values());
}

/// Fill a record batch with non nullable Integer 8 Bit usigned integer. Since that type would never
/// interferred from the Database automatically it must be specified explicitly in a schema
#[test]
fn fetch_8bit_unsigned_integer_explicit_schema() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    // Setup a table on the database with some floats (so we can fetch them)
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["TINYINT NOT NULL"]).unwrap();
    let sql = format!("INSERT INTO {table_name} (a) VALUES (1),(2),(3)");
    conn.execute(&sql, ()).unwrap();

    // Query column with values to get a cursor
    let sql = format!("SELECT a FROM {table_name}");
    let cursor = conn.execute(&sql, ()).unwrap().unwrap();

    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.

    // Specify Uint8 manually, since inference of the arrow type from the sql type would yield a
    // signed 8 bit integer.
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::UInt8, false)]));
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(100)
        .with_schema(schema)
        .build(cursor)
        .unwrap();

    // Batch for batch copy values from ODBC buffer into arrow batches
    let arrow_batch = reader.next().unwrap().unwrap();

    // Assert that the correct values are found within the arrow batch
    let array_vals = arrow_batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .unwrap();
    assert_eq!([1, 2, 3], *array_vals.values());
}

/// Fill a record batch with non nullable Integer 8 Bit usigned integer. Since that type would never
/// interferred from the Database automatically it must be specified explicitly in a schema
#[test]
fn fetch_decimal128_negative_scale_unsupported() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    // Setup table with dummy value, we won't be able to read it though
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["NUMERIC(5,0) NOT NULL"]).unwrap();
    let sql = format!("INSERT INTO {table_name} (a) VALUES (12300)");
    conn.execute(&sql, ()).unwrap();

    // Query column with values to get a cursor
    let sql = format!("SELECT a FROM {table_name}");
    let cursor = conn.execute(&sql, ()).unwrap().unwrap();

    // Specify Uint8 manually, since inference of the arrow type from the sql type would yield a
    // signed 8 bit integer.
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Decimal128(3, -2),
        false,
    )]));
    let result = OdbcReaderBuilder::new().with_schema(schema).build(cursor);

    assert!(matches!(
        result,
        Err(Error::ColumnFailure {
            source: ColumnFailure::UnsupportedArrowType(DataType::Decimal128(3, -2)),
            index: 0,
            name: _
        })
    ))
}

/// Observe that an explicitly specified Uint16 triggers an unsupported error
#[test]
fn unsupported_16bit_unsigned_integer() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    // Setup a table on the database with some floats (so we can fetch them)
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["SMALLINT NOT NULL"]).unwrap();
    let sql = format!("INSERT INTO {table_name} (a) VALUES (1),(2),(3)");
    conn.execute(&sql, ()).unwrap();

    // Query column with values to get a cursor
    let sql = format!("SELECT a FROM {table_name}");
    let cursor = conn.execute(&sql, ()).unwrap().unwrap();

    // Specify Uint16 manually, since inference of the arrow type from the sql type would yield a
    // signed 16 bit integer.
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::UInt16, false)]));
    let result = OdbcReaderBuilder::new().with_schema(schema).build(cursor);

    assert!(matches!(
        result,
        Err(Error::ColumnFailure {
            source: ColumnFailure::UnsupportedArrowType(DataType::UInt16),
            index: 0,
            name: _
        })
    ))
}

/// Fill a record batch with non nullable Boolean from Bits
#[test]
fn fetch_boolean() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any = fetch_arrow_data(table_name, "BIT NOT NULL", "(1),(0),(1)").unwrap();

    let array_vals = array_any.as_any().downcast_ref::<BooleanArray>().unwrap();

    // Assert that the correct values are found within the arrow batch
    assert!(array_vals.value(0));
    assert!(!array_vals.value(1));
    assert!(array_vals.value(2));
}

/// Fill a record batch with nullable Booleans from Bits
#[test]
fn fetch_nullable_boolean() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any = fetch_arrow_data(table_name, "BIT", "(1),(NULL),(0)").unwrap();

    let array_vals = array_any.as_any().downcast_ref::<BooleanArray>().unwrap();

    // Assert that the correct values are found within the arrow batch
    assert!(array_vals.is_valid(0));
    assert!(array_vals.value(0));
    assert!(array_vals.is_null(1));
    assert!(array_vals.is_valid(2));
    assert!(!array_vals.value(2));
}

/// Fill a record batch with non nullable `f32` directly from the datasource
#[test]
fn fetch_32bit_floating_point() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any = fetch_arrow_data(table_name, "REAL NOT NULL", "(1),(2),(3)").unwrap();

    let array_vals = array_any.as_any().downcast_ref::<Float32Array>().unwrap();
    assert_float_eq!([1., 2., 3.][..], array_vals.values(), abs_all <= 000.1);
}

/// Fill a record batch with non nullable `f64` directly from the datasource
#[test]
fn fetch_64bit_floating_point() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any =
        fetch_arrow_data(table_name, "DOUBLE PRECISION NOT NULL", "(1),(2),(3)").unwrap();

    let array_vals = array_any.as_any().downcast_ref::<Float64Array>().unwrap();

    // Assert that the correct values are found within the arrow batch
    assert_float_eq!([1., 2., 3.][..], array_vals.values(), abs_all <= 000.1);
}

/// Fill a record batch with non nullable `i64` directly from the datasource
#[test]
fn fetch_64bit_integer() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any = fetch_arrow_data(table_name, "BIGINT NOT NULL", "(1),(2),(3)").unwrap();

    let array_vals = array_any.as_any().downcast_ref::<Int64Array>().unwrap();

    // Assert that the correct values are found within the arrow batch
    assert_eq!([1, 2, 3], *array_vals.values());
}

/// Fill a record batch of Strings from a varchar source column
#[test]
fn fetch_varchar() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any =
        fetch_arrow_data(table_name, "VARCHAR(50)", "('Hello'),('Bonjour'),(NULL)").unwrap();

    let array_vals = array_any.as_any().downcast_ref::<StringArray>().unwrap();

    // Assert that the correct values are found within the arrow batch
    assert_eq!("Hello", array_vals.value(0));
    assert_eq!("Bonjour", array_vals.value(1));
    assert!(array_vals.is_null(2));
}

/// Fill a record batch of Strings from a varchar source column
#[test]
fn trim_fixed_sized_character_data() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let cursor = cursor_over(table_name, "CHAR(4)", "('1234'),(' 123'),('123 ')");
    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(4)
        .trim_fixed_sized_characters(true)
        .build(cursor)
        .unwrap()
        .into_concurrent()
        .unwrap();
    // Batch for batch copy values from ODBC buffer into arrow batches
    let record_batch = reader.next().unwrap().unwrap();
    let array_any = record_batch.column(0).clone();

    // Assert that the correct values are found within the arrow batch
    let array_vals = array_any.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!("1234", array_vals.value(0));
    assert_eq!("123", array_vals.value(1));
    assert_eq!("123", array_vals.value(2));
}

/// Fill a record batch of Strings from a nvarchar source column
#[test]
fn fetch_nvarchar() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any =
        fetch_arrow_data(table_name, "NVARCHAR(50)", "('Hello'),('Bonjour'),(NULL)").unwrap();

    let array_vals = array_any.as_any().downcast_ref::<StringArray>().unwrap();

    // Assert that the correct values are found within the arrow batch
    assert_eq!("Hello", array_vals.value(0));
    assert_eq!("Bonjour", array_vals.value(1));
    assert!(array_vals.is_null(2));
}

/// Fill a record batch of Dates
#[test]
fn fetch_dates() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any =
        fetch_arrow_data(table_name, "DATE", "('2021-04-09'),(NULL),('2002-09-30')").unwrap();

    let array_vals = array_any.as_any().downcast_ref::<Date32Array>().unwrap();

    // Assert that the correct values are found within the arrow batch
    assert_eq!(
        Some(NaiveDate::from_ymd_opt(2021, 4, 9).unwrap()),
        array_vals.value_as_date(0)
    );
    assert!(array_vals.is_null(1));
    assert_eq!(
        Some(NaiveDate::from_ymd_opt(2002, 9, 30).unwrap()),
        array_vals.value_as_date(2)
    );
}

/// Fill a record batch of non nullable Dates
#[test]
fn fetch_non_null_dates() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any =
        fetch_arrow_data(table_name, "DATE NOT NULL", "('2021-04-09'),('2002-09-30')").unwrap();

    let array_vals = array_any.as_any().downcast_ref::<Date32Array>().unwrap();

    // Assert that the correct values are found within the arrow batch
    assert_eq!(
        Some(NaiveDate::from_ymd_opt(2021, 4, 9).unwrap()),
        array_vals.value_as_date(0)
    );
    assert_eq!(
        Some(NaiveDate::from_ymd_opt(2002, 9, 30).unwrap()),
        array_vals.value_as_date(1)
    );
}

/// Fill a record batch of non nullable timestamps with milliseconds precision
#[test]
fn fetch_non_null_date_time() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any = fetch_arrow_data(
        table_name,
        "DATETIME NOT NULL",
        "('2021-04-09 18:57:50.12'),('2002-09-30 12:43:17.45')",
    )
    .unwrap();

    let array_vals = array_any
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .unwrap();

    // Assert that the correct values are found within the arrow batch
    assert_eq!(
        Some(
            NaiveDate::from_ymd_opt(2021, 4, 9)
                .unwrap()
                .and_hms_milli_opt(18, 57, 50, 120)
                .unwrap()
        ),
        array_vals.value_as_datetime(0)
    );
    assert_eq!(
        Some(
            NaiveDate::from_ymd_opt(2002, 9, 30)
                .unwrap()
                .and_hms_milli_opt(12, 43, 17, 450)
                .unwrap()
        ),
        array_vals.value_as_datetime(1)
    );
}

/// Fill a record batch of nullable timestamps with milliseconds precision
#[test]
fn fetch_date_time_us() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any = fetch_arrow_data(
        table_name,
        "DATETIME2(6)",
        "('2021-04-09 18:57:50'),(NULL),('2002-09-30 12:43:17')",
    )
    .unwrap();

    let array_vals = array_any
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();

    // Assert that the correct values are found within the arrow batch
    assert_eq!(
        Some(
            NaiveDate::from_ymd_opt(2021, 4, 9)
                .unwrap()
                .and_hms_milli_opt(18, 57, 50, 0)
                .unwrap()
        ),
        array_vals.value_as_datetime(0)
    );
    assert!(array_vals.is_null(1));
    assert_eq!(
        Some(
            NaiveDate::from_ymd_opt(2002, 9, 30)
                .unwrap()
                .and_hms_milli_opt(12, 43, 17, 00)
                .unwrap()
        ),
        array_vals.value_as_datetime(2)
    );
}

/// Fill a record batch of nullable timestamps with milliseconds precision
#[test]
fn fetch_date_time_ms() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any = fetch_arrow_data(
        table_name,
        "DATETIME",
        "('2021-04-09 18:57:50'),(NULL),('2002-09-30 12:43:17')",
    )
    .unwrap();

    let array_vals = array_any
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .unwrap();

    // Assert that the correct values are found within the arrow batch
    assert_eq!(
        Some(
            NaiveDate::from_ymd_opt(2021, 4, 9)
                .unwrap()
                .and_hms_milli_opt(18, 57, 50, 0)
                .unwrap()
        ),
        array_vals.value_as_datetime(0)
    );
    assert!(array_vals.is_null(1));
    assert_eq!(
        Some(
            NaiveDate::from_ymd_opt(2002, 9, 30)
                .unwrap()
                .and_hms_milli_opt(12, 43, 17, 00)
                .unwrap()
        ),
        array_vals.value_as_datetime(2)
    );
}

/// Fetch a timestamp older than unix epoch, i.e 1970-01-01 00:00:00. See issue:
/// <https://github.com/pacman82/arrow-odbc/issues/111>
#[test]
fn fetch_date_time_ms_before_epoch() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any =
        fetch_arrow_data(table_name, "DATETIME", "('1900-01-01 12:43:17.123')").unwrap();

    let array_vals = array_any
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .unwrap();

    // Assert that the correct values are found within the arrow batch
    assert_eq!(
        Some(
            NaiveDate::from_ymd_opt(1900, 1, 1)
                .unwrap()
                .and_hms_milli_opt(12, 43, 17, 123)
                .unwrap()
        ),
        array_vals.value_as_datetime(0)
    );
}

/// Overflows could occur if reusing the same conversion logic across different time units (e.g. ns
/// and ms) due to the difference in time ranges an i64 associated with each unit might be able
/// to represent.
#[test]
fn fetch_timestamp_ms_which_could_not_be_represented_as_i64_ns() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    // Earliest date representable in ns is  1677-09-21T00:12:43.145224192, so this is earlier, but
    // should still work, due to the precision, being set to ms
    let array_any =
        fetch_arrow_data(table_name, "DATETIME2(3)", "('1600-06-18T23:12:44.123Z')").unwrap();

    let array_vals = array_any
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .unwrap();

    // Assert that the correct values are found within the arrow batch
    assert_eq!(
        Some(
            NaiveDate::from_ymd_opt(1600, 6, 18)
                .unwrap()
                .and_hms_milli_opt(23, 12, 44, 123)
                .unwrap()
        ),
        array_vals.value_as_datetime(0)
    );
}

/// Fill a record batch of non nullable timestamps with nanoseconds precision
#[test]
fn fetch_non_null_date_time_ns() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let array_any = fetch_arrow_data(
        table_name,
        "DATETIME2 NOT NULL",
        "('2021-04-09 18:57:50.1234567'),('2002-09-30 12:43:17.456')",
    )
    .unwrap();

    let array_vals = array_any
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();

    // Assert that the correct values are found within the arrow batch
    assert_eq!(
        Some(
            NaiveDate::from_ymd_opt(2021, 4, 9)
                .unwrap()
                .and_hms_nano_opt(18, 57, 50, 123_456_700)
                .unwrap()
        ),
        array_vals.value_as_datetime(0)
    );
    assert_eq!(
        Some(
            NaiveDate::from_ymd_opt(2002, 9, 30)
                .unwrap()
                .and_hms_nano_opt(12, 43, 17, 456_000_000)
                .unwrap()
        ),
        array_vals.value_as_datetime(1)
    );
}

/// Precision 7 timestamps need to be mapped to nanoseconds. Nanoseconds timestamps have a valid
/// range in arrow between 1677-09-21 00:12:44 and 2262-04-11 23:47:16.854775807 due to be
/// represented as a signed 64Bit Integer
#[test]
fn fetch_out_of_range_date_time_ns() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    let error = fetch_arrow_data(
        table_name,
        "DATETIME2 NOT NULL",
        "('2300-01-01 00:00:00.1234567')",
    )
    .unwrap_err();

    assert_eq!(
        "External error: Timestamp is not representable in arrow: 2300-01-01 00:00:00.123456700\n\
        Timestamps with nanoseconds precision are represented using a signed 64 Bit integer. This \
        limits their range to values between 1677-09-21 00:12:44 and \
        2262-04-11 23:47:16.854775807. The value returned from the database is outside of this \
        range. Suggestions to fix this error either reduce the precision or fetch the values as \
        text.",
        error.to_string()
    )
}

/// Precision 7 timestamps need to be mapped to nanoseconds. Nanoseconds timestamps have a valid
/// range in arrow between 1677-09-21 00:12:44 and 2262-04-11 23:47:16.854775807 due to be
/// represented as a signed 64Bit Integer. Default behaviour is to emit an error. In this case we
/// want to map such values to NULL though.
#[test]
fn map_out_of_range_date_time_to_null() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = cursor_over(
        table_name,
        "DATETIME2 NOT NULL",
        "('2300-01-01 00:00:00.1234567'),('2002-09-30 12:43:17.456')",
    );

    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(100)
        .value_errors_as_null(true)
        .build(cursor)
        .unwrap()
        .into_concurrent()
        .unwrap();
    // Batch for batch copy values from ODBC buffer into arrow batches
    let record_batch = reader.next().unwrap().unwrap();
    let array = record_batch.column(0).clone();

    // Assert that the correct values are found within the arrow batch
    let array_vals = array
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();
    assert!(array_vals.is_null(0));
    assert_eq!(
        Some(
            NaiveDate::from_ymd_opt(2002, 9, 30)
                .unwrap()
                .and_hms_nano_opt(12, 43, 17, 456_000_000)
                .unwrap()
        ),
        array_vals.value_as_datetime(1)
    );
}

/// Fill a record batch of Decimals
#[test]
fn fetch_decimals() {
    // Given a cursor over a table with one decimal column
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = cursor_over(table_name, "DECIMAL(5,2) NOT NULL", "(123.45),(678.90)");

    // When fetching it in batches of five
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(5)
        .build(cursor)
        .unwrap();
    let record_batch = reader.next().unwrap().unwrap();

    // Then the elements in the first column of the first batch must match the decimals in the
    // database.
    let column = record_batch.column(0).clone();
    let array_vals = column.as_any().downcast_ref::<Decimal128Array>().unwrap();

    // Assert that the correct values are found within the arrow batch
    assert_eq!("123.45", array_vals.value_as_string(0));
    assert_eq!("678.90", array_vals.value_as_string(1));
}

/// Ensure we do not drop sign in Decimal parsing
#[test]
fn fetch_negative_decimal() {
    // Given a cursor over a table with one decimal column
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = cursor_over(table_name, "DECIMAL(5,2) NOT NULL", "(-123.45)");

    // When fetching it in batches of five
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(5)
        .build(cursor)
        .unwrap();
    let record_batch = reader.next().unwrap().unwrap();

    // Then the elements in the first column of the first batch must match the decimals in the
    // database.
    let column = record_batch.column(0).clone();
    let array_vals = column.as_any().downcast_ref::<Decimal128Array>().unwrap();

    // Assert that the correct values are found within the arrow batch
    assert_eq!("-123.45", array_vals.value_as_string(0));
}

/// Fetch variable sized binary data binary data
#[test]
fn fetch_varbinary_data() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    // Setup a table on the database with some values (so we can fetch them)
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["VARBINARY(30) NOT NULL"]).unwrap();
    let sql = format!("INSERT INTO {table_name} (a) VALUES (?)");
    // Use prepared query and arguments for insertion, since literal representation depends a lot
    // on the DB under test.
    let mut insert = conn.prepare(&sql).unwrap();
    insert.execute(&b"Hello".into_parameter()).unwrap();
    insert.execute(&b"World".into_parameter()).unwrap();

    // Query column with values to get a cursor
    let sql = format!("SELECT a FROM {table_name} ORDER BY id");
    let cursor = conn.execute(&sql, ()).unwrap().unwrap();

    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        // Batches will contain at most 100 entries.
        .with_max_num_rows_per_batch(100)
        // Instantiate reader with Arrow schema and ODBC cursor
        .build(cursor)
        .unwrap();

    // Batch for batch copy values from ODBC buffer into arrow batches
    let arrow_batch = reader.next().unwrap().unwrap();

    // Assert that the correct values are found within the arrow batch
    let array_vals = arrow_batch
        .column(0)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();
    assert_eq!(b"Hello", array_vals.value(0));
    assert_eq!(b"World", array_vals.value(1));
}

/// Fetch fixed sized binary data binary data
#[test]
fn fetch_fixed_sized_binary_data() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    // Setup a table on the database with some values (so we can fetch them)
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["BINARY(5) NOT NULL"]).unwrap();
    let sql = format!("INSERT INTO {table_name} (a) VALUES (?)");
    // Use prepared query and arguments for insertion, since literal representation depends a lot
    // on the DB under test.
    let mut insert = conn.prepare(&sql).unwrap();
    insert.execute(&b"Hello".into_parameter()).unwrap();
    insert.execute(&b"World".into_parameter()).unwrap();

    // Query column with values to get a cursor
    let sql = format!("SELECT a FROM {table_name} ORDER BY id");
    let cursor = conn.execute(&sql, ()).unwrap().unwrap();

    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        // Batches will contain at most 100 entries.
        .with_max_num_rows_per_batch(100)
        // Instantiate reader with Arrow schema and ODBC cursor
        .build(cursor)
        .unwrap();

    // Batch for batch copy values from ODBC buffer into arrow batches
    let arrow_batch = reader.next().unwrap().unwrap();

    // Assert that the correct values are found within the arrow batch
    let array_vals = arrow_batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .unwrap();
    assert_eq!(b"Hello", array_vals.value(0));
    assert_eq!(b"World", array_vals.value(1));
}

/// Like [`fetch_32bit_floating_point`], but utilizing a prepared query instead of a one shot.
#[test]
fn prepared_query() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    // Setup a table on the database with some floats (so we can fetch them)
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["REAL NOT NULL"]).unwrap();
    let sql = format!("INSERT INTO {table_name} (a) VALUES (1),(2),(3)");
    conn.execute(&sql, ()).unwrap();

    // Query column with values to get a cursor
    let sql = format!("SELECT a FROM {table_name}");
    let mut prepared = conn.prepare(&sql).unwrap();
    let cursor = prepared.execute(()).unwrap().unwrap();

    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        // Batches will contain at most 100 entries.
        .with_max_num_rows_per_batch(100)
        // Instantiate reader with Arrow schema and ODBC cursor
        .build(cursor)
        .unwrap();

    // Batch for batch copy values from ODBC buffer into arrow batches
    let arrow_batch = reader.next().unwrap().unwrap();

    // Assert that the correct values are found within the arrow batch
    let array_vals = arrow_batch
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_float_eq!([1., 2., 3.][..], array_vals.values(), abs_all <= 000.1);
}

#[test]
fn infer_schema() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    // Setup a table on the database with some floats (so we can fetch them)
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["REAL NOT NULL"]).unwrap();

    // Prepare query to get metadata
    let sql = format!("SELECT a FROM {table_name}");
    let cursor = conn.execute(&sql, ()).unwrap().unwrap();

    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(1)
        // Instantiate reader with Arrow schema and ODBC cursor
        .build(cursor)
        .unwrap();

    let actual = reader.schema();
    let expected = Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, false)]));

    assert_eq!(expected, actual)
}

#[test]
fn fetch_schema_for_table() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;

    // Setup a table on the database with some floats (so we can fetch them)
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["REAL NOT NULL"]).unwrap();

    // Prepare query to get metadata
    let sql = format!("SELECT a FROM {table_name}");
    let mut prepared = conn.prepare(&sql).unwrap();

    // Now that we have prepared statement, we want to use it to query metadata.
    let schema = arrow_schema_from(&mut prepared, false).unwrap();

    assert_eq!(
        "Field { \
            name: \"a\", \
            data_type: Float32, \
            nullable: false, \
            dict_id: 0, \
            dict_is_ordered: false, \
            metadata: {} \
        }",
        schema.to_string()
    )
}

/// Allocating octet length bytes is not enough if the column on the database is encoded in UTF-16
/// since all codepoints in range from U+0800 to U+FFFF take three bytes in UTF-8 but only two bytes
/// in UTF-16. We test this with the 'Trade Mark Sign' (`™`) (U+2122).
///
/// For this test to be meaningful it must run on a Linux platform, since only then we do query the
/// wide column with a narrow buffer and therfore convert UTF-16 to UTF-8 within the dbms.
#[test]
fn should_allocate_enough_memory_for_wchar_column_bound_to_u8() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = query_single_value(table_name, "NCHAR(1) NOT NULL", "™");

    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(1)
        // Instantiate reader with Arrow schema and ODBC cursor
        .build(cursor)
        .unwrap();
    // Batch for batch copy values from ODBC buffer into arrow batches
    let record_batch = reader.next().unwrap().unwrap();
    let array_any = record_batch.column(0).clone();

    // Assert that the correct values are found within the arrow batch
    let array_vals = array_any.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!("™", array_vals.value(0));
}

/// Allocating octet length bytes is not enough if the column on the database is encoded in UTF-8
/// since all codepoints in range from U+0000 to U+007F take two bytes in UTF-16 but only one byte
/// in UTF-8. We test this with the letter a (U+0061).
///
/// For this test to be meaningful it must run on a windows platform, since only then we do query the
/// wide column with a narrow buffer and therfore convert UTF-8 to UTF-16 within the dbms.
#[test]
fn should_allocate_enough_memory_for_varchar_column_bound_to_u16() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = query_single_value(table_name, "CHAR(1) NOT NULL", "Ü");

    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(1)
        // Instantiate reader with Arrow schema and ODBC cursor
        .build(cursor)
        .unwrap();
    // Batch for batch copy values from ODBC buffer into arrow batches
    let record_batch = reader.next().unwrap().unwrap();
    let array_any = record_batch.column(0).clone();

    // Assert that the correct values are found within the arrow batch
    let array_vals = array_any.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!("Ü", array_vals.value(0));
}

/// Often than VARCHAR(MAX) is used the actual values in these columns are in the range of 100kb and
/// not several kb. Sadly if we allocate the buffers, we have to assume the largest possible element
/// this test verifies that users can specify sensible upper limits using their domain knowledge
/// about the table in order to prevent out of memory issues.
#[test]
fn should_allow_to_fetch_from_varchar_max() {
    // Given
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["VARCHAR(MAX)"]).unwrap();
    let sql = format!("SELECT a FROM {table_name}");
    let cursor = conn.execute(&sql, ()).unwrap().unwrap();

    // When
    let result = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(100)
        .with_max_text_size(1024)
        .build(cursor);

    // Then
    // In particular we do **not** get either a zero sized column or out of memory error.
    assert!(result.is_ok())
}

/// If column limits are too small and truncation occurs, we expect an error to be raised.
#[test]
fn should_error_for_truncation() {
    // Given a column with one value of length 9
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["VARCHAR(MAX)"]).unwrap();
    let sql = format!("INSERT INTO {table_name} (a) VALUES ('123456789')");
    conn.execute(&sql, ()).unwrap();
    let sql = format!("SELECT a FROM {table_name}");
    let cursor = conn.execute(&sql, ()).unwrap().unwrap();

    // When fetching that value with a text limit of 5
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(1)
        .with_max_text_size(5)
        .build(cursor)
        .unwrap();
    let result = reader.next().unwrap();

    // Then we get an error, rather than the truncation only occurring as a warning.
    assert!(result.is_err())
}

#[test]
fn should_allow_to_fetch_from_varbinary_max() {
    // Given
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["VARBINARY(MAX)"]).unwrap();
    let sql = format!("SELECT a FROM {table_name}");
    let cursor = conn.execute(&sql, ()).unwrap().unwrap();

    // When
    let result = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(100)
        .with_max_binary_size(1024)
        .build(cursor);

    // Then
    // In particular we do **not** get either a zero sized column or out of memory error.
    assert!(result.is_ok())
}

#[test]
fn fallibale_allocations() {
    // Given
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["VARBINARY(4096)"]).unwrap();
    let sql = format!("SELECT a FROM {table_name}");
    let cursor = conn.execute(&sql, ()).unwrap().unwrap();

    // When
    let result = OdbcReaderBuilder::new()
        .with_max_bytes_per_batch(usize::MAX)
        .with_max_num_rows_per_batch(100_000_000)
        .with_fallibale_allocations(true)
        .build(cursor);

    // Then
    // In particular we do **not** get either a zero sized column or out of memory error.
    assert!(result.is_err());
    assert!(matches!(
        result.err().unwrap(),
        Error::ColumnFailure {
            name: _,
            index: 0,
            source: ColumnFailure::TooLarge {
                num_elements: 100_000_000,
                element_size: 4096
            }
        }
    ));
}

#[test]
fn read_multiple_result_sets() {
    // Given a cursor returning two result sets
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    let cursor = conn
        .execute("SELECT 1 AS A; SELECT 2 AS B;", ())
        .unwrap()
        .unwrap();

    // When
    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(1)
        .build(cursor)
        .unwrap();
    let first = reader.next().unwrap().unwrap();
    let cursor = reader.into_cursor().unwrap();
    let cursor = cursor.more_results().unwrap().unwrap();
    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(1)
        .build(cursor)
        .unwrap();
    let second = reader.next().unwrap().unwrap();

    // Then
    let first_vals = first
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(1, first_vals.value(0));
    let second_vals = second
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(2, second_vals.value(0));
}

#[test]
fn read_multiple_result_sets_with_second_no_schema() {
    // Given a batch of three SQL statements, the second being result-free
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    let cursor = conn
        .execute(
            "SELECT 1 AS A; SELECT 1 AS A INTO #local_temp_table; SELECT A FROM #local_temp_table;",
            (),
        )
        .unwrap()
        .unwrap();

    // When
    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(1)
        .build(cursor)
        .unwrap();
    let first = reader.next().unwrap().unwrap();
    let cursor = reader.into_cursor().unwrap();
    let cursor = cursor.more_results().unwrap().unwrap();
    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(1)
        .build(cursor)
        .unwrap();
    let second_schema = reader.schema();

    let cursor = reader.into_cursor().unwrap();
    let cursor = cursor.more_results().unwrap().unwrap();
    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(1)
        .build(cursor)
        .unwrap();
    let third = reader.next().unwrap().unwrap();

    // Then
    let first_vals = first
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(1, first_vals.value(0));
    let second_column_count = second_schema.fields().len();
    assert_eq!(0, second_column_count);
    let third_vals = third
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(1, third_vals.value(0));
}

#[test]
fn applies_row_limit_for_default_constructed_readers() {
    // Given a cursor over a datascheme with a small per row memory footprint
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = cursor_over(table_name, "INTEGER", "(42)");

    // When constructing a reader from that cursor without specifying an explicity memory or row
    // limit
    let reader = OdbcReaderBuilder::new().build(cursor).unwrap();

    // Then the row limit is set to u16::MAX
    assert_eq!(reader.max_rows_per_batch(), 65535)
}

#[test]
fn applies_memory_size_limit() {
    // Given a cursor over a datascheme with a small per row memory footprint
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = cursor_over(table_name, "VARCHAR(512)", "('Hello')");

    // When constructing a reader from that cursor without specifying an explicity memory or row
    // limit
    let reader = OdbcReaderBuilder::new()
        // Limit buffer to 10 MiB
        .with_max_bytes_per_batch(10 * 1024 * 1024)
        .build(cursor)
        .unwrap();

    // Buffer now holds less than 65535 rows due to size limit
    assert!(reader.max_rows_per_batch() < 65535)
}

#[test]
fn memory_size_limit_can_not_hold_a_single_row() {
    // Given a cursor over a datascheme with a small per row memory footprint
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = cursor_over(table_name, "VARCHAR(512)", "('Hello')");

    // When constructing a reader from that cursor without specifying an explicity memory or row
    // limit
    let result = OdbcReaderBuilder::new()
        // Limit buffer to 1 Byte
        .with_max_bytes_per_batch(1)
        .build(cursor);

    // Then
    assert!(matches!(
        result,
        Err(Error::OdbcBufferTooSmall {
            max_bytes_per_batch: 1,
            bytes_per_row: _
        })
    ))
}

#[test]
fn insert_does_not_support_list_type() {
    // Given a table and a db connection.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["VARCHAR(4096)"]).unwrap();

    // When we try to create an OdbcWriter inserting an Arrow List
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::List(Arc::new(Field::new("b", DataType::Utf8, true))),
        true,
    )]));

    let insert = format!("INSERT INTO {table_name} (a) VALUES (?)");
    let prepared = conn.prepare(&insert).unwrap();
    let result = OdbcWriter::new(10, schema.as_ref(), prepared);

    // Then we recive an unsupported error
    assert!(matches!(
        result,
        Err(WriterError::UnsupportedArrowDataType(_))
    ))
}

#[test]
fn insert_text() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["VARCHAR(4096)"]).unwrap();
    let array = StringArray::from(vec![Some("Hello"), None, Some("World")]);
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Utf8, true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "Hello\nNULL\nWorld";
    assert_eq!(expected, actual);
}

/// Insert multiple batches into the database using only one roundtrip.
///
/// For this test we are sending two batches, each containing one string for the same column. The
/// second string is longer than the first one. This reproduces an issue which occurred in the
/// context of arrow-odbc-py using chunked arrays .
///
/// See issue: <https://github.com/pacman82/arrow-odbc-py/issues/115>
#[test]
fn insert_multiple_small_batches() {
    // Given
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["VARCHAR(10)"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Utf8, true)]));
    let first_string = StringArray::from(vec![Some("a")]);
    let first_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(first_string)]).unwrap();
    let second_string = StringArray::from(vec![Some("bc")]);
    let second_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(second_string)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![first_batch, second_batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "a\nbc";
    assert_eq!(expected, actual);
}

/// This test is most relevant on windows platforms, the UTF-8 is not the default encoding and text
/// should be encoded as UTF-16
#[test]
fn insert_non_ascii_text() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["VARCHAR(50)"]).unwrap();
    let array = StringArray::from(vec![Some("Frühstück µ")]);
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Utf8, true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "Frühstück µ";
    assert_eq!(expected, actual);
}

#[test]
fn insert_nullable_booleans() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["BIT"]).unwrap();
    let array = BooleanArray::from(vec![Some(true), None, Some(false)]);
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Boolean, true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch.clone(), batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1\nNULL\n0\n1\nNULL\n0";
    assert_eq!(expected, actual);
}

#[test]
fn insert_non_nullable_booleans() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["BIT"]).unwrap();
    let array = BooleanArray::from(vec![Some(true), Some(false), Some(false)]);
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Boolean, false)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch.clone(), batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1\n0\n0\n1\n0\n0";
    assert_eq!(expected, actual);
}

#[test]
fn insert_nullable_int8() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["TINYINT"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, true)]));
    let array1 = Int8Array::from(vec![Some(1), None, Some(3)]);
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(array1)]).unwrap();
    let array2 = Int8Array::from(vec![Some(4), None, Some(6)]);
    let batch2 = RecordBatch::try_new(schema.clone(), vec![Arc::new(array2)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch1, batch2]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1\nNULL\n3\n4\nNULL\n6";
    assert_eq!(expected, actual);
}

#[test]
fn insert_non_nullable_int8() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["TINYINT"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)]));
    let array1 = Int8Array::from(vec![Some(1), Some(2), Some(3)]);
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(array1)]).unwrap();
    let array2 = Int8Array::from(vec![Some(4), Some(5), Some(6)]);
    let batch2 = RecordBatch::try_new(schema.clone(), vec![Arc::new(array2)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch1, batch2]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1\n2\n3\n4\n5\n6";
    assert_eq!(expected, actual);
}

#[test]
fn insert_nullable_int16() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["SMALLINT"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int16, true)]));
    let array1 = Int16Array::from(vec![Some(1), None, Some(3)]);
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(array1)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch1]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1\nNULL\n3";
    assert_eq!(expected, actual);
}

#[test]
fn insert_nullable_int32() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["INTEGER"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    let array1 = Int32Array::from(vec![Some(1), None, Some(3)]);
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(array1)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch1]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1\nNULL\n3";
    assert_eq!(expected, actual);
}

#[test]
fn insert_nullable_int64() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["BIGINT"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
    let array1 = Int64Array::from(vec![Some(1), None, Some(3)]);
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(array1)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch1]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1\nNULL\n3";
    assert_eq!(expected, actual);
}

#[test]
fn insert_non_nullable_unsigned_int8() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["SMALLINT"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::UInt8, false)]));
    let array1 = UInt8Array::from(vec![Some(1), Some(2), Some(3)]);
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(array1)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch1]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1\n2\n3";
    assert_eq!(expected, actual);
}

#[test]
fn insert_nullable_f32() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["REAL"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, true)]));
    let array1 = Float32Array::from(vec![Some(1.), None, Some(3.)]);
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(array1)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch1]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1.0\nNULL\n3.0";
    assert_eq!(expected, actual);
}

#[test]
fn insert_nullable_f64() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["FLOAT(25)"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Float64, true)]));
    let array1 = Float64Array::from(vec![Some(1.), None, Some(3.)]);
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(array1)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch1]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1.0\nNULL\n3.0";
    assert_eq!(expected, actual);
}

#[test]
fn insert_nullable_f16() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["REAL"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Float16, true)]));
    let array1: Float16Array = [Some(F16::from_f32(1.0)), None, Some(F16::from_f32(3.0))]
        .into_iter()
        .collect();
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(array1)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch1]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1.0\nNULL\n3.0";
    assert_eq!(expected, actual);
}

#[test]
fn insert_non_nullable_f16() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["REAL"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Float16, false)]));
    let array1: Float16Array = [
        Some(F16::from_f32(1.0)),
        Some(F16::from_f32(2.0)),
        Some(F16::from_f32(3.0)),
    ]
    .into_iter()
    .collect();
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(array1)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch1]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1.0\n2.0\n3.0";
    assert_eq!(expected, actual);
}

#[test]
fn insert_timestamp_with_seconds_precisions() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["DATETIME2(0)"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Timestamp(TimeUnit::Second, None),
        false,
    )]));
    // Corresponds to single element array with entry 1970-05-09T14:25:11+0:00
    let array = TimestampSecondArray::from(vec![11111111]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1970-05-09 14:25:11";
    assert_eq!(expected, actual);
}

#[test]
fn insert_timestamp_with_milliseconds_precisions() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["DATETIME2(3)"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Timestamp(TimeUnit::Millisecond, None),
        false,
    )]));
    // Corresponds to single element array with entry 1970-05-09T14:25:11.111
    let array = TimestampMillisecondArray::from(vec![11111111111]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1970-05-09 14:25:11.111";
    assert_eq!(expected, actual);
}

/// Overflows could occur if reusing the same conversion logic across different time units (e.g. ns
/// and ms) due to the difference in time ranges an i64 associated with each unit might be able
/// to represent.
///
/// See issue: <https://github.com/pacman82/arrow-odbc/issues/113>
#[test]
fn insert_timestamp_with_milliseconds_precisions_which_is_not_representable_as_i64_ns() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["DATETIME2(3)"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Timestamp(TimeUnit::Millisecond, None),
        false,
    )]));
    // Corresponds to single element array with entry 1970-05-09T14:25:11.111
    let ndt = NaiveDate::from_ymd_opt(1600, 6, 18)
        .unwrap()
        .and_hms_milli_opt(23, 12, 44, 123)
        .unwrap();
    let epoch_ms = ndt.and_utc().timestamp_millis();
    let array = TimestampMillisecondArray::from(vec![epoch_ms]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1600-06-18 23:12:44.123";
    assert_eq!(expected, actual);
}

#[test]
fn insert_timestamp_with_microseconds_precisions() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["DATETIME2(6)"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )]));
    // Corresponds to single element array with entry 1970-05-09T14:25:11.111111
    let array = TimestampMicrosecondArray::from(vec![11111111111111]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1970-05-09 14:25:11.111111";
    assert_eq!(expected, actual);
}

#[test]
fn insert_timestamp_with_nanoseconds_precisions() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["DATETIME2(7)"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        false,
    )]));
    // Corresponds to single element array with entry 1970-05-09T14:25:11.111111111
    let array = TimestampNanosecondArray::from(vec![11111111111111111]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1970-05-09 14:25:11.1111111";
    assert_eq!(expected, actual);
}

#[test]
fn insert_date32_array() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["DATE"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Date32, false)]));
    // Corresponds to single element array with entry 1970-01-01
    let array: Date32Array = [Some(0)].into_iter().collect();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1970-01-01";
    assert_eq!(expected, actual);
}

#[test]
fn insert_date64_array() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["DATE"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Date64, false)]));
    // Corresponds to single element array with entry 1970-01-01
    let array: Date64Array = [Some(0)].into_iter().collect();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "1970-01-01";
    assert_eq!(expected, actual);
}

#[test]
fn insert_time32_second_array() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["TIME(0)"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Time32(TimeUnit::Second),
        false,
    )]));
    // Corresponds to single element array with entry 03:05:11
    let array: Time32SecondArray = [Some(11_111)].into_iter().collect();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "03:05:11";
    assert_eq!(expected, actual);
}

#[test]
fn insert_time32_ms_array() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["TIME(3)"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Time32(TimeUnit::Millisecond),
        false,
    )]));
    // Corresponds to single element array with entry 03:05:11.111
    let array: Time32MillisecondArray = [Some(11_111_111)].into_iter().collect();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "03:05:11.111";
    assert_eq!(expected, actual);
}

#[test]
fn insert_time64_us_array() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["TIME(6)"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Time64(TimeUnit::Microsecond),
        false,
    )]));
    // Corresponds to single element array with entry 03:05:11.111111
    let array: Time64MicrosecondArray = [Some(11_111_111_111)].into_iter().collect();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "03:05:11.111111";
    assert_eq!(expected, actual);
}

#[test]
fn insert_time64_ns_array() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["TIME(7)"]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Time64(TimeUnit::Nanosecond),
        false,
    )]));
    // Corresponds to single element array with entry 03:05:11.111111111
    let array: Time64NanosecondArray = [Some(11_111_111_111_111)].into_iter().collect();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    // We currently insert nanoseconds with precision 7
    let expected = "03:05:11.1111111";
    assert_eq!(expected, actual);
}

#[test]
fn insert_binary() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["VARBINARY(4096)"]).unwrap();
    let array = BinaryArray::from(vec![
        Some([1, 2].as_slice()),
        None,
        Some([3, 4, 5, 6, 7].as_slice()),
    ]);
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Binary, true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "0102\nNULL\n0304050607";
    assert_eq!(expected, actual);
}

#[test]
fn insert_fixed_binary() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["VARBINARY(4096)"]).unwrap();
    let array = BinaryArray::from(vec![
        Some([1, 2].as_slice()),
        None,
        Some([3, 4, 5, 6, 7].as_slice()),
    ]);
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Binary, true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "0102\nNULL\n0304050607";
    assert_eq!(expected, actual);
}

#[test]
fn insert_decimal_128() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["NUMERIC(5,3)"]).unwrap();
    let array: Decimal128Array = [Some(12345), None, Some(67891), Some(1), Some(1000)]
        .into_iter()
        .collect();
    let array = array.with_precision_and_scale(5, 3).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Decimal128(5, 3),
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 2).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "12.345\nNULL\n67.891\n.001\n1.000";
    assert_eq!(expected, actual);
}

#[test]
fn insert_decimal_256() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["NUMERIC(5,3)"]).unwrap();
    let mut builder = Decimal256Builder::new();
    let mut bytes = [0u8; 32];
    type I256 = <Decimal256Type as ArrowPrimitiveType>::Native;
    bytes[0..4].copy_from_slice(12345i32.to_le_bytes().as_slice());
    builder.append_value(I256::from_le_bytes(bytes));
    builder.append_null();
    let array = builder.finish().with_precision_and_scale(5, 3).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Decimal256(5, 3),
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "12.345\nNULL";
    assert_eq!(expected, actual);
}

#[test]
fn insert_decimal_128_with_negative_scale() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["NUMERIC(5,0)"]).unwrap();
    let array: Decimal128Array = [Some(123), None, Some(456), Some(1), Some(10)]
        .into_iter()
        .collect();
    let array = array.with_precision_and_scale(3, -2).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Decimal128(3, -2),
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 2).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "12300\nNULL\n45600\n100\n1000";
    assert_eq!(expected, actual);
}

#[test]
fn insert_decimal_256_with_negative_scale() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["NUMERIC(5,0)"]).unwrap();
    let mut builder = Decimal256Builder::new();
    let mut bytes = [0u8; 32];
    type I256 = <Decimal256Type as ArrowPrimitiveType>::Native;
    bytes[0..4].copy_from_slice(123i32.to_le_bytes().as_slice());
    builder.append_value(I256::from_le_bytes(bytes));
    builder.append_null();
    let array = builder.finish().with_precision_and_scale(3, -2).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Decimal256(3, -2),
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "12300\nNULL";
    assert_eq!(expected, actual);
}

#[test]
fn insert_taking_ownership_of_connection() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["VARCHAR(4096)"]).unwrap();
    let array = StringArray::from(vec![Some("Hello"), None, Some("World")]);
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Utf8, true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let reader = StubBatchReader::new(schema.clone(), vec![batch]);

    // When
    let mut inserter = {
        let conn = ENV
            .connect_with_connection_string(MSSQL, Default::default())
            .unwrap();
        let row_capacity = 50;
        OdbcWriter::from_connection(conn, &schema, table_name, row_capacity).unwrap()
    };
    inserter.write_all(reader).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "Hello\nNULL\nWorld";
    assert_eq!(expected, actual);
}

#[test]
fn insert_large_text() {
    // Given a table and a record batch reader returning a batch with a text column.
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &["VARCHAR(4096)"]).unwrap();
    let array = LargeStringArray::from(vec![Some("Hello"), None, Some("World")]);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::LargeUtf8,
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);

    // When
    insert_into_table(&conn, &mut reader, table_name, 5).unwrap();

    // Then
    let actual = table_to_string(&conn, table_name, &["a"]);
    let expected = "Hello\nNULL\nWorld";
    assert_eq!(expected, actual);
}

#[test]
fn sanatize_column_names() {
    // Given a table with a column name containing a space ...
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    let drop_table = &format!("DROP TABLE IF EXISTS {table_name}");
    let create_table = format!(
        "CREATE TABLE {table_name} (id int IDENTITY(1,1),\"column name with spaces\" INTEGER);"
    );
    conn.execute(drop_table, ()).unwrap();
    conn.execute(&create_table, ()).unwrap();
    let array = Int32Array::from(vec![Some(42)]);

    // When inserting from a reader which features a schema with an unescaped column name
    let schema = Arc::new(Schema::new(vec![Field::new(
        "column name with spaces",
        DataType::Int32,
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let mut reader = StubBatchReader::new(schema, vec![batch]);
    let result = insert_into_table(&conn, &mut reader, table_name, 1);

    // Then the insert does not throw an error, but inserts correctly using the sanatized column
    // name.
    assert!(result.is_ok());
    let actual = table_to_string(&conn, table_name, &["\"column name with spaces\""]);
    let expected = "42";
    assert_eq!(expected, actual);
}

/// Fill a record batch with non nullable Integer 32 Bit directly from the datasource
#[test]
fn fetch_integer_concurrently() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = cursor_over(table_name, "INTEGER", "(1),(NULL),(3)");
    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(100)
        .build(cursor)
        .unwrap();
    // Batch for batch copy values from ODBC buffer into arrow batches
    let record_batch = reader.next().unwrap().unwrap();

    let array_any = record_batch.column(0).clone();
    let array_vals = array_any.as_any().downcast_ref::<Int32Array>().unwrap();
    assert!(array_vals.is_valid(0));
    assert!(array_vals.is_null(1));
    assert!(array_vals.is_valid(2));
    assert_eq!([1, 0, 3], *array_vals.values());
}

#[test]
fn fetch_empty_cursor_concurrently() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = empty_cursor(table_name, "INTEGER");

    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(100)
        .build(cursor)
        .unwrap();
    // Batch for batch copy values from ODBC buffer into arrow batches
    let record_batch = reader.next();

    assert!(record_batch.is_none())
}

/// Concurrent ODBC reader should forward errors from the fetch thread correctly to the main branch
/// calling next.
#[test]
fn fetch_with_error_concurrently() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = cursor_over(table_name, "VARCHAR(50)", "('Hello, World!')");

    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(100)
        // We set text size too small to hold 'Hello, World!' so we get a truncation error.
        .with_max_text_size(1)
        .build(cursor)
        .unwrap()
        .into_concurrent()
        .unwrap();

    // Batch for batch copy values from ODBC buffer into arrow batches
    let record_batch = reader.next().unwrap();

    assert!(record_batch.is_err())
}

#[test]
fn fetch_row_groups_repeatedly_concurrently() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = cursor_over(table_name, "INTEGER", "(1),(NULL),(3)");
    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.

    let mut reader = OdbcReaderBuilder::new()
        // Choose a batch size 1, so we get 3 batches.
        .with_max_num_rows_per_batch(1)
        .build(cursor)
        .unwrap()
        .into_concurrent()
        .unwrap();

    // Batch for batch copy values from ODBC buffer into arrow batches
    let record_batch = reader.next().unwrap().unwrap();
    let array_any = record_batch.column(0).clone();
    let array_vals_1 = array_any.as_any().downcast_ref::<Int32Array>().unwrap();
    let record_batch = reader.next().unwrap().unwrap();
    let array_any = record_batch.column(0).clone();
    let array_vals_2 = array_any.as_any().downcast_ref::<Int32Array>().unwrap();
    let record_batch = reader.next().unwrap().unwrap();
    let array_any = record_batch.column(0).clone();
    let array_vals_3 = array_any.as_any().downcast_ref::<Int32Array>().unwrap();

    assert!(array_vals_1.is_valid(0));
    assert_eq!([1], *array_vals_1.values());
    assert!(array_vals_2.is_null(0));
    assert!(array_vals_3.is_valid(0));
    assert_eq!([3], *array_vals_3.values());
}

#[test]
fn fetch_empty_cursor_concurrently_twice() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = empty_cursor(table_name, "INTEGER");

    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(100)
        .build(cursor)
        .unwrap()
        .into_concurrent()
        .unwrap();
    let _ = reader.next();
    let record_batch = reader.next();

    assert!(record_batch.is_none())
}

#[test]
fn read_multiple_result_sets_using_concurrent_cursor() {
    // Given a cursor returning two result sets
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    let cursor = conn
        .into_cursor("SELECT 1 AS A; SELECT 2 AS B;", ())
        .unwrap()
        .unwrap();

    // When
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(1)
        .build(cursor)
        .unwrap()
        .into_concurrent()
        .unwrap();
    let first = reader.next().unwrap().unwrap();
    let cursor = reader.into_cursor().unwrap();
    let cursor = cursor.more_results().unwrap().unwrap();
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(1)
        .build(cursor)
        .unwrap()
        .into_concurrent()
        .unwrap();
    let second = reader.next().unwrap().unwrap();

    // Then
    let first_vals = first
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(1, first_vals.value(0));
    let second_vals = second
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(2, second_vals.value(0));
}

#[test]
fn promote_sequential_to_concurrent_cursor() {
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = cursor_over(table_name, "INTEGER", "(42)");
    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(100)
        .build(cursor)
        .unwrap()
        .into_concurrent()
        .unwrap();

    let record_batch = reader.next().unwrap().unwrap();
    let array_any = record_batch.column(0).clone();
    let array_vals = array_any.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!([42], *array_vals.values());
}

#[test]
fn concurrent_reader_is_send() {
    // Given a conucurrent_reader
    let table_name = function_name!().rsplit_once(':').unwrap().1;
    let cursor = cursor_over(table_name, "INTEGER", "(42)");
    let mut concurrent_reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(100)
        .build(cursor)
        .unwrap()
        .into_concurrent()
        .unwrap();

    // When send to another thread. compile time error otherwise, most important implicit assertion
    // of this test.
    let record_batch = thread::spawn(move || concurrent_reader.next().unwrap().unwrap())
        .join()
        .unwrap();

    // Then
    let array_any = record_batch.column(0).clone();
    let array_vals = array_any.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!([42], *array_vals.values());
}

/// Creates the table and assures it is empty. Columns are named a,b,c, etc.
fn setup_empty_table(
    conn: &Connection,
    table_name: &str,
    column_types: &[&str],
) -> Result<(), odbc_api::Error> {
    let drop_table = &format!("DROP TABLE IF EXISTS {table_name}");

    let column_names = &["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"];
    let cols = column_types
        .iter()
        .zip(column_names)
        .map(|(ty, name)| format!("{name} {ty}"))
        .collect::<Vec<_>>()
        .join(", ");

    let create_table = format!("CREATE TABLE {table_name} (id int IDENTITY(1,1),{cols});");
    conn.execute(drop_table, ())?;
    conn.execute(&create_table, ())?;
    Ok(())
}

/// Query the table and prints it contents to a string
pub fn table_to_string(conn: &Connection<'_>, table_name: &str, column_names: &[&str]) -> String {
    let cols = column_names.join(", ");
    let query = format!("SELECT {cols} FROM {table_name}");
    let cursor = conn.execute(&query, ()).unwrap().unwrap();
    cursor_to_string(cursor)
}

pub fn cursor_to_string(mut cursor: impl Cursor) -> String {
    let batch_size = 20;
    let mut buffer = TextRowSet::for_cursor(batch_size, &mut cursor, Some(8192)).unwrap();
    let mut row_set_cursor = cursor.bind_buffer(&mut buffer).unwrap();

    let mut text = String::new();

    while let Some(row_set) = row_set_cursor.fetch().unwrap() {
        for row_index in 0..row_set.num_rows() {
            if row_index != 0 {
                text.push('\n');
            }
            for col_index in 0..row_set.num_cols() {
                if col_index != 0 {
                    text.push(',');
                }
                text.push_str(
                    row_set
                        .at_as_str(col_index, row_index)
                        .unwrap()
                        .unwrap_or("NULL"),
                );
            }
        }
    }

    text
}

/// Inserts the values in the literal into the database and returns them as an Arrow array.
fn fetch_arrow_data(
    table_name: &str,
    column_type: &str,
    literal: &str,
) -> Result<ArrayRef, anyhow::Error> {
    let cursor = cursor_over(table_name, column_type, literal);
    // Now that we have a cursor, we want to iterate over its rows and fill an arrow batch with it.
    let mut reader = OdbcReaderBuilder::new()
        .with_max_num_rows_per_batch(100)
        .build(cursor)
        .unwrap()
        .into_concurrent()
        .unwrap();

    // Batch for batch copy values from ODBC buffer into arrow batches
    let record_batch = reader.next().unwrap()?;

    Ok(record_batch.column(0).clone())
}

fn cursor_over(
    table_name: &str,
    column_type: &str,
    literal: &str,
) -> CursorImpl<StatementConnection<'static>> {
    // Setup a table on the database
    let conn = ENV
        .connect_with_connection_string(MSSQL, ConnectionOptions::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &[column_type]).unwrap();
    // Insert values using literals
    let sql = format!("INSERT INTO {table_name} (a) VALUES {literal}");
    conn.execute(&sql, ()).unwrap();
    // Query column with values to get a cursor
    let sql = format!("SELECT a FROM {table_name}");
    let cursor = conn.into_cursor(&sql, ()).unwrap().unwrap();
    cursor
}

fn empty_cursor(table_name: &str, column_type: &str) -> CursorImpl<StatementConnection<'static>> {
    // Setup a table on the database
    let conn = ENV
        .connect_with_connection_string(MSSQL, ConnectionOptions::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &[column_type]).unwrap();
    // Query column with values to get a cursor
    let sql = format!("SELECT a FROM {table_name}");
    let cursor = conn.into_cursor(&sql, ()).unwrap().unwrap();
    cursor
}

fn query_single_value(
    table_name: &str,
    column_type: &str,
    value: impl IntoParameter,
) -> impl Cursor {
    // Setup a table on the database
    let conn = ENV
        .connect_with_connection_string(MSSQL, Default::default())
        .unwrap();
    setup_empty_table(&conn, table_name, &[column_type]).unwrap();
    // Insert values using literals
    let sql = format!("INSERT INTO {table_name} (a) VALUES (?)");
    conn.execute(&sql, &value.into_parameter()).unwrap();
    // Query column with values to get a cursor
    let sql = format!("SELECT a FROM {table_name}");
    conn.into_cursor(&sql, ()).unwrap().unwrap()
}

/// An arrow batch reader emitting predefined batches. Used to test insertion.
struct StubBatchReader {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

impl StubBatchReader {
    pub fn new(schema: SchemaRef, mut batches: Vec<RecordBatch>) -> Self {
        // We pop elements from the end, so we revert order of the batches. This way we do not
        // betray, the expectation that the batches will be emitted in the same order as constructed
        // in the `Vec` given to us.
        batches.reverse();
        Self { schema, batches }
    }
}

impl Iterator for StubBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.batches.pop().map(Ok)
    }
}

impl RecordBatchReader for StubBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
