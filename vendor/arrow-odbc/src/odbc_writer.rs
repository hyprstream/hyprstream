use std::{borrow::Cow, cmp::min};

use thiserror::Error;

use arrow::{
    array::Array,
    datatypes::{
        DataType, Date32Type, Date64Type, Field, Float16Type, Float32Type, Float64Type, Int16Type,
        Int32Type, Int64Type, Int8Type, Schema, Time32MillisecondType, Time32SecondType,
        Time64MicrosecondType, Time64NanosecondType, TimeUnit, TimestampMicrosecondType,
        TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type,
    },
    error::ArrowError,
    record_batch::{RecordBatch, RecordBatchReader},
};
use odbc_api::{
    buffers::{AnyBuffer, AnySliceMut, BufferDesc},
    handles::{AsStatementRef, StatementImpl},
    ColumnarBulkInserter, Connection, Prepared, StatementConnection,
};

use crate::{
    date_time::{
        epoch_to_date, epoch_to_timestamp_ms, epoch_to_timestamp_ns, epoch_to_timestamp_s,
        epoch_to_timestamp_us, sec_since_midnight_to_time, NullableTimeAsText,
    },
    decimal::{NullableDecimal128AsText, NullableDecimal256AsText},
};

use self::{
    binary::VariadicBinary,
    boolean::boolean_to_bit,
    map_arrow_to_odbc::MapArrowToOdbc,
    text::{LargeUtf8ToNativeText, Utf8ToNativeText},
};

mod binary;
mod boolean;
mod map_arrow_to_odbc;
mod text;

/// Fastest and most convinient way to stream the contents of arrow record batches into a database
/// table. For usecase there you want to insert repeatedly into the same table from different
/// streams it is more efficient to create an instance of [`self::OdbcWriter`] and reuse it.
///
/// **Note:**
///
/// If table or column names are derived from user input, be sure to sanatize the input in order to
/// prevent SQL injection attacks.
pub fn insert_into_table(
    connection: &Connection,
    batches: &mut impl RecordBatchReader,
    table_name: &str,
    batch_size: usize,
) -> Result<(), WriterError> {
    let schema = batches.schema();
    let mut inserter =
        OdbcWriter::with_connection(connection, schema.as_ref(), table_name, batch_size)?;
    inserter.write_all(batches)
}

/// Generates an insert statement using the table and column names.
///
/// `INSERT INTO <table> (<column_names 0>, <column_names 1>, ...) VALUES (?, ?, ...)`
fn insert_statement_text(table: &str, column_names: &[&'_ str]) -> String {
    // Generate statement text from table name and headline
    let column_names = column_names
        .iter()
        .map(|cn| quote_column_name(cn))
        .collect::<Vec<_>>();
    let columns = column_names.join(", ");
    let values = column_names
        .iter()
        .map(|_| "?")
        .collect::<Vec<_>>()
        .join(", ");
    // Do not finish the statement with a semicolon. There is anecodtical evidence of IBM db2 not
    // allowing the command, because it expects now multiple statements.
    // See: <https://github.com/pacman82/arrow-odbc/issues/63>
    format!("INSERT INTO {table} ({columns}) VALUES ({values})")
}

/// Wraps column name in quotes, if need be
fn quote_column_name(column_name: &str) -> Cow<'_, str> {
    if column_name.contains(|c| !valid_in_column_name(c)) {
        Cow::Owned(format!("\"{column_name}\""))
    } else {
        Cow::Borrowed(column_name)
    }
}

/// Check if this character is allowed in an unquoted column name
fn valid_in_column_name(c: char) -> bool {
    // See:
    // <https://stackoverflow.com/questions/4200351/what-characters-are-valid-in-an-sql-server-database-name>
    c.is_alphanumeric() || c == '@' || c == '$' || c == '#' || c == '_'
}

/// Creates an SQL insert statement from an arrow schema. The resulting statement will have one
/// placeholer (`?`) for each column in the statement.
///
/// **Note:**
///
/// If the column name contains any character which would make it not a valid qualifier for transact
/// SQL it will be wrapped in double quotes (`"`) within the insert schema. Valid names consist of
/// alpha numeric characters, `@`, `$`, `#` and `_`.
///
/// # Example
///
/// ```
/// use arrow_odbc::{
///     insert_statement_from_schema,
///     arrow::datatypes::{Field, DataType, Schema},
/// };
///
/// let field_a = Field::new("a", DataType::Int64, false);
/// let field_b = Field::new("b", DataType::Boolean, false);
///
/// let schema = Schema::new(vec![field_a, field_b]);
/// let sql = insert_statement_from_schema(&schema, "MyTable");
///
/// assert_eq!("INSERT INTO MyTable (a, b) VALUES (?, ?)", sql)
/// ```
///
/// This function is automatically invoked by [`crate::OdbcWriter::with_connection`].
pub fn insert_statement_from_schema(schema: &Schema, table_name: &str) -> String {
    let fields = schema.fields();
    let num_columns = fields.len();
    let column_names: Vec<_> = (0..num_columns)
        .map(|i| fields[i].name().as_str())
        .collect();
    insert_statement_text(table_name, &column_names)
}

/// Emitted writing values from arror arrays into a table on the database
#[derive(Debug, Error)]
pub enum WriterError {
    #[error("Failure to bind the array parameter buffers to the statement.\n{0}")]
    BindParameterBuffers(#[source] odbc_api::Error),
    #[error("Failure to execute the sql statement, sending the data to the database.\n{0}")]
    ExecuteStatment(#[source] odbc_api::Error),
    #[error("An error occured rebinding a parameter buffer to the sql statement.\n{0}")]
    RebindBuffer(#[source] odbc_api::Error),
    #[error("The arrow data type {0} is not supported for insertion.")]
    UnsupportedArrowDataType(DataType),
    #[error("An error occured extracting a record batch from an error reader.\n{0}")]
    ReadingRecordBatch(#[source] ArrowError),
    #[error("An error occurred preparing SQL statement. SQL:\n{sql}\n{source}")]
    PreparingInsertStatement {
        #[source]
        source: odbc_api::Error,
        sql: String,
    },
}

/// Inserts batches from an [`arrow::record_batch::RecordBatchReader`] into a database.
pub struct OdbcWriter<S> {
    /// Prepared statement with bound array parameter buffers. Data is copied into these buffers
    /// until they are full. Then we execute the statement. This is repeated until we run out of
    /// data.
    inserter: ColumnarBulkInserter<S, AnyBuffer>,
    /// For each field in the arrow schema we decide on which buffer to use to send the parameters
    /// to the database, and need to remember how to copy the data from an arrow array to an odbc
    /// mutable buffer slice for any column.
    strategies: Vec<Box<dyn WriteStrategy>>,
}

impl<S> OdbcWriter<S>
where
    S: AsStatementRef,
{
    /// Construct a new ODBC writer using an alredy existing prepared statement. Usually you want to
    /// call a higher level constructor like [`Self::with_connection`]. Yet, this constructor is
    /// useful in two scenarios.
    ///
    /// 1. The prepared statement is already constructed and you do not want to spend the time to
    ///    prepare it again.
    /// 2. You want to use the arrow arrays as arrar parameters for a statement, but that statement
    ///    is not necessarily an INSERT statement with a simple 1to1 mapping of columns between
    ///    table and arrow schema.
    ///
    /// # Parameters
    ///
    /// * `row_capacity`: The amount of rows send to the database in each chunk. With the exception
    ///   of the last chunk, which may be smaller.
    /// * `schema`: Schema needs to have one column for each positional parameter of the statement
    ///   and match the data which will be supplied to the instance later. Otherwise your code will
    ///   panic.
    /// * `statement`: A prepared statement whose SQL text representation contains one placeholder
    ///   for each column. The order of the placeholers must correspond to the orders of the columns
    ///   in the `schema`.
    pub fn new(
        row_capacity: usize,
        schema: &Schema,
        statement: Prepared<S>,
    ) -> Result<Self, WriterError> {
        let strategies: Vec<_> = schema
            .fields()
            .iter()
            .map(|field| field_to_write_strategy(field.as_ref()))
            .collect::<Result<_, _>>()?;
        let descriptions = strategies.iter().map(|cws| cws.buffer_desc());
        let inserter = statement
            .into_column_inserter(row_capacity, descriptions)
            .map_err(WriterError::BindParameterBuffers)?;

        Ok(Self {
            inserter,
            strategies,
        })
    }

    /// Consumes all the batches in the record batch reader and sends them chunk by chunk to the
    /// database.
    pub fn write_all(
        &mut self,
        reader: impl Iterator<Item = Result<RecordBatch, ArrowError>>,
    ) -> Result<(), WriterError> {
        for result in reader {
            let record_batch = result.map_err(WriterError::ReadingRecordBatch)?;
            self.write_batch(&record_batch)?;
        }
        self.flush()?;
        Ok(())
    }

    /// Consumes a single batch and sends it chunk by chunk to the database. The last batch may not
    /// be consumed until [`Self::flush`] is called.
    pub fn write_batch(&mut self, record_batch: &RecordBatch) -> Result<(), WriterError> {
        let capacity = self.inserter.capacity();
        let mut remanining_rows = record_batch.num_rows();
        // The record batch may contain more rows than the capacity of our writer can hold. So we
        // need to be able to fill the buffers multiple times and send them to the database in
        // between.
        while remanining_rows != 0 {
            let chunk_size = min(capacity - self.inserter.num_rows(), remanining_rows);
            let param_offset = self.inserter.num_rows();
            self.inserter.set_num_rows(param_offset + chunk_size);
            let chunk = record_batch.slice(record_batch.num_rows() - remanining_rows, chunk_size);
            for (index, (array, strategy)) in chunk
                .columns()
                .iter()
                .zip(self.strategies.iter())
                .enumerate()
            {
                strategy.write_rows(param_offset, self.inserter.column_mut(index), array)?
            }

            // If we used up all capacity we send the parameters to the database and reset the
            // parameter buffers.
            if self.inserter.num_rows() == capacity {
                self.flush()?;
            }
            remanining_rows -= chunk_size;
        }

        Ok(())
    }

    /// The number of row in an individual record batch must not necessarily match the capacity of
    /// the buffers owned by this writer. Therfore sometimes records are not send to the database
    /// immediatly but rather we wait for the buffers to be filled then reading the next batch. Once
    /// we reach the last batch however, there is no "next batch" anymore. In that case we call this
    /// method in order to send the remainder of the records to the database as well.
    pub fn flush(&mut self) -> Result<(), WriterError> {
        self.inserter
            .execute()
            .map_err(WriterError::ExecuteStatment)?;
        self.inserter.clear();
        Ok(())
    }
}

impl<'env> OdbcWriter<StatementConnection<'env>> {
    /// A writer which takes ownership of the connection and inserts the given schema into a table
    /// with matching column names.
    ///
    /// **Note:**
    ///
    /// If the column name contains any character which would make it not a valid qualifier for transact
    /// SQL it will be wrapped in double quotes (`"`) within the insert schema. Valid names consist of
    /// alpha numeric characters, `@`, `$`, `#` and `_`.
    pub fn from_connection(
        connection: Connection<'env>,
        schema: &Schema,
        table_name: &str,
        row_capacity: usize,
    ) -> Result<Self, WriterError> {
        let sql = insert_statement_from_schema(schema, table_name);
        let statement = connection
            .into_prepared(&sql)
            .map_err(|source| WriterError::PreparingInsertStatement { source, sql })?;
        Self::new(row_capacity, schema, statement)
    }
}

impl<'o> OdbcWriter<StatementImpl<'o>> {
    /// A writer which borrows the connection and inserts the given schema into a table with
    /// matching column names.
    ///
    /// **Note:**
    ///
    /// If the column name contains any character which would make it not a valid qualifier for transact
    /// SQL it will be wrapped in double quotes (`"`) within the insert schema. Valid names consist of
    /// alpha numeric characters, `@`, `$`, `#` and `_`.
    pub fn with_connection(
        connection: &'o Connection<'o>,
        schema: &Schema,
        table_name: &str,
        row_capacity: usize,
    ) -> Result<Self, WriterError> {
        let sql = insert_statement_from_schema(schema, table_name);
        let statement = connection
            .prepare(&sql)
            .map_err(|source| WriterError::PreparingInsertStatement { source, sql })?;
        Self::new(row_capacity, schema, statement)
    }
}

pub trait WriteStrategy {
    /// Describe the buffer used to hold the array parameters for the column
    fn buffer_desc(&self) -> BufferDesc;

    /// # Parameters
    ///
    /// * `param_offset`: Start writing parameters at that position. Number of rows in the parameter
    ///   buffer before inserting the current chunk.
    /// * `column_buf`: Buffer to write the data into
    /// * `array`: Buffer to read the data from
    fn write_rows(
        &self,
        param_offset: usize,
        column_buf: AnySliceMut<'_>,
        array: &dyn Array,
    ) -> Result<(), WriterError>;
}

fn field_to_write_strategy(field: &Field) -> Result<Box<dyn WriteStrategy>, WriterError> {
    let is_nullable = field.is_nullable();
    let strategy = match field.data_type() {
        DataType::Utf8 => Box::new(Utf8ToNativeText {}),
        DataType::Boolean => boolean_to_bit(is_nullable),
        DataType::LargeUtf8 => Box::new(LargeUtf8ToNativeText {}),
        DataType::Int8 => Int8Type::identical(is_nullable),
        DataType::Int16 => Int16Type::identical(is_nullable),
        DataType::Int32 => Int32Type::identical(is_nullable),
        DataType::Int64 => Int64Type::identical(is_nullable),
        DataType::UInt8 => UInt8Type::identical(is_nullable),
        DataType::Float16 => Float16Type::map_with(is_nullable, |half| half.to_f32()),
        DataType::Float32 => Float32Type::identical(is_nullable),
        DataType::Float64 => Float64Type::identical(is_nullable),
        DataType::Timestamp(TimeUnit::Second, None) => {
            TimestampSecondType::map_with(is_nullable, epoch_to_timestamp_s)
        }
        DataType::Timestamp(TimeUnit::Millisecond, None) => {
            TimestampMillisecondType::map_with(is_nullable, epoch_to_timestamp_ms)
        }
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            TimestampMicrosecondType::map_with(is_nullable, epoch_to_timestamp_us)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, None) => {
            TimestampNanosecondType::map_with(is_nullable, |ns| {
                // Drop the last to digits of precision, since we bind it with precision 7 and not 9.
                epoch_to_timestamp_ns((ns / 100) * 100)
            })
        }
        DataType::Date32 => Date32Type::map_with(is_nullable, epoch_to_date),
        DataType::Date64 => Date64Type::map_with(is_nullable, |days_since_epoch| {
            epoch_to_date(days_since_epoch.try_into().unwrap())
        }),
        DataType::Time32(TimeUnit::Second) => {
            Time32SecondType::map_with(is_nullable, sec_since_midnight_to_time)
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            Box::new(NullableTimeAsText::<Time32MillisecondType>::new())
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            Box::new(NullableTimeAsText::<Time64MicrosecondType>::new())
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            Box::new(NullableTimeAsText::<Time64NanosecondType>::new())
        }
        DataType::Binary => Box::new(VariadicBinary::new(1)),
        DataType::FixedSizeBinary(length) => {
            Box::new(VariadicBinary::new((*length).try_into().unwrap()))
        }
        DataType::Decimal128(precision, scale) => {
            Box::new(NullableDecimal128AsText::new(*precision, *scale))
        }
        DataType::Decimal256(precision, scale) => {
            Box::new(NullableDecimal256AsText::new(*precision, *scale))
        }
        unsupported => return Err(WriterError::UnsupportedArrowDataType(unsupported.clone())),
    };
    Ok(strategy)
}
