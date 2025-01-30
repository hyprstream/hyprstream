use std::cmp::min;

use arrow::{
    datatypes::SchemaRef,
    error::ArrowError,
    record_batch::{RecordBatch, RecordBatchReader},
};
use odbc_api::{buffers::ColumnarAnyBuffer, BlockCursor, Cursor};

use crate::{BufferAllocationOptions, ConcurrentOdbcReader, Error};

use super::to_record_batch::ToRecordBatch;

/// Arrow ODBC reader. Implements the [`arrow::record_batch::RecordBatchReader`] trait so it can be
/// used to fill Arrow arrays from an ODBC data source.
///
/// This reader is generic over the cursor type so it can be used in cases there the cursor only
/// borrows a statement handle (most likely the case then using prepared queries), or owned
/// statement handles (recommened then using one shot queries, to have an easier life with the
/// borrow checker).
///
/// # Example
///
/// ```no_run
/// use arrow_odbc::{odbc_api::{Environment, ConnectionOptions}, OdbcReaderBuilder};
///
/// const CONNECTION_STRING: &str = "\
///     Driver={ODBC Driver 17 for SQL Server};\
///     Server=localhost;\
///     UID=SA;\
///     PWD=My@Test@Password1;\
/// ";
///
/// fn main() -> Result<(), anyhow::Error> {
///
///     let odbc_environment = Environment::new()?;
///     
///     // Connect with database.
///     let connection = odbc_environment.connect_with_connection_string(
///         CONNECTION_STRING,
///         ConnectionOptions::default()
///     )?;
///
///     // This SQL statement does not require any arguments.
///     let parameters = ();
///
///     // Execute query and create result set
///     let cursor = connection
///         .execute("SELECT * FROM MyTable", parameters)?
///         .expect("SELECT statement must produce a cursor");
///
///     // Read result set as arrow batches. Infer Arrow types automatically using the meta
///     // information of `cursor`.
///     let arrow_record_batches = OdbcReaderBuilder::new()
///         .build(cursor)?;
///
///     for batch in arrow_record_batches {
///         // ... process batch ...
///     }
///     Ok(())
/// }
/// ```
pub struct OdbcReader<C: Cursor> {
    /// Converts the content of ODBC buffers into Arrow record batches
    converter: ToRecordBatch,
    /// Fetches values from the ODBC datasource using columnar batches. Values are streamed batch
    /// by batch in order to avoid reallocation of the buffers used for tranistion.
    batch_stream: BlockCursor<C, ColumnarAnyBuffer>,
    /// We remember if the user decided to use fallibale allocations or not in case we need to
    /// allocate another buffer due to a state transition towards [`ConcurrentOdbcReader`].
    fallibale_allocations: bool,
}

impl<C: Cursor> OdbcReader<C> {
    /// Consume this instance to create a similar ODBC reader which fetches batches asynchronously.
    ///
    /// Steals all resources from this [`OdbcReader`] instance, and allocates another buffer for
    /// transiting data from the ODBC data source to the application. This way one buffer can be
    /// written to by a dedicated system thread, while the other is read by the application. Use
    /// this if you want to trade memory for speed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use arrow_odbc::{odbc_api::{Environment, ConnectionOptions}, OdbcReaderBuilder};
    /// use std::sync::OnceLock;
    ///
    /// // In order to fetch in a dedicated system thread we need a cursor with static lifetime,
    /// // this implies a static ODBC environment.
    /// static ENV: OnceLock<Environment> = OnceLock::new();
    ///
    /// const CONNECTION_STRING: &str = "\
    ///     Driver={ODBC Driver 17 for SQL Server};\
    ///     Server=localhost;\
    ///     UID=SA;\
    ///     PWD=My@Test@Password1;\
    /// ";
    ///
    /// fn main() -> Result<(), anyhow::Error> {
    ///
    ///     let odbc_environment = ENV.get_or_init(|| {Environment::new().unwrap() });
    ///     
    ///     // Connect with database.
    ///     let connection = odbc_environment.connect_with_connection_string(
    ///         CONNECTION_STRING,
    ///         ConnectionOptions::default()
    ///     )?;
    ///
    ///     // This SQL statement does not require any arguments.
    ///     let parameters = ();
    ///
    ///     // Execute query and create result set
    ///     let cursor = connection
    ///         // Using `into_cursor` instead of `execute` takes ownership of the connection and
    ///         // allows for a cursor with static lifetime.
    ///         .into_cursor("SELECT * FROM MyTable", parameters)
    ///         .map_err(|e|e.error)?
    ///         .expect("SELECT statement must produce a cursor");
    ///
    ///     // Construct ODBC reader ...
    ///     let arrow_record_batches = OdbcReaderBuilder::new()
    ///         .build(cursor)?
    ///         // ... and make it concurrent
    ///         .into_concurrent()?;
    ///
    ///     for batch in arrow_record_batches {
    ///         // ... process batch ...
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn into_concurrent(self) -> Result<ConcurrentOdbcReader<C>, Error>
    where
        C: Send + 'static,
    {
        ConcurrentOdbcReader::from_block_cursor(
            self.batch_stream,
            self.converter,
            self.fallibale_allocations,
        )
    }

    /// Destroy the ODBC arrow reader and yield the underlyinng cursor object.
    ///
    /// One application of this is to process more than one result set in case you executed a stored
    /// procedure.
    pub fn into_cursor(self) -> Result<C, odbc_api::Error> {
        let (cursor, _buffer) = self.batch_stream.unbind()?;
        Ok(cursor)
    }

    /// Size of the internal preallocated buffer bound to the cursor and filled by your ODBC driver
    /// in rows. Each record batch will at most have this many rows. Only the last one may have
    /// less.
    pub fn max_rows_per_batch(&self) -> usize {
        self.batch_stream.row_array_size()
    }
}

impl<C> Iterator for OdbcReader<C>
where
    C: Cursor,
{
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.batch_stream.fetch_with_truncation_check(true) {
            // We successfully fetched a batch from the database. Try to copy it into a record batch
            // and forward errors if any.
            Ok(Some(batch)) => {
                let result_record_batch = self
                    .converter
                    .buffer_to_record_batch(batch)
                    .map_err(|mapping_error| ArrowError::ExternalError(Box::new(mapping_error)));
                Some(result_record_batch)
            }
            // We ran out of batches in the result set. End the iterator.
            Ok(None) => None,
            // We had an error fetching the next batch from the database, let's report it as an
            // external error.
            Err(odbc_error) => Some(Err(odbc_to_arrow_error(odbc_error))),
        }
    }
}

impl<C> RecordBatchReader for OdbcReader<C>
where
    C: Cursor,
{
    fn schema(&self) -> SchemaRef {
        self.converter.schema().clone()
    }
}

/// Creates instances of [`OdbcReader`] based on [`odbc_api::Cursor`].
///
/// Using a builder pattern instead of passing structs with all required arguments to the
/// constructors of [`OdbcReader`] allows `arrow_odbc` to introduce new paramters to fine tune the
/// creation and behavior of the readers without breaking the code of downstream applications.
#[derive(Default, Clone)]
pub struct OdbcReaderBuilder {
    /// `Some` implies the user has set this explicitly using
    /// [`OdbcReaderBuilder::with_max_num_rows_per_batch`]. `None` implies that we have to choose
    /// for the user.
    max_num_rows_per_batch: usize,
    max_bytes_per_batch: usize,
    schema: Option<SchemaRef>,
    max_text_size: Option<usize>,
    max_binary_size: Option<usize>,
    map_value_errors_to_null: bool,
    fallibale_allocations: bool,
    trim_fixed_sized_character_strings: bool,
}

impl OdbcReaderBuilder {
    pub fn new() -> Self {
        // In the abscence of an explicit row limit set by the user we choose u16 MAX (65535). This
        // is a reasonable high value to allow for siginificantly reducing IO overhead as opposed to
        // row by row fetching already. Likely for many database schemas a memory limitation will
        // kick in before this limit. If not however it can still be dangerous to go beyond this
        // number. Some drivers use a 16Bit integer to count rows and you can run into overflow
        // errors if you use one of them. Once such issue occurred with SAP anywhere.
        const DEFAULT_MAX_ROWS_PER_BATCH: usize = u16::MAX as usize;
        const DEFAULT_MAX_BYTES_PER_BATCH: usize = 512 * 1024 * 1024;

        OdbcReaderBuilder {
            max_num_rows_per_batch: DEFAULT_MAX_ROWS_PER_BATCH,
            max_bytes_per_batch: DEFAULT_MAX_BYTES_PER_BATCH,
            schema: None,
            max_text_size: None,
            max_binary_size: None,
            fallibale_allocations: false,
            map_value_errors_to_null: false,
            trim_fixed_sized_character_strings: false,
        }
    }

    /// Limits the maximum amount of rows which are fetched in a single roundtrip to the datasource.
    /// Higher numbers lower the IO overhead and may speed up your runtime, but also require larger
    /// preallocated buffers and use more memory. This value defaults to `65535` which is `u16` max.
    /// Some ODBC drivers use a 16Bit integer to count rows so this can avoid overflows. The
    /// improvements in saving IO overhead going above that number are estimated to be small. Your
    /// milage may vary of course.
    pub fn with_max_num_rows_per_batch(&mut self, max_num_rows_per_batch: usize) -> &mut Self {
        self.max_num_rows_per_batch = max_num_rows_per_batch;
        self
    }

    /// In addition to a row size limit you may specify an upper bound in bytes for allocating the
    /// transit buffer. This is useful if you do not know the database schema, or your code has to
    /// work with different ones, but you know the amount of memory in your machine. This limit is
    /// applied in addition to [`OdbcReaderBuilder::with_max_num_rows_per_batch`]. Whichever of
    /// these leads to a smaller buffer is used. This defaults to 512 MiB.
    pub fn with_max_bytes_per_batch(&mut self, max_bytes_per_batch: usize) -> &mut Self {
        self.max_bytes_per_batch = max_bytes_per_batch;
        self
    }

    /// Describes the types of the Arrow Arrays in the record batches. It is also used to determine
    /// CData type requested from the data source. If this is not explicitly set the type is infered
    /// from the schema information provided by the ODBC driver. A reason for setting this
    /// explicitly could be that you have superior knowledge about your data compared to the ODBC
    /// driver. E.g. a type for an unsigned byte (`u8`) is not part of the ODBC standard. Therfore
    /// the driver might at best be able to tell you that this is an (`i8`). If you want to still
    /// have `u8`s in the resulting array you need to specify the schema manually. Also many drivers
    /// struggle with reporting nullability correctly and just report every column as nullable.
    /// Explicitly specifying a schema can also compensate for such shortcomings if it turns out to
    /// be relevant.
    pub fn with_schema(&mut self, schema: SchemaRef) -> &mut Self {
        self.schema = Some(schema);
        self
    }

    /// An upper limit for the size of buffers bound to variadic text columns of the data source.
    /// This limit does not (directly) apply to the size of the created arrow buffers, but rather
    /// applies to the buffers used for the data in transit. Use this option if you have e.g.
    /// `VARCHAR(MAX)` fields in your database schema. In such a case without an upper limit, the
    /// ODBC driver of your data source is asked for the maximum size of an element, and is likely
    /// to answer with either `0` or a value which is way larger than any actual entry in the column
    /// If you can not adapt your database schema, this limit might be what you are looking for. On
    /// windows systems the size is double words (16Bit), as windows utilizes an UTF-16 encoding. So
    /// this translates to roughly the size in letters. On non windows systems this is the size in
    /// bytes and the datasource is assumed to utilize an UTF-8 encoding. If this method is not
    /// called no upper limit is set and the maximum element size, reported by ODBC is used to
    /// determine buffer sizes.
    pub fn with_max_text_size(&mut self, max_text_size: usize) -> &mut Self {
        self.max_text_size = Some(max_text_size);
        self
    }

    /// An upper limit for the size of buffers bound to variadic binary columns of the data source.
    /// This limit does not (directly) apply to the size of the created arrow buffers, but rather
    /// applies to the buffers used for the data in transit. Use this option if you have e.g.
    /// `VARBINARY(MAX)` fields in your database schema. In such a case without an upper limit, the
    /// ODBC driver of your data source is asked for the maximum size of an element, and is likely
    /// to answer with either `0` or a value which is way larger than any actual entry in the
    /// column. If you can not adapt your database schema, this limit might be what you are looking
    /// for. This is the maximum size in bytes of the binary column. If this method is not called no
    /// upper limit is set and the maximum element size, reported by ODBC is used to determine
    /// buffer sizes.
    pub fn with_max_binary_size(&mut self, max_binary_size: usize) -> &mut Self {
        self.max_binary_size = Some(max_binary_size);
        self
    }

    /// Set to `true` in order to trigger an [`crate::ColumnFailure::TooLarge`] instead of a panic
    /// in case the buffers can not be allocated due to their size. This might have a performance
    /// cost for constructing the reader. `false` by default.
    pub fn with_fallibale_allocations(&mut self, fallibale_allocations: bool) -> &mut Self {
        self.fallibale_allocations = fallibale_allocations;
        self
    }

    /// Set to `true` in order to map a value in the database which can not be successfully
    /// converted into its target type to NULL, rather than emitting an external Arrow Error.
    /// E.g. currently mapping errors can happen if a datetime value is not in the rang
    /// representable by arrow. Default is `false`.
    pub fn value_errors_as_null(&mut self, map_value_errors_to_null: bool) -> &mut Self {
        self.map_value_errors_to_null = map_value_errors_to_null;
        self
    }

    /// If set to `true` text in fixed sized character columns like e.g. CHAR are trimmed of
    /// whitespaces before converted into Arrow UTF-8 arrays. Default is `false`.
    pub fn trim_fixed_sized_characters(
        &mut self,
        fixed_sized_character_strings_are_trimmed: bool,
    ) -> &mut Self {
        self.trim_fixed_sized_character_strings = fixed_sized_character_strings_are_trimmed;
        self
    }

    /// No matter if the user explicitly specified a limit in row size, a memory limit, both or
    /// neither. In order to construct a reader we need to decide on the buffer size in rows.
    fn buffer_size_in_rows(&self, bytes_per_row: usize) -> Result<usize, Error> {
        // If schema is empty, return before division by zero error.
        if bytes_per_row == 0 {
            return Ok(self.max_bytes_per_batch);
        }
        let rows_per_batch = self.max_bytes_per_batch / bytes_per_row;
        if rows_per_batch == 0 {
            Err(Error::OdbcBufferTooSmall {
                max_bytes_per_batch: self.max_bytes_per_batch,
                bytes_per_row,
            })
        } else {
            Ok(min(self.max_num_rows_per_batch, rows_per_batch))
        }
    }

    /// Constructs an [`OdbcReader`] which consumes the giver cursor. The cursor will also be used
    /// to infer the Arrow schema if it has not been supplied explicitly.
    ///
    /// # Parameters
    ///
    /// * `cursor`: ODBC cursor used to fetch batches from the data source. The constructor will
    ///   bind buffers to this cursor in order to perform bulk fetches from the source. This is
    ///   usually faster than fetching results row by row as it saves roundtrips to the database.
    ///   The type of these buffers will be inferred from the arrow schema. Not every arrow type is
    ///   supported though.
    pub fn build<C>(&self, mut cursor: C) -> Result<OdbcReader<C>, Error>
    where
        C: Cursor,
    {
        let buffer_allocation_options = BufferAllocationOptions {
            max_text_size: self.max_text_size,
            max_binary_size: self.max_binary_size,
            fallibale_allocations: self.fallibale_allocations,
        };
        let converter = ToRecordBatch::new(
            &mut cursor,
            self.schema.clone(),
            buffer_allocation_options,
            self.map_value_errors_to_null,
            self.trim_fixed_sized_character_strings,
        )?;
        let bytes_per_row = converter.row_size_in_bytes();
        let buffer_size_in_rows = self.buffer_size_in_rows(bytes_per_row)?;
        let row_set_buffer =
            converter.allocate_buffer(buffer_size_in_rows, self.fallibale_allocations)?;
        let batch_stream = cursor.bind_buffer(row_set_buffer).unwrap();

        Ok(OdbcReader {
            converter,
            batch_stream,
            fallibale_allocations: self.fallibale_allocations,
        })
    }
}

pub fn odbc_to_arrow_error(odbc_error: odbc_api::Error) -> ArrowError {
    ArrowError::from_external_error(Box::new(odbc_error))
}
