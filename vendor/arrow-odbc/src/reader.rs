use std::{convert::TryInto, sync::Arc};

use arrow::{
    array::{ArrayRef, BooleanBuilder},
    datatypes::{
        DataType as ArrowDataType, Date32Type, Field, Float32Type, Float64Type, Int16Type,
        Int32Type, Int64Type, Int8Type, TimeUnit, TimestampMicrosecondType,
        TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type,
    },
};

use log::debug;
use odbc_api::{
    buffers::{AnySlice, BufferDesc, Item},
    Bit, DataType as OdbcDataType, ResultSetMetadata,
};
use thiserror::Error;

mod binary;
mod concurrent_odbc_reader;
mod decimal;
mod map_odbc_to_arrow;
mod odbc_reader;
mod text;
mod to_record_batch;

use crate::date_time::{
    days_since_epoch, ms_since_epoch, ns_since_epoch, seconds_since_epoch, us_since_epoch,
};

pub use self::{
    binary::{Binary, FixedSizedBinary},
    concurrent_odbc_reader::ConcurrentOdbcReader,
    decimal::Decimal,
    map_odbc_to_arrow::{MapOdbcToArrow, MappingError},
    odbc_reader::{OdbcReader, OdbcReaderBuilder},
    text::choose_text_strategy,
};

/// All decisions needed to copy data from an ODBC buffer to an Arrow Array
pub trait ReadStrategy {
    /// Describes the buffer which is bound to the ODBC cursor.
    fn buffer_desc(&self) -> BufferDesc;

    /// Create an arrow array from an ODBC buffer described in [`Self::buffer_description`].
    fn fill_arrow_array(&self, column_view: AnySlice) -> Result<ArrayRef, MappingError>;
}

pub struct NonNullableBoolean;

impl ReadStrategy for NonNullableBoolean {
    fn buffer_desc(&self) -> BufferDesc {
        BufferDesc::Bit { nullable: false }
    }

    fn fill_arrow_array(&self, column_view: AnySlice) -> Result<ArrayRef, MappingError> {
        let values = Bit::as_slice(column_view).unwrap();
        let mut builder = BooleanBuilder::new();
        for bit in values {
            builder.append_value(bit.as_bool());
        }
        Ok(Arc::new(builder.finish()))
    }
}

pub struct NullableBoolean;

impl ReadStrategy for NullableBoolean {
    fn buffer_desc(&self) -> BufferDesc {
        BufferDesc::Bit { nullable: true }
    }

    fn fill_arrow_array(&self, column_view: AnySlice) -> Result<ArrayRef, MappingError> {
        let values = Bit::as_nullable_slice(column_view).unwrap();
        let mut builder = BooleanBuilder::new();
        for bit in values {
            builder.append_option(bit.copied().map(Bit::as_bool))
        }
        Ok(Arc::new(builder.finish()))
    }
}

/// Allows setting limits for buffers bound to the ODBC data source. Check this out if you find that
/// you get memory allocation, or zero sized column errors. Used than constructing a reader using
/// [`crate::OdbcReaderBuilder`].
#[derive(Default, Debug, Clone, Copy)]
pub struct BufferAllocationOptions {
    /// An upper limit for the size of buffers bound to variadic text columns of the data source.
    /// This limit does not (directly) apply to the size of the created arrow buffers, but rather
    /// applies to the buffers used for the data in transit. Use this option if you have e.g.
    /// `VARCHAR(MAX)` fields in your database schema. In such a case without an upper limit, the
    /// ODBC driver of your data source is asked for the maximum size of an element, and is likely
    /// to answer with either `0` or a value which is way larger than any actual entry in the column
    /// If you can not adapt your database schema, this limit might be what you are looking for. On
    /// windows systems the size is double words (16Bit), as windows utilizes an UTF-16 encoding. So
    /// this translates to roughly the size in letters. On non windows systems this is the size in
    /// bytes and the datasource is assumed to utilize an UTF-8 encoding. `None` means no upper
    /// limit is set and the maximum element size, reported by ODBC is used to determine buffer
    /// sizes.
    pub max_text_size: Option<usize>,
    /// An upper limit for the size of buffers bound to variadic binary columns of the data source.
    /// This limit does not (directly) apply to the size of the created arrow buffers, but rather
    /// applies to the buffers used for the data in transit. Use this option if you have e.g.
    /// `VARBINARY(MAX)` fields in your database schema. In such a case without an upper limit, the
    /// ODBC driver of your data source is asked for the maximum size of an element, and is likely
    /// to answer with either `0` or a value which is way larger than any actual entry in the
    /// column. If you can not adapt your database schema, this limit might be what you are looking
    /// for. This is the maximum size in bytes of the binary column.
    pub max_binary_size: Option<usize>,
    /// Set to `true` in order to trigger an [`ColumnFailure::TooLarge`] instead of a panic in case
    /// the buffers can not be allocated due to their size. This might have a performance cost for
    /// constructing the reader. `false` by default.
    pub fallibale_allocations: bool,
}

pub fn choose_column_strategy(
    field: &Field,
    query_metadata: &mut impl ResultSetMetadata,
    col_index: u16,
    buffer_allocation_options: BufferAllocationOptions,
    map_value_errors_to_null: bool,
    trim_fixed_sized_character_strings: bool,
) -> Result<Box<dyn ReadStrategy + Send>, ColumnFailure> {
    let strat: Box<dyn ReadStrategy + Send> = match field.data_type() {
        ArrowDataType::Boolean => {
            if field.is_nullable() {
                Box::new(NullableBoolean)
            } else {
                Box::new(NonNullableBoolean)
            }
        }
        ArrowDataType::Int8 => Int8Type::identical(field.is_nullable()),
        ArrowDataType::Int16 => Int16Type::identical(field.is_nullable()),
        ArrowDataType::Int32 => Int32Type::identical(field.is_nullable()),
        ArrowDataType::Int64 => Int64Type::identical(field.is_nullable()),
        ArrowDataType::UInt8 => UInt8Type::identical(field.is_nullable()),
        ArrowDataType::Float32 => Float32Type::identical(field.is_nullable()),
        ArrowDataType::Float64 => Float64Type::identical(field.is_nullable()),
        ArrowDataType::Date32 => Date32Type::map_infalliable(field.is_nullable(), days_since_epoch),
        ArrowDataType::Utf8 => {
            let sql_type = query_metadata
                .col_data_type(col_index)
                .map_err(ColumnFailure::FailedToDescribeColumn)?;
            // Use a zero based index here, because we use it everywhere else there we communicate
            // with users.
            debug!("Relational type of column {}: {sql_type:?}", col_index - 1);
            let lazy_display_size = || query_metadata.col_display_size(col_index);
            // Use the SQL type first to determine buffer length.
            choose_text_strategy(
                sql_type,
                lazy_display_size,
                buffer_allocation_options.max_text_size,
                trim_fixed_sized_character_strings,
            )?
        }
        ArrowDataType::Decimal128(precision, scale @ 0..) => {
            Box::new(Decimal::new(*precision, *scale))
        }
        ArrowDataType::Binary => {
            let sql_type = query_metadata
                .col_data_type(col_index)
                .map_err(ColumnFailure::FailedToDescribeColumn)?;
            let length = sql_type.column_size();
            let length = match (length, buffer_allocation_options.max_binary_size) {
                (None, None) => return Err(ColumnFailure::ZeroSizedColumn { sql_type }),
                (None, Some(limit)) => limit,
                (Some(len), None) => len.get(),
                (Some(len), Some(limit)) => {
                    if len.get() < limit {
                        len.get()
                    } else {
                        limit
                    }
                }
            };
            Box::new(Binary::new(length))
        }
        ArrowDataType::Timestamp(TimeUnit::Second, _) => {
            TimestampSecondType::map_infalliable(field.is_nullable(), seconds_since_epoch)
        }
        ArrowDataType::Timestamp(TimeUnit::Millisecond, _) => {
            TimestampMillisecondType::map_infalliable(field.is_nullable(), ms_since_epoch)
        }
        ArrowDataType::Timestamp(TimeUnit::Microsecond, _) => {
            TimestampMicrosecondType::map_infalliable(field.is_nullable(), us_since_epoch)
        }
        ArrowDataType::Timestamp(TimeUnit::Nanosecond, _) => {
            TimestampNanosecondType::map_falliable(
                field.is_nullable(),
                map_value_errors_to_null,
                ns_since_epoch,
            )
        }
        ArrowDataType::FixedSizeBinary(length) => {
            Box::new(FixedSizedBinary::new((*length).try_into().unwrap()))
        }
        unsupported_arrow_type => {
            return Err(ColumnFailure::UnsupportedArrowType(
                unsupported_arrow_type.clone(),
            ))
        }
    };
    Ok(strat)
}

/// Read error related to a specific column
#[derive(Error, Debug)]
pub enum ColumnFailure {
    /// We are getting a display or column size from ODBC but it is not larger than 0.
    #[error(
        "The ODBC driver did not specify a sensible upper bound for the column. This usually \
        happens for large variadic types (E.g. VARCHAR(max)). In other cases it can be a \
        shortcoming of the ODBC driver. Try casting the column into a type with a sensible upper \
        bound. `arrow-odbc` also allows the application to specify a generic upper bound, which it \
        would automatically apply. The type of the column causing this error is {:?}.",
        sql_type
    )]
    ZeroSizedColumn { sql_type: OdbcDataType },
    /// Unable to retrieve the column display size for the column.
    #[error(
        "Unable to deduce the maximum string length for the SQL Data Type reported by the ODBC \
        driver. Reported SQL data type is: {:?}.\n Error fetching column display or octet size: \
        {source}",
        sql_type
    )]
    UnknownStringLength {
        sql_type: OdbcDataType,
        source: odbc_api::Error,
    },
    /// The type specified in the arrow schema is not supported to be fetched from the database.
    #[error(
        "Unsupported arrow type: `{0}`. This type can currently not be fetched from an ODBC data \
        source by an instance of OdbcReader."
    )]
    UnsupportedArrowType(ArrowDataType),
    /// At ODBC api calls gaining information about the columns did fail.
    #[error(
        "An error occurred fetching the column description or data type from the metainformation \
        attached to the ODBC result set:\n{0}"
    )]
    FailedToDescribeColumn(#[source] odbc_api::Error),
    #[error(
        "Column buffer is too large to be allocated. Tried to alloacte {num_elements} elements \
        with {element_size} bytes in size each."
    )]
    TooLarge {
        num_elements: usize,
        element_size: usize,
    },
}

impl ColumnFailure {
    /// Provides the error with additional context of Error with column name and index.
    pub fn into_crate_error(self, name: String, index: usize) -> crate::Error {
        crate::Error::ColumnFailure {
            name,
            index,
            source: self,
        }
    }
}
