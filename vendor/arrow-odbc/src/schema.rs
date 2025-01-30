use arrow::datatypes::{DataType as ArrowDataType, Field, Schema, TimeUnit};
use log::debug;
use odbc_api::{ColumnDescription, DataType as OdbcDataType, ResultSetMetadata};
use std::convert::TryInto;

use crate::{ColumnFailure, Error};

/// Query the metadata to create an arrow schema. This method is invoked automatically for you by
/// [`crate::OdbcReaderBuilder::build`]. You may want to call this method in situtation ther you want
/// to create an arrow schema without creating the reader yet.
///
/// # Example
///
/// ```
/// use anyhow::Error;
///
/// use arrow_odbc::{arrow_schema_from, arrow::datatypes::Schema, odbc_api::Connection};
///
/// fn fetch_schema_for_table(
///     table_name: &str,
///     connection: &Connection<'_>
/// ) -> Result<Schema, Error> {
///     // Query column with values to get a cursor
///     let sql = format!("SELECT * FROM {}", table_name);
///     let mut prepared = connection.prepare(&sql)?;
///     
///     // Now that we have prepared statement, we want to use it to query metadata.
///     let map_errors_to_null = false;
///     let schema = arrow_schema_from(&mut prepared, map_errors_to_null)?;
///     Ok(schema)
/// }
/// ```
pub fn arrow_schema_from(
    resut_set_metadata: &mut impl ResultSetMetadata,
    map_value_errors_to_null: bool,
) -> Result<Schema, Error> {
    let num_cols: u16 = resut_set_metadata
        .num_result_cols()
        .map_err(Error::UnableToRetrieveNumCols)?
        .try_into()
        .unwrap();
    let mut fields = Vec::new();
    for index in 0..num_cols {
        let field = arrow_field_from(resut_set_metadata, index, map_value_errors_to_null)?;

        fields.push(field)
    }
    Ok(Schema::new(fields))
}

fn arrow_field_from(
    resut_set_metadata: &mut impl ResultSetMetadata,
    index: u16,
    map_value_errors_to_null: bool,
) -> Result<Field, Error> {
    let mut column_description = ColumnDescription::default();
    resut_set_metadata
        .describe_col(index + 1, &mut column_description)
        .map_err(|cause| Error::ColumnFailure {
            name: "Unknown".to_owned(),
            index: index as usize,
            source: ColumnFailure::FailedToDescribeColumn(cause),
        })?;
    let name = column_description
        .name_to_string()
        .map_err(|source| Error::EncodingInvalid { source })?;
    debug!(
        "ODBC driver reported for column {index}. Relational type: {:?}; Nullability: {:?}; \
            Name: '{name}';",
        column_description.data_type, column_description.nullability
    );
    let data_type = match column_description.data_type {
        OdbcDataType::Numeric {
            precision: p @ 0..=38,
            scale,
        }
        | OdbcDataType::Decimal {
            precision: p @ 0..=38,
            scale,
        } => ArrowDataType::Decimal128(p as u8, scale.try_into().unwrap()),
        OdbcDataType::Integer => ArrowDataType::Int32,
        OdbcDataType::SmallInt => ArrowDataType::Int16,
        OdbcDataType::Real | OdbcDataType::Float { precision: 0..=24 } => ArrowDataType::Float32,
        OdbcDataType::Float { precision: _ } | OdbcDataType::Double => ArrowDataType::Float64,
        OdbcDataType::Date => ArrowDataType::Date32,
        OdbcDataType::Timestamp { precision: 0 } => {
            ArrowDataType::Timestamp(TimeUnit::Second, None)
        }
        OdbcDataType::Timestamp { precision: 1..=3 } => {
            ArrowDataType::Timestamp(TimeUnit::Millisecond, None)
        }
        OdbcDataType::Timestamp { precision: 4..=6 } => {
            ArrowDataType::Timestamp(TimeUnit::Microsecond, None)
        }
        OdbcDataType::Timestamp { precision: _ } => {
            ArrowDataType::Timestamp(TimeUnit::Nanosecond, None)
        }
        OdbcDataType::BigInt => ArrowDataType::Int64,
        OdbcDataType::TinyInt => {
            let is_unsigned = resut_set_metadata
                .column_is_unsigned(index + 1)
                .map_err(|e| Error::ColumnFailure {
                    name: name.clone(),
                    index: index as usize,
                    source: ColumnFailure::FailedToDescribeColumn(e),
                })?;
            if is_unsigned {
                ArrowDataType::UInt8
            } else {
                ArrowDataType::Int8
            }
        }
        OdbcDataType::Bit => ArrowDataType::Boolean,
        OdbcDataType::Binary { length } => {
            let length = length
                .ok_or_else(|| Error::ColumnFailure {
                    name: name.clone(),
                    index: index as usize,
                    source: ColumnFailure::ZeroSizedColumn {
                        sql_type: OdbcDataType::Binary { length },
                    },
                })?
                .get()
                .try_into()
                .unwrap();
            ArrowDataType::FixedSizeBinary(length)
        }
        OdbcDataType::LongVarbinary { length: _ } | OdbcDataType::Varbinary { length: _ } => {
            ArrowDataType::Binary
        }
        OdbcDataType::Unknown
        | OdbcDataType::Time { precision: _ }
        | OdbcDataType::Numeric { .. }
        | OdbcDataType::Decimal { .. }
        | OdbcDataType::Other {
            data_type: _,
            column_size: _,
            decimal_digits: _,
        }
        | OdbcDataType::WChar { length: _ }
        | OdbcDataType::Char { length: _ }
        | OdbcDataType::WVarchar { length: _ }
        | OdbcDataType::LongVarchar { length: _ }
        | OdbcDataType::Varchar { length: _ } => ArrowDataType::Utf8,
    };
    let is_falliable = matches!(data_type, ArrowDataType::Timestamp(TimeUnit::Nanosecond, _));
    let nullable =
        column_description.could_be_nullable() || (is_falliable && map_value_errors_to_null);
    let field = Field::new(name, data_type, nullable);
    Ok(field)
}
