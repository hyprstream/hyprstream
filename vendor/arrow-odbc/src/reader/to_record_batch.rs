use std::sync::Arc;

use arrow::{
    datatypes::{Schema, SchemaRef},
    record_batch::RecordBatch,
};
use log::info;
use odbc_api::{buffers::ColumnarAnyBuffer, ResultSetMetadata};

use crate::{arrow_schema_from, BufferAllocationOptions, ColumnFailure, Error};

use super::{choose_column_strategy, MappingError, ReadStrategy};

/// Transforms batches fetched from an ODBC data source in a
/// [`odbc_api::bufferers::ColumnarAnyBuffer`] into arrow tables of the specified schemas. It also
/// allocates the buffers to hold the ODBC batches with the matching buffer descriptions.
pub struct ToRecordBatch {
    /// Must contain one item for each field in [`Self::schema`]. Encapsulates all the column type
    /// specific decisions which go into filling an Arrow array from an ODBC data source.
    column_strategies: Vec<Box<dyn ReadStrategy + Send>>,
    /// Arrow schema describing the arrays we want to fill from the Odbc data source.
    schema: SchemaRef,
}

impl ToRecordBatch {
    pub fn new(
        cursor: &mut impl ResultSetMetadata,
        schema: Option<SchemaRef>,
        buffer_allocation_options: BufferAllocationOptions,
        map_value_errors_to_null: bool,
        trim_fixed_sized_character_strings: bool,
    ) -> Result<Self, Error> {
        // Infer schema if not given by the user
        let schema = if let Some(schema) = schema {
            schema
        } else {
            Arc::new(arrow_schema_from(cursor, map_value_errors_to_null)?)
        };

        let column_strategies: Vec<Box<dyn ReadStrategy + Send>> = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(index, field)| {
                let col_index = (index + 1).try_into().unwrap();
                choose_column_strategy(
                    field,
                    cursor,
                    col_index,
                    buffer_allocation_options,
                    map_value_errors_to_null,
                    trim_fixed_sized_character_strings,
                )
                .map_err(|cause| cause.into_crate_error(field.name().clone(), index))
            })
            .collect::<Result<_, _>>()?;

        Ok(ToRecordBatch {
            column_strategies,
            schema,
        })
    }

    /// Logs buffer description and sizes
    pub fn row_size_in_bytes(&self) -> usize {
        let mut total_bytes = 0;
        for (read, field) in self.column_strategies.iter().zip(self.schema.fields()) {
            let name = field.name();
            let desc = read.buffer_desc();
            let bytes_per_row = desc.bytes_per_row();
            info!("Column '{name}'\nBytes used per row: {bytes_per_row}");
            total_bytes += bytes_per_row;
        }
        info!("Total memory usage per row for single transit buffer: {total_bytes}");
        total_bytes
    }

    pub fn allocate_buffer(
        &self,
        max_batch_size: usize,
        fallibale_allocations: bool,
    ) -> Result<ColumnarAnyBuffer, Error> {
        let descs = self.column_strategies.iter().map(|cs| cs.buffer_desc());

        let row_set_buffer = if fallibale_allocations {
            ColumnarAnyBuffer::try_from_descs(max_batch_size, descs)
                .map_err(|err| map_allocation_error(err, &self.schema))?
        } else {
            ColumnarAnyBuffer::from_descs(max_batch_size, descs)
        };
        Ok(row_set_buffer)
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub fn buffer_to_record_batch(
        &self,
        odbc_buffer: &ColumnarAnyBuffer,
    ) -> Result<RecordBatch, MappingError> {
        let arrow_columns = self
            .column_strategies
            .iter()
            .enumerate()
            .map(|(index, strat)| {
                let column_view = odbc_buffer.column(index);
                strat.fill_arrow_array(column_view)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let record_batch = RecordBatch::try_new(self.schema.clone(), arrow_columns).unwrap();
        Ok(record_batch)
    }
}

fn map_allocation_error(error: odbc_api::Error, schema: &Schema) -> Error {
    match error {
        odbc_api::Error::TooLargeColumnBufferSize {
            buffer_index,
            num_elements,
            element_size,
        } => Error::ColumnFailure {
            name: schema.field(buffer_index as usize).name().clone(),
            index: buffer_index as usize,
            source: ColumnFailure::TooLarge {
                num_elements,
                element_size,
            },
        },
        _ => {
            panic!("Unexpected error in upstream ODBC api error library")
        }
    }
}
