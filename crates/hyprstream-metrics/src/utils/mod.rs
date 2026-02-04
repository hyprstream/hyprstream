use duckdb::arrow::array::{Int64Array, Float64Array, StringArray};
use duckdb::arrow::array::RecordBatch;
use duckdb::arrow::datatypes::DataType;
use serde_json::{Map, Value as JsonValue};
use std::error::Error;

/// Convert an Arrow RecordBatch to a JSON array of objects
pub fn record_batch_to_json(batch: &RecordBatch) -> Result<Vec<JsonValue>, Box<dyn Error>> {
    let mut json_rows = Vec::new();
    for row_idx in 0..batch.num_rows() {
        let mut row = Map::new();
        for col_idx in 0..batch.num_columns() {
            let col = batch.column(col_idx);
            let schema = batch.schema();
            let field = schema.field(col_idx);
            let col_name = field.name();
            let value = match col.data_type() {
                DataType::Int64 => {
                    let array = col.as_any().downcast_ref::<Int64Array>()
                        .ok_or("Failed to downcast to Int64Array")?;
                    JsonValue::Number(array.value(row_idx).into())
                }
                DataType::Float64 => {
                    let array = col.as_any().downcast_ref::<Float64Array>()
                        .ok_or("Failed to downcast to Float64Array")?;
                    let num = serde_json::Number::from_f64(array.value(row_idx))
                        .ok_or("Failed to convert f64 to JSON number")?;
                    JsonValue::Number(num)
                }
                _ => {
                    let array = col.as_any().downcast_ref::<StringArray>()
                        .ok_or("Failed to downcast to StringArray")?;
                    JsonValue::String(array.value(row_idx).to_owned())
                }
            };
            row.insert(col_name.clone(), value);
        }
        json_rows.push(JsonValue::Object(row));
    }
    Ok(json_rows)
}