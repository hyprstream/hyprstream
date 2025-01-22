use arrow::array::{Int64Array, Float64Array, StringArray};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::DataType;
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
                    let array = col.as_any().downcast_ref::<Int64Array>().unwrap();
                    JsonValue::Number(array.value(row_idx).into())
                }
                DataType::Float64 => {
                    let array = col.as_any().downcast_ref::<Float64Array>().unwrap();
                    JsonValue::Number(serde_json::Number::from_f64(array.value(row_idx)).unwrap())
                }
                _ => {
                    let array = col.as_any().downcast_ref::<StringArray>().unwrap();
                    JsonValue::String(array.value(row_idx).to_string())
                }
            };
            row.insert(col_name.to_string(), value);
        }
        json_rows.push(JsonValue::Object(row));
    }
    Ok(json_rows)
}