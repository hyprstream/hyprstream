use std::error::Error;

pub mod validation;

// /// Convert an Arrow RecordBatch to a JSON array of objects
// pub fn record_batch_to_json(batch: &RecordBatch) -> Result<Vec<JsonValue>, Box<dyn Error>> {
//     let mut json_rows = Vec::new();
//     for row_idx in 0..batch.num_rows() {
//         let mut row = Map::new();
//         for col_idx in 0..batch.num_columns() {
//             let col = batch.column(col_idx);
//             let schema = batch.schema();
//             let field = schema.field(col_idx);
//             let col_name = field.name();
//             let value = match col.data_type() {
//                 DataType::Int64 => {
//                     let array = col.as_any().downcast_ref::<Int64Array>().unwrap();
//                     JsonValue::Number(array.value(row_idx).into())
//                 }
//                 DataType::Float64 => {
//                     let array = col.as_any().downcast_ref::<Float64Array>().unwrap();
//                     JsonValue::Number(serde_json::Number::from_f64(array.value(row_idx)).unwrap())
//                 }
//                 _ => {
//                     let array = col.as_any().downcast_ref::<StringArray>().unwrap();
//                     JsonValue::String(array.value(row_idx).to_string())
//                 }
//             };
//             row.insert(col_name.to_string(), value);
//         }
//         json_rows.push(JsonValue::Object(row));
//     }
//     Ok(json_rows)
// }

/// Sanitize a filename to be safe for filesystem storage
pub fn sanitize_filename(name: &str) -> String {
    name.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_")
        .replace(' ', "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("hello/world"), "hello_world");
        assert_eq!(sanitize_filename("model:v1.0"), "model_v1.0");
        assert_eq!(sanitize_filename("test file.safetensors"), "test_file.safetensors");
        assert_eq!(sanitize_filename("normal_name"), "normal_name");
        assert_eq!(sanitize_filename(""), "");
    }
}