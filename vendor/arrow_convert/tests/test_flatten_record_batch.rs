use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::{array::*, datatypes::DataType};
use arrow_convert::{serialize::*, ArrowField, ArrowSerialize};
use std::sync::Arc;

#[test]
fn test_flatten_chunk() {
    #[derive(Debug, Clone, ArrowField, ArrowSerialize)]
    struct Struct {
        a: i64,
        b: i64,
    }

    let target = RecordBatch::try_from_iter([
        (
            "a",
            Arc::new(Int64Array::from_iter(&[Some(1), Some(2)])) as ArrayRef,
        ),
        (
            "b",
            Arc::new(Int64Array::from_iter(&[Some(1), Some(2)])) as ArrayRef,
        ),
    ])
    .unwrap();

    let array = vec![Struct { a: 1, b: 1 }, Struct { a: 2, b: 2 }];

    let array: ArrayRef = array.try_into_arrow().unwrap();
    let chunk: RecordBatch = RecordBatch::try_from_iter([("struct", array)]).unwrap();

    let flattened: RecordBatch = chunk.flatten().unwrap();

    assert_eq!(flattened, target);
}

#[test]
fn test_flatten_chunk_empty_chunk_error() {
    let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
    let chunk: RecordBatch = RecordBatch::new_empty(Arc::new(schema));
    assert!(chunk.flatten().is_err());
}

#[test]
fn test_flatten_chunk_no_single_struct_array_error() {
    #[derive(Debug, Clone, ArrowField, ArrowSerialize)]
    struct Struct {
        a: i64,
        b: String,
    }

    let array = vec![
        Struct {
            a: 1,
            b: "one".to_string(),
        },
        Struct {
            a: 2,
            b: "two".to_string(),
        },
    ];

    let array: ArrayRef = array.try_into_arrow().unwrap();

    let arrays = vec![("s1", array.clone()), ("s2", array.clone())];
    let chunk = RecordBatch::try_from_iter(arrays).unwrap();

    assert!(chunk.flatten().is_err());
}

#[test]
fn test_flatten_chunk_type_not_struct_error() {
    let array: ArrayRef = Arc::new(Int32Array::from_iter(&[Some(1), None, Some(3)]));
    let chunk = RecordBatch::try_from_iter(vec![("array", array)]).unwrap();

    assert!(chunk.flatten().is_err());
}
