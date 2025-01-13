use arrow::buffer::ScalarBuffer;
use arrow::error::Result;
use arrow::{array::*, buffer::Buffer};
use arrow_convert::field::ArrowField;
use arrow_convert::{deserialize::*, serialize::*, ArrowDeserialize, ArrowField, ArrowSerialize};

#[test]
fn test_deserialize_iterator() {
    use arrow::array::*;
    use arrow_convert::deserialize::*;
    use arrow_convert::serialize::*;
    use std::borrow::Borrow;

    #[derive(Debug, Clone, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    struct S {
        a1: i64,
    }

    let original_array = [S { a1: 1 }, S { a1: 100 }, S { a1: 1000 }];
    let b: ArrayRef = original_array.try_into_arrow().unwrap();
    let iter = arrow_array_deserialize_iterator::<S>(b.borrow()).unwrap();
    for (i, k) in iter.zip(original_array.iter()) {
        assert_eq!(&i, k);
    }

    let original_array = [Some(Some(1_i32)), Some(Some(100)), Some(None), None];
    let expected = [Some(Some(1_i32)), Some(Some(100)), None, None];
    let b: ArrayRef = original_array.try_into_arrow().unwrap();
    let iter = arrow_array_deserialize_iterator::<Option<Option<i32>>>(b.borrow()).unwrap();
    for (i, k) in iter.zip(expected.iter()) {
        assert_eq!(&i, k);
    }
}

fn data_mismatch_error<Expected: ArrowField, Actual: ArrowField>() -> arrow::error::ArrowError {
    arrow::error::ArrowError::InvalidArgumentError(format!(
        "Data type mismatch. Expected type={:#?} is_nullable={}, but was type={:#?} is_nullable={}",
        Expected::data_type(),
        Expected::is_nullable(),
        Actual::data_type(),
        Actual::is_nullable()
    ))
}

#[test]
fn test_deserialize_schema_mismatch_error() {
    #[derive(Debug, Clone, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    struct S1 {
        a: i64,
    }
    #[derive(Debug, Clone, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    struct S2 {
        a: String,
    }

    let arr1 = vec![S1 { a: 1 }, S1 { a: 2 }];
    let arr1: ArrayRef = arr1.try_into_arrow().unwrap();
    let result: Result<Vec<S2>> = arr1.try_into_collection();
    assert_eq!(
        result.unwrap_err().to_string(),
        data_mismatch_error::<S2, S1>().to_string()
    );

    let arr1 = vec![S1 { a: 1 }, S1 { a: 2 }];
    let arr1: ArrayRef = arr1.try_into_arrow().unwrap();
    let result: Result<Vec<_>> = arr1.try_into_collection_as_type::<S2>();
    assert_eq!(
        result.unwrap_err().to_string(),
        data_mismatch_error::<S2, S1>().to_string()
    );
}

#[test]
fn test_deserialize_large_types_schema_mismatch_error() {
    #[derive(Debug, Clone, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    struct S1 {
        a: String,
    }
    #[derive(Debug, Clone, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    struct S2 {
        #[arrow_field(type = "arrow_convert::field::LargeString")]
        a: String,
    }

    let arr1 = vec![S1 { a: "123".to_string() }, S1 { a: "333".to_string() }];
    let arr1: ArrayRef = arr1.try_into_arrow().unwrap();

    let result: Result<Vec<S2>> = arr1.try_into_collection();
    assert_eq!(
        result.unwrap_err().to_string(),
        data_mismatch_error::<S2, S1>().to_string()
    );
}

#[test]
fn test_deserialize_scalar_buffer_u16() {
    // test Buffer
    let original_array = [Buffer::from_iter(0u16..5), Buffer::from_iter(7..9)];
    let b: ArrayRef = original_array.try_into_arrow().unwrap();
    let iter = arrow_array_deserialize_iterator::<Buffer>(b.as_ref()).unwrap();
    for (i, k) in iter.zip(original_array.iter()) {
        assert_eq!(&i, k);
    }

    // test ScalarBuffer<u8>
    let original_array = [ScalarBuffer::from_iter(0u16..5), ScalarBuffer::from_iter(7..9)];
    let b: ArrayRef = original_array.try_into_arrow().unwrap();
    let iter = arrow_array_deserialize_iterator::<ScalarBuffer<u16>>(b.as_ref()).unwrap();
    for (i, k) in iter.zip(original_array.iter()) {
        assert_eq!(&i, k);
    }
}

#[test]
fn test_deserialize_scalar_buffer_u8() {
    let original_array = [ScalarBuffer::from_iter(0u8..5), ScalarBuffer::from_iter(7..9)];
    let b: ArrayRef = original_array.try_into_arrow().unwrap();
    let iter = arrow_array_deserialize_iterator::<ScalarBuffer<u8>>(b.as_ref()).unwrap();
    for (i, k) in iter.zip(original_array.iter()) {
        assert_eq!(&i, k);
    }

    let original_array = [
        Some(ScalarBuffer::from_iter(0u8..5)),
        None,
        Some(ScalarBuffer::from_iter(7..9)),
    ];
    let b: ArrayRef = original_array.try_into_arrow().unwrap();
    let iter = arrow_array_deserialize_iterator::<Option<ScalarBuffer<u8>>>(b.as_ref()).unwrap();
    for (i, k) in iter.zip(original_array.iter()) {
        assert_eq!(&i, k);
    }
}
