#[cfg(feature = "rust_decimal")]
#[test]
fn test_decimal_roundtrip() {
    use arrow::array::{Array, ArrayRef};
    use arrow::datatypes::DECIMAL_DEFAULT_SCALE;
    use arrow::{array::Decimal128Array, datatypes::DECIMAL128_MAX_PRECISION};
    use arrow_convert::deserialize::TryIntoCollection;
    use arrow_convert::serialize::*;
    use pretty_assertions::assert_eq;
    use rust_decimal::Decimal;

    let original: Vec<Decimal> = vec![
        Decimal::from_str_exact("123.45").unwrap(),
        Decimal::from_str_exact("67890").unwrap(),
        Decimal::from_str_exact("0.11111").unwrap(),
        Decimal::from_str_exact("67890").unwrap(),
        Decimal::from_str_exact("0.11111").unwrap(),
        Decimal::from_str_exact("-9876.54321").unwrap(),
        Decimal::from_str_exact("1000000.000001").unwrap(),
        Decimal::from_str_exact("0.0000000001").unwrap(),
        Decimal::from_str_exact("-0.9999999999").unwrap(),
    ];

    let arrow_array: ArrayRef = original.try_into_arrow().expect("Failed to convert to Arrow array");
    assert!(arrow_array.as_any().is::<Decimal128Array>());

    let decimal_array = arrow_array
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("Failed to downcast to Decimal128Array");

    assert_eq!(decimal_array.precision(), DECIMAL128_MAX_PRECISION);
    assert_eq!(decimal_array.scale(), DECIMAL_DEFAULT_SCALE);

    let roundtrip: Vec<Decimal> = arrow_array
        .try_into_collection()
        .expect("Failed to convert back to Vec<Decimal>");
    assert_eq!(original, roundtrip);
}

#[cfg(feature = "rust_decimal")]
#[test]
fn test_decimal_edge_values() {
    use arrow::array::{Array, ArrayRef};
    use arrow::datatypes::DECIMAL_DEFAULT_SCALE;
    use arrow::{array::Decimal128Array, datatypes::DECIMAL128_MAX_PRECISION};
    use arrow_convert::deserialize::TryIntoCollection;
    use arrow_convert::serialize::*;
    use rust_decimal::Decimal;

    let original: Vec<Decimal> = vec![
        Decimal::new(i64::MAX, DECIMAL_DEFAULT_SCALE as _),
        Decimal::new(i64::MIN, DECIMAL_DEFAULT_SCALE as _),
        Decimal::new(0, DECIMAL_DEFAULT_SCALE as _),
    ];

    let arrow_array: ArrayRef = original.try_into_arrow().expect("Failed to convert to Arrow array");
    assert!(arrow_array.as_any().is::<Decimal128Array>());

    let decimal_array = arrow_array.as_any().downcast_ref::<Decimal128Array>().unwrap();
    assert_eq!(decimal_array.precision(), DECIMAL128_MAX_PRECISION);
    assert_eq!(decimal_array.scale(), DECIMAL_DEFAULT_SCALE);

    let roundtrip: Vec<Decimal> = arrow_array
        .try_into_collection()
        .expect("Failed to convert from Arrow array");
    assert_eq!(original, roundtrip);
}
