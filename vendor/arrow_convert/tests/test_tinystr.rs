#[cfg(feature = "tinystr")]
#[test]
fn test_tinyasciistr_roundtrip() {
    use arrow::array::{Array, ArrayRef, FixedSizeBinaryArray};
    use arrow_convert::deserialize::TryIntoCollection;
    use arrow_convert::serialize::TryIntoArrow;
    use tinystr::TinyAsciiStr;

    let original: Vec<TinyAsciiStr<3>> = vec![
        TinyAsciiStr::try_from_str("ABC").unwrap(),
        TinyAsciiStr::try_from_str("XYZ").unwrap(),
        TinyAsciiStr::try_from_str("123").unwrap(),
    ];

    // Serialize to Arrow
    let arrow_array: ArrayRef = original.try_into_arrow().unwrap();

    // Verify the array type
    assert!(arrow_array.as_any().is::<FixedSizeBinaryArray>());
    let fixed_size_array = arrow_array.as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
    assert_eq!(fixed_size_array.value_length(), 3);

    // Deserialize back to Vec<TinyAsciiStr<3>>
    let roundtrip: Vec<TinyAsciiStr<3>> = arrow_array.try_into_collection().unwrap();

    // Verify the roundtrip
    assert_eq!(original, roundtrip);
}

#[cfg(feature = "tinystr")]
#[test]
fn test_tinyasciistr_max_length() {
    use arrow::array::{Array, ArrayRef, FixedSizeBinaryArray};
    use arrow_convert::deserialize::TryIntoCollection;
    use arrow_convert::serialize::TryIntoArrow;
    use tinystr::TinyAsciiStr;

    let original: Vec<TinyAsciiStr<10>> = vec![TinyAsciiStr::try_from_str("ABCDEFGHIJ").unwrap()];

    let arrow_array: ArrayRef = original.try_into_arrow().expect("Failed to convert to Arrow array");
    assert!(arrow_array.as_any().is::<FixedSizeBinaryArray>());
    let fixed_size_array = arrow_array.as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
    assert_eq!(fixed_size_array.value_length(), 10);

    let roundtrip: Vec<TinyAsciiStr<10>> = arrow_array
        .try_into_collection()
        .expect("Failed to convert from Arrow array");
    assert_eq!(original, roundtrip);
}
