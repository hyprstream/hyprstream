use std::sync::Arc;

/// Simple example
use arrow::array::Array;
use arrow_convert::{
    deserialize::TryIntoCollection, serialize::TryIntoArrow, ArrowDeserialize, ArrowField, ArrowSerialize,
};

#[derive(Debug, Clone, PartialEq, Eq, ArrowField, ArrowSerialize, ArrowDeserialize)]
pub struct Foo {
    name: String,
}

#[test]
fn test_simple_roundtrip() {
    let original_array = [
        Foo {
            name: "hello".to_string(),
        },
        Foo {
            name: "one more".to_string(),
        },
        Foo {
            name: "good bye".to_string(),
        },
    ];

    // serialize to an arrow array. try_into_arrow() is enabled by the TryIntoArrow trait
    let arrow_array: Arc<dyn Array> = original_array.try_into_arrow().unwrap();

    // which can be cast to an Arrow StructArray and be used for all kinds of IPC, FFI, etc.
    // supported by `arrow`
    let struct_array = arrow_array
        .as_any()
        .downcast_ref::<arrow::array::StructArray>()
        .unwrap();
    assert_eq!(struct_array.len(), 3);

    // deserialize back to our original vector via TryIntoCollection trait.
    let round_trip_array: Vec<Foo> = arrow_array.try_into_collection().unwrap();
    assert_eq!(round_trip_array, original_array);
}
