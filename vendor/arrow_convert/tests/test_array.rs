use arrow::array::Array;
use arrow::array::ArrayRef;
use arrow_convert::deserialize::TryIntoCollection;
use arrow_convert::serialize::TryIntoArrow;
/// Simple example
use arrow_convert::{ArrowDeserialize, ArrowField, ArrowSerialize};

#[derive(Debug, PartialEq, Clone, Copy, ArrowField, ArrowSerialize, ArrowDeserialize)]
pub struct QuadPoints {
    points: [Point; 4],
}

#[derive(Clone, Copy, ArrowField, ArrowSerialize, ArrowDeserialize)]
pub struct AABB {
    min: Point,
    max: Point,
}

#[derive(Debug, PartialEq, Clone, Copy, ArrowField, ArrowSerialize, ArrowDeserialize)]
pub struct Point {
    x: f64,
    y: f64,
}

#[test]
fn test_simple_roundtrip() {
    let original_array = vec![
        QuadPoints {
            points: [
                Point { x: 0.0, y: 0.0 },
                Point { x: 1.0, y: 0.0 },
                Point { x: 1.0, y: 1.0 },
                Point { x: 0.0, y: 1.0 },
            ],
        },
        QuadPoints {
            points: [
                Point { x: 0.0, y: 0.0 },
                Point { x: 2.0, y: 0.0 },
                Point { x: 2.0, y: 2.0 },
                Point { x: 0.0, y: 2.0 },
            ],
        },
        QuadPoints {
            points: [
                Point { x: 0.0, y: 0.0 },
                Point { x: 3.0, y: 0.0 },
                Point { x: 3.0, y: 3.0 },
                Point { x: 0.0, y: 3.0 },
            ],
        },
    ];

    // serialize to an arrow array. try_into_arrow() is enabled by the TryIntoArrow trait
    let arrow_array: ArrayRef = original_array.try_into_arrow().unwrap();

    // which can be cast to an Arrow StructArray and be used for all kinds of IPC, FFI, etc.
    // supported by `arrow`
    let struct_array = arrow_array
        .as_any()
        .downcast_ref::<arrow::array::StructArray>()
        .unwrap();
    assert_eq!(struct_array.len(), 3);

    // deserialize back to our original vector via TryIntoCollection trait.
    let round_trip_array: Vec<QuadPoints> = arrow_array.try_into_collection().unwrap();
    assert_eq!(round_trip_array, original_array);
}
