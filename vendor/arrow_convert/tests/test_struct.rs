use arrow::array::*;
use arrow_convert::{deserialize::*, serialize::*, ArrowDeserialize, ArrowField, ArrowSerialize};

#[test]
fn test_nested_optional_struct_array() {
    #[derive(Debug, Clone, ArrowField, ArrowSerialize, ArrowDeserialize, PartialEq)]
    struct Top {
        child_array: Vec<Option<Child>>,
    }
    #[derive(Debug, Clone, ArrowField, ArrowSerialize, ArrowDeserialize, PartialEq)]
    struct Child {
        a1: i64,
    }

    let original_array = vec![
        Top {
            child_array: vec![
                Some(Child { a1: 10 }),
                None,
                Some(Child { a1: 12 }),
                Some(Child { a1: 14 }),
            ],
        },
        Top {
            child_array: vec![None, None, None, None],
        },
        Top {
            child_array: vec![None, None, Some(Child { a1: 12 }), None],
        },
    ];

    let b: ArrayRef = original_array.try_into_arrow().unwrap();
    let round_trip: Vec<Top> = b.try_into_collection().unwrap();
    assert_eq!(original_array, round_trip);
}

#[test]
fn test_slice() {
    #[derive(Debug, Clone, ArrowField, ArrowSerialize, ArrowDeserialize, PartialEq)]
    struct T {
        a1: i64,
    }

    let original = vec![T { a1: 1 }, T { a1: 2 }, T { a1: 3 }, T { a1: 4 }];

    let b: ArrayRef = original.try_into_arrow().unwrap();

    for i in 0..original.len() {
        let arrow_slice = b.slice(i, original.len() - i);
        let original_slice = &original[i..original.len()];
        let round_trip: Vec<T> = arrow_slice.try_into_collection().unwrap();
        assert_eq!(round_trip, original_slice);
    }
}

#[test]
fn test_nested_slice() {
    #[derive(Debug, Clone, ArrowField, ArrowSerialize, ArrowDeserialize, PartialEq)]
    struct Top {
        child_array: Vec<Option<Child>>,
    }
    #[derive(Debug, Clone, ArrowField, ArrowSerialize, ArrowDeserialize, PartialEq)]
    struct Child {
        a1: i64,
    }

    let original = vec![
        Top {
            child_array: vec![
                Some(Child { a1: 10 }),
                None,
                Some(Child { a1: 12 }),
                Some(Child { a1: 14 }),
            ],
        },
        Top {
            child_array: vec![None, None, None, None],
        },
        Top {
            child_array: vec![None, None, Some(Child { a1: 12 }), None],
        },
    ];

    let b: ArrayRef = original.try_into_arrow().unwrap();

    for i in 0..original.len() {
        let arrow_slice = b.slice(i, original.len() - i);
        let original_slice = &original[i..original.len()];
        let round_trip: Vec<Top> = arrow_slice.try_into_collection().unwrap();
        assert_eq!(round_trip, original_slice);
    }
}
