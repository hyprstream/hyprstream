use arrow::array::*;
use arrow::datatypes::DataType;
use arrow_convert::deserialize::{arrow_array_deserialize_iterator, TryIntoCollection};
use arrow_convert::field::ArrowField;
use arrow_convert::serialize::TryIntoArrow;
/// Complex example that uses the following features:
///
/// - Deeply Nested structs and lists
/// - Custom types
use arrow_convert::{ArrowDeserialize, ArrowField, ArrowSerialize};
use chrono::DateTime;
use std::borrow::Borrow;

#[derive(Debug, Clone, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
pub struct Root {
    name: Option<String>,
    area: [u8; 6],
    age: u8,
    is_deleted: bool,
    a1: Option<f64>,
    a2: i64,
    // binary
    a3: Option<Vec<u8>>,
    // date32
    a4: chrono::NaiveDate,
    // timestamp(ns, None)
    a5: chrono::NaiveDateTime,
    // timestamp(ns, None)
    a6: Option<chrono::NaiveDateTime>,
    // array of date times
    date_time_list: Vec<chrono::NaiveDateTime>,
    // optional list array of optional strings
    nullable_list: Option<Vec<Option<String>>>,
    // optional list array of required strings
    required_list: Vec<Option<String>>,
    // custom type
    custom: CustomType,
    // custom optional type
    #[arrow_field(name = "cullable_nustom")]
    nullable_custom: Option<CustomType>,
    // vec custom type
    custom_list: Vec<CustomType>,
    // nested struct
    child: Child,
    // int 32 array
    int32_array: Vec<i32>,
    // large binary
    #[arrow_field(type = "arrow_convert::field::LargeBinary", name = "barge_linary")]
    large_binary: Vec<u8>,
    // fixed size binary
    #[arrow_field(type = "arrow_convert::field::FixedSizeBinary<3>")]
    fixed_size_binary: Vec<u8>,
    // large string
    #[arrow_field(type = "arrow_convert::field::LargeString")]
    large_string: String,
    // large vec
    #[arrow_field(type = "arrow_convert::field::LargeVec<i64>")]
    large_vec: Vec<i64>,
    // fixed size vec
    #[arrow_field(type = "arrow_convert::field::FixedSizeVec<i64, 3>")]
    fixed_size_vec: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, ArrowField, ArrowSerialize, ArrowDeserialize)]
pub struct Child {
    a1: i64,
    a2: String,
    // nested struct array
    child_array: Vec<ChildChild>,
}

#[derive(Debug, Clone, PartialEq, Eq, ArrowField, ArrowSerialize, ArrowDeserialize)]
pub struct ChildChild {
    a1: i32,
    bool_array: Vec<bool>,
    int64_array: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// A newtype around a u64
pub struct CustomType(u64);

/// To use with Arrow three traits need to be implemented:
/// - ArrowField
/// - ArrowSerialize
/// - ArrowDeserialize
impl arrow_convert::field::ArrowField for CustomType {
    type Type = Self;

    #[inline]
    fn data_type() -> arrow::datatypes::DataType {
        arrow::datatypes::DataType::UInt64
    }
}

impl arrow_convert::serialize::ArrowSerialize for CustomType {
    type ArrayBuilderType = arrow::array::UInt64Builder;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::default()
    }

    #[inline]
    fn arrow_serialize(v: &Self, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_option(Some(v.0));
        Ok(())
    }
}

impl arrow_convert::deserialize::ArrowDeserialize for CustomType {
    type ArrayType = arrow::array::UInt64Array;

    #[inline]
    fn arrow_deserialize(v: Option<u64>) -> Option<Self> {
        v.map(CustomType)
    }
}

// enable Vec<CustomType>
arrow_convert::arrow_enable_vec_for_type!(CustomType);

fn item1() -> Root {
    use chrono::NaiveDate;

    Root {
        name: Some("a".to_string()),
        area: "SYDNEY".as_bytes().try_into().unwrap(),
        age: 28,
        is_deleted: false,
        a1: Some(0.1),
        a2: 1,
        a3: Some(b"aa".to_vec()),
        a4: NaiveDate::from_ymd_opt(1970, 1, 2).unwrap(),
        a5: DateTime::from_timestamp(10000, 0).unwrap().naive_local(),
        a6: DateTime::from_timestamp(10001, 0).map(|dt| dt.naive_local()),
        date_time_list: vec![
            DateTime::from_timestamp(10000, 10).unwrap().naive_local(),
            DateTime::from_timestamp(10000, 11).unwrap().naive_local(),
        ],
        nullable_list: Some(vec![Some("cc".to_string()), Some("dd".to_string())]),
        required_list: vec![Some("aa".to_string()), Some("bb".to_string())],
        custom: CustomType(10),
        nullable_custom: Some(CustomType(11)),
        custom_list: vec![CustomType(12), CustomType(13)],
        child: Child {
            a1: 10,
            a2: "hello".to_string(),
            child_array: vec![
                ChildChild {
                    a1: 100,
                    bool_array: vec![false],
                    int64_array: vec![45555, 2124214, 224, 24214, 2424],
                },
                ChildChild {
                    a1: 101,
                    bool_array: vec![true, true, true],
                    int64_array: vec![4533, 22222, 2323, 333, 33322],
                },
            ],
        },
        int32_array: vec![0, 1, 3],
        large_binary: b"aa".to_vec(),
        fixed_size_binary: b"aaa".to_vec(),
        large_string: "abcdefg".to_string(),
        large_vec: vec![1, 2, 3, 4],
        fixed_size_vec: vec![10, 20, 30],
    }
}

fn item2() -> Root {
    use chrono::NaiveDate;

    Root {
        name: Some("b".to_string()),
        area: "SYDNEY".as_bytes().try_into().unwrap(),
        age: 28,
        is_deleted: true,
        a1: Some(0.1),
        a2: 1,
        a3: Some(b"aa".to_vec()),
        a4: NaiveDate::from_ymd_opt(1970, 1, 2).unwrap(),
        a5: DateTime::from_timestamp(10000, 0).unwrap().naive_local(),
        a6: None,
        date_time_list: vec![
            DateTime::from_timestamp(10000, 10).unwrap().naive_local(),
            DateTime::from_timestamp(10000, 11).unwrap().naive_local(),
        ],
        nullable_list: None,
        required_list: vec![Some("ee".to_string()), Some("ff".to_string())],
        custom: CustomType(11),
        nullable_custom: None,
        custom_list: vec![CustomType(14), CustomType(13)],
        child: Child {
            a1: 11,
            a2: "hello again".to_string(),
            child_array: vec![
                ChildChild {
                    a1: 100,
                    bool_array: vec![true, false, false, true],
                    int64_array: vec![111111, 2222, 33],
                },
                ChildChild {
                    a1: 102,
                    bool_array: vec![false],
                    int64_array: vec![45555, 2124214, 224, 24214, 2424],
                },
            ],
        },
        int32_array: vec![111, 1],
        large_binary: b"bb".to_vec(),
        fixed_size_binary: b"bbb".to_vec(),
        large_string: "abdefag".to_string(),
        large_vec: vec![5, 4, 3, 2],
        fixed_size_vec: vec![11, 21, 32],
    }
}

#[test]
fn test_round_trip() -> arrow::error::Result<()> {
    // serialize to an arrow array
    let original_array = [item1(), item2()];

    let array: ArrayRef = original_array.try_into_arrow()?;

    let struct_array = array.as_any().downcast_ref::<arrow::array::StructArray>().unwrap();
    assert_eq!(struct_array.len(), 2);

    let values = struct_array.columns();
    assert_eq!(values.len(), 23);
    assert_eq!(struct_array.len(), 2);

    let DataType::Struct(fields) = Root::data_type() else {
        panic!("Should be a struct")
    };

    assert_eq!(fields, Root::arrow_schema().fields);

    let names = struct_array.column_names();
    assert!(!names.iter().any(|x| *x == "nullable_custom"));
    assert!(names.iter().any(|x| *x == "cullable_nustom"));
    assert!(!names.iter().any(|x| *x == "large_binary"));
    assert!(names.iter().any(|x| *x == "barge_linary"));

    // can iterate one struct at a time without collecting
    for _i in arrow_array_deserialize_iterator::<Root>(array.borrow())? {
        // do something
    }

    // or can back to our original vector
    let foo_array: Vec<Root> = array.try_into_collection()?;
    assert_eq!(foo_array, original_array);
    Ok(())
}
