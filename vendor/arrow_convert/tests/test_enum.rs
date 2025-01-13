use arrow::{array::*, datatypes::*};
use arrow_convert::{
    deserialize::TryIntoCollection, serialize::TryIntoArrow, ArrowDeserialize, ArrowField, ArrowSerialize,
};
use pretty_assertions::assert_eq;

#[test]
fn test_dense_enum_unit_variant() {
    #[derive(Debug, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    #[arrow_field(type = "dense")]
    enum TestEnum {
        VAL1,
        VAL2,
        VAL3,
        VAL4,
    }

    let enums = vec![TestEnum::VAL1, TestEnum::VAL2, TestEnum::VAL3, TestEnum::VAL4];
    let b: ArrayRef = enums.try_into_arrow().unwrap();
    assert_eq!(
        b.data_type(),
        &DataType::Union(
            UnionFields::new(
                vec![0, 1, 2, 3],
                vec![
                    Field::new("VAL1", DataType::Boolean, false),
                    Field::new("VAL2", DataType::Boolean, false),
                    Field::new("VAL3", DataType::Boolean, false),
                    Field::new("VAL4", DataType::Boolean, false),
                ]
            ),
            UnionMode::Dense
        )
    );
    let round_trip: Vec<TestEnum> = b.try_into_collection().unwrap();
    assert_eq!(round_trip, enums);
}

#[test]
fn test_sparse_enum_unit_variant() {
    #[derive(Debug, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    #[arrow_field(type = "sparse")]
    enum TestEnum {
        VAL1,
        VAL2,
        VAL3,
        VAL4,
    }

    let enums = vec![TestEnum::VAL1, TestEnum::VAL2, TestEnum::VAL3, TestEnum::VAL4];
    let b: ArrayRef = enums.try_into_arrow().unwrap();
    assert_eq!(
        b.data_type(),
        &DataType::Union(
            UnionFields::new(
                vec![0, 1, 2, 3],
                vec![
                    Field::new("VAL1", DataType::Boolean, false),
                    Field::new("VAL2", DataType::Boolean, false),
                    Field::new("VAL3", DataType::Boolean, false),
                    Field::new("VAL4", DataType::Boolean, false),
                ]
            ),
            UnionMode::Sparse
        )
    );
    let round_trip: Vec<TestEnum> = b.try_into_collection().unwrap();
    assert_eq!(round_trip, enums);
}

#[test]
fn test_nested_unit_variant() {
    #[derive(Debug, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    struct TestStruct {
        a1: i64,
    }

    #[derive(Debug, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    #[arrow_field(type = "dense")]
    enum TestDenseEnum {
        VAL1,
        VAL2(i32),
        VAL3(f64),
        VAL4(TestStruct),
        VAL5(DenseChildEnum),
        VAL6(SparseChildEnum),
    }

    #[derive(Debug, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    #[arrow_field(type = "sparse")]
    enum TestSparseEnum {
        VAL1,
        VAL2(i32),
        VAL3(f64),
        VAL4(TestStruct),
        VAL5(DenseChildEnum),
        VAL6(SparseChildEnum),
    }

    #[derive(Debug, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    #[arrow_field(type = "dense")]
    enum DenseChildEnum {
        VAL1,
        VAL2(i32),
        VAL3(f64),
        VAL4(TestStruct),
    }

    #[derive(Debug, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    #[arrow_field(type = "sparse")]
    enum SparseChildEnum {
        VAL1,
        VAL2(i32),
        VAL3(f64),
        VAL4(TestStruct),
    }

    let enums = vec![
        TestDenseEnum::VAL1,
        TestDenseEnum::VAL2(2),
        TestDenseEnum::VAL3(1.2),
        TestDenseEnum::VAL4(TestStruct { a1: 10 }),
        TestDenseEnum::VAL5(DenseChildEnum::VAL4(TestStruct { a1: 17 })),
        TestDenseEnum::VAL6(SparseChildEnum::VAL4(TestStruct { a1: 42 })),
    ];

    let b: ArrayRef = enums.try_into_arrow().unwrap();
    let round_trip: Vec<TestDenseEnum> = b.try_into_collection().unwrap();
    assert_eq!(round_trip, enums);

    let enums = vec![
        TestSparseEnum::VAL1,
        TestSparseEnum::VAL2(2),
        TestSparseEnum::VAL3(1.2),
        TestSparseEnum::VAL4(TestStruct { a1: 10 }),
        TestSparseEnum::VAL5(DenseChildEnum::VAL4(TestStruct { a1: 17 })),
        TestSparseEnum::VAL6(SparseChildEnum::VAL4(TestStruct { a1: 42 })),
    ];

    let b: ArrayRef = enums.try_into_arrow().unwrap();
    let round_trip: Vec<TestSparseEnum> = b.try_into_collection().unwrap();
    assert_eq!(round_trip, enums);
}

// TODO: reenable this test once slices for enums is fixed.
#[test]
#[allow(unused)]
fn test_slice() {
    #[derive(Debug, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    struct TestStruct {
        a1: i64,
    }

    #[derive(Debug, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    #[arrow_field(type = "dense")]
    enum TestEnum {
        VAL1,
        VAL2(i32),
        VAL3(f64),
        VAL4(TestStruct),
        VAL5(ChildEnum),
    }

    #[derive(Debug, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    #[arrow_field(type = "sparse")]
    enum ChildEnum {
        VAL1,
        VAL2(i32),
        VAL3(f64),
        VAL4(TestStruct),
        VAL5([i32; 2]),
    }

    let enums = vec![
        TestEnum::VAL4(TestStruct { a1: 11 }),
        TestEnum::VAL1,
        TestEnum::VAL2(2),
        TestEnum::VAL3(1.2),
        TestEnum::VAL4(TestStruct { a1: 10 }),
    ];

    let b: ArrayRef = enums.try_into_arrow().unwrap();

    for i in 0..enums.len() {
        let arrow_slice = b.slice(i, enums.len() - i);
        let original_slice = &enums[i..enums.len()];
        let round_trip: Vec<TestEnum> = arrow_slice.try_into_collection().unwrap();
        assert_eq!(round_trip, original_slice);
    }
}
