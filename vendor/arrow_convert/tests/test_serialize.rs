use arrow::array::{Array, ArrayRef};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::record_batch::RecordBatch;
use arrow_convert::field::{ArrowField, FixedSizeBinary};
use arrow_convert::serialize::*;

#[test]
fn test_error_exceed_fixed_size_binary() {
    let strs = [b"abc".to_vec()];
    let r: arrow::error::Result<ArrayRef> = strs.try_into_arrow_as_type::<FixedSizeBinary<2>>();
    assert!(r.is_err())
}

#[test]
fn test_record_batch() {
    let strs = [b"abc".to_vec()];
    let r: RecordBatch = strs.try_into_arrow_as_type::<FixedSizeBinary<3>>().unwrap();
    assert_eq!(r.num_rows(), 1);
    assert_eq!(
        r.columns()[0].data_type(),
        &<FixedSizeBinary<3> as ArrowField>::data_type()
    );

    let r: RecordBatch = strs.try_into_arrow().unwrap();
    assert_eq!(r.num_rows(), 1);
    assert_eq!(
        r.columns()[0].data_type(),
        &<Vec<u8> as ArrowField>::data_type()
    );
}

#[test]
fn test_array() {
    let strs = [b"abc".to_vec()];
    let r: ArrayRef = strs.try_into_arrow_as_type::<FixedSizeBinary<3>>().unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r.data_type(), &<FixedSizeBinary<3> as ArrowField>::data_type());

    let r: ArrayRef = strs.try_into_arrow().unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r.data_type(), &<Vec<u8> as ArrowField>::data_type());
}

#[test]
fn test_buffer() {
    // Buffer, ScalarBuffer<u8> and Vec<u8> should serialize into BinaryArray
    let b: Vec<Buffer> = vec![(0..10).collect()];
    let rb: ArrayRef = b.try_into_arrow().unwrap();
    let dat: Vec<ScalarBuffer<u8>> = vec![(0..10).collect()];
    let r: ArrayRef = dat.try_into_arrow().unwrap();
    assert_eq!(rb.len(), 1);
    assert_eq!(r.len(), 1);
    assert_eq!(r.data_type(), &<Buffer as ArrowField>::data_type());
    assert_eq!(r.data_type(), &<ScalarBuffer<u8> as ArrowField>::data_type());
    assert_eq!(r.data_type(), &<Vec<u8> as ArrowField>::data_type());

    // ScalarBuffer<u16> and Vec<u16> should serialize into ListArray
    let dat: Vec<ScalarBuffer<u16>> = vec![(0..10).collect()];
    let r: ArrayRef = dat.try_into_arrow().unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r.data_type(), &<ScalarBuffer<u16> as ArrowField>::data_type());
    assert_eq!(r.data_type(), &<Vec<u16> as ArrowField>::data_type());
}

// #[test]
// fn test_field_serialize_error() {
//     pub struct CustomType(u64);

//     impl arrow_convert::field::ArrowField for CustomType {
//         type Type = Self;

//         #[inline]
//         fn data_type() -> arrow::datatypes::DataType {
//             arrow::datatypes::DataType::Extension(
//                 "custom".to_string(),
//                 Box::new(arrow::datatypes::DataType::UInt64),
//                 None,
//             )
//         }
//     }

//     impl arrow_convert::serialize::ArrowSerialize for CustomType {
//         type ArrayBuilderType = arrow::array::UInt64Builder;

//         #[inline]
//         fn new_array() -> Self::ArrayBuilderType {
//             Self::ArrayBuilderType::from(<Self as arrow_convert::field::ArrowField>::data_type())
//         }

//         #[inline]
//         fn arrow_serialize(_: &Self, _: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
//             Err(arrow::error::Error::NotYetImplemented("".to_owned()))
//         }
//     }

//     impl arrow_convert::deserialize::ArrowDeserialize for CustomType {
//         type ArrayType = arrow::array::PrimitiveArray<u64>;

//         #[inline]
//         fn arrow_deserialize(v: Option<&u64>) -> Option<Self> {
//             v.map(|t| CustomType(*t))
//         }
//     }

//     let arr = vec![CustomType(0)];
//     let r: arrow::error::Result<ArrayRef> = arr.try_into_arrow();
//     assert!(r.is_err())
// }
