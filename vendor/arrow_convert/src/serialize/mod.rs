//! Implementation and traits for serializing to Arrow.

use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::ArrowNativeType;
use arrow::{array::*, datatypes};
use chrono::{NaiveDate, NaiveDateTime};
use std::sync::Arc;

mod push_null;
pub use push_null::*;

use crate::field::*;

/// Trait that is implemented by all types that are serializable to Arrow.
///
/// Implementations are provided for all built-in arrow types as well as Vec<T>, and Option<T>
/// if T implements ArrowSerialize.
///
/// Note that Vec<T> implementation needs to be enabled by the [`crate::arrow_enable_vec_for_type`] macro.
pub trait ArrowSerialize: ArrowField {
    /// The [`arrow::array::ArrayBuilder`] that holds this value
    type ArrayBuilderType: arrow::array::ArrayBuilder;

    /// Create a new mutable array
    fn new_array() -> Self::ArrayBuilderType;

    /// Serialize this field to arrow
    fn arrow_serialize(v: &<Self as ArrowField>::Type, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()>;
}

// Macro to facilitate implementation of serializable traits for numeric types and numeric mutable arrays.
macro_rules! impl_numeric_type {
    ($physical_type:ty, $primitive_type:ty) => {
        impl ArrowSerialize for $physical_type {
            type ArrayBuilderType = PrimitiveBuilder<$primitive_type>;

            #[inline]
            fn new_array() -> Self::ArrayBuilderType {
                Self::ArrayBuilderType::default()
            }

            #[inline]
            fn arrow_serialize(v: &Self, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
                array.append_option(Some(*v));
                Ok(())
            }
        }
    };
}

// blanket implementation for optional fields
impl<T> ArrowSerialize for Option<T>
where
    T: ArrowSerialize,
    T::ArrayBuilderType: PushNull,
{
    type ArrayBuilderType = <T as ArrowSerialize>::ArrayBuilderType;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        <T as ArrowSerialize>::new_array()
    }

    #[inline]
    fn arrow_serialize(v: &<Self as ArrowField>::Type, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        match v.as_ref() {
            Some(t) => <T as ArrowSerialize>::arrow_serialize(t, array),
            None => {
                array.push_null();
                Ok(())
            }
        }
    }
}

impl_numeric_type!(u8, datatypes::UInt8Type);
impl_numeric_type!(u16, datatypes::UInt16Type);
impl_numeric_type!(u32, datatypes::UInt32Type);
impl_numeric_type!(u64, datatypes::UInt64Type);
impl_numeric_type!(i8, datatypes::Int8Type);
impl_numeric_type!(i16, datatypes::Int16Type);
impl_numeric_type!(i32, datatypes::Int32Type);
impl_numeric_type!(i64, datatypes::Int64Type);
impl_numeric_type!(half::f16, datatypes::Float16Type);
impl_numeric_type!(f32, datatypes::Float32Type);
impl_numeric_type!(f64, datatypes::Float64Type);

impl<const PRECISION: u8, const SCALE: i8> ArrowSerialize for I128<PRECISION, SCALE> {
    type ArrayBuilderType = PrimitiveBuilder<datatypes::Decimal128Type>;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::default().with_data_type(<Self as ArrowField>::data_type())
    }

    #[inline]
    fn arrow_serialize(v: &i128, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_option(Some(*v));
        Ok(())
    }
}

impl<'a> ArrowSerialize for &'a str {
    type ArrayBuilderType = StringBuilder;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::default()
    }

    #[inline]
    fn arrow_serialize(v: &Self, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_option(Some(v));
        Ok(())
    }
}

impl ArrowSerialize for String {
    type ArrayBuilderType = StringBuilder;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::default()
    }

    #[inline]
    fn arrow_serialize(v: &Self, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_option(Some(v));
        Ok(())
    }
}

impl ArrowSerialize for LargeString {
    type ArrayBuilderType = LargeStringBuilder;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::default()
    }

    #[inline]
    fn arrow_serialize(v: &String, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_option(Some(v));
        Ok(())
    }
}

impl ArrowSerialize for bool {
    type ArrayBuilderType = BooleanBuilder;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::default()
    }

    #[inline]
    fn arrow_serialize(v: &Self, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_value(*v);
        Ok(())
    }
}

impl ArrowSerialize for NaiveDateTime {
    type ArrayBuilderType = TimestampNanosecondBuilder;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::default().with_data_type(<Self as ArrowField>::data_type())
    }

    #[inline]
    fn arrow_serialize(v: &Self, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_option(v.and_utc().timestamp_nanos_opt());
        Ok(())
    }
}

impl ArrowSerialize for NaiveDate {
    type ArrayBuilderType = Date32Builder;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::default().with_data_type(<Self as ArrowField>::data_type())
    }

    #[inline]
    fn arrow_serialize(v: &Self, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_option(Some(
            chrono::Datelike::num_days_from_ce(v) - arrow::temporal_conversions::UNIX_EPOCH_DAY as i32,
        ));
        Ok(())
    }
}

// Treat both Buffer and ScalarBuffer<u8> the same
impl ArrowSerialize for Buffer {
    type ArrayBuilderType = BinaryBuilder;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::default()
    }

    #[inline]
    fn arrow_serialize(v: &Self, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_option(Some(v.as_slice()));
        Ok(())
    }
}
impl ArrowSerialize for ScalarBuffer<u8> {
    type ArrayBuilderType = BinaryBuilder;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::default()
    }

    #[inline]
    fn arrow_serialize(v: &Self, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_option(Some(v));
        Ok(())
    }
}

impl ArrowSerialize for Vec<u8> {
    type ArrayBuilderType = BinaryBuilder;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::default()
    }

    #[inline]
    fn arrow_serialize(v: &Self, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_option(Some(v));
        Ok(())
    }
}

impl ArrowSerialize for LargeBinary {
    type ArrayBuilderType = LargeBinaryBuilder;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::default()
    }

    #[inline]
    fn arrow_serialize(v: &Vec<u8>, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_option(Some(v));
        Ok(())
    }
}

impl<const SIZE: i32> ArrowSerialize for FixedSizeBinary<SIZE> {
    type ArrayBuilderType = FixedSizeBinaryBuilder;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::new(SIZE)
    }

    #[inline]
    fn arrow_serialize(v: &Vec<u8>, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_value(v)
    }
}

impl<const SIZE: usize> ArrowSerialize for [u8; SIZE] {
    type ArrayBuilderType = FixedSizeBinaryBuilder;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::new(SIZE as i32)
    }

    #[inline]
    fn arrow_serialize(v: &Self, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_value(v)
    }
}

// Blanket implementation for Buffer
impl<T> ArrowSerialize for ScalarBuffer<T>
where
    T: ArrowNativeType + ArrowSerialize + ArrowEnableVecForType + ArrowField<Type = T>,
{
    type ArrayBuilderType = ListBuilder<<T as ArrowSerialize>::ArrayBuilderType>;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        let field = Arc::new(<T as ArrowField>::field(DEFAULT_FIELD_NAME));
        ListBuilder::new(<T as ArrowSerialize>::new_array()).with_field(field)
    }

    #[inline]
    fn arrow_serialize(v: &<Self as ArrowField>::Type, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        let values = array.values();
        for i in v.iter() {
            <T as ArrowSerialize>::arrow_serialize(i, values)?;
        }
        array.append(true);
        Ok(())
    }
}

// Blanket implementation for Vec
impl<T> ArrowSerialize for Vec<T>
where
    T: ArrowSerialize + ArrowEnableVecForType + 'static,
    <T as ArrowSerialize>::ArrayBuilderType: Default,
{
    type ArrayBuilderType = ListBuilder<<T as ArrowSerialize>::ArrayBuilderType>;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        let field = Arc::new(<T as ArrowField>::field(DEFAULT_FIELD_NAME));
        ListBuilder::new(<T as ArrowSerialize>::new_array()).with_field(field)
    }

    fn arrow_serialize(v: &<Self as ArrowField>::Type, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        let values = array.values();
        for i in v.iter() {
            <T as ArrowSerialize>::arrow_serialize(i, values)?;
        }
        array.append(true);
        Ok(())
    }
}

impl<T> ArrowSerialize for LargeVec<T>
where
    T: ArrowSerialize + ArrowEnableVecForType + 'static,
    <T as ArrowSerialize>::ArrayBuilderType: Default,
{
    type ArrayBuilderType = LargeListBuilder<<T as ArrowSerialize>::ArrayBuilderType>;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        let field = Arc::new(<T as ArrowField>::field(DEFAULT_FIELD_NAME));
        Self::ArrayBuilderType::new(<T as ArrowSerialize>::new_array()).with_field(field)
    }

    fn arrow_serialize(v: &<Self as ArrowField>::Type, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        let values = array.values();
        for i in v.iter() {
            <T as ArrowSerialize>::arrow_serialize(i, values)?;
        }
        array.append(true);
        Ok(())
    }
}

impl<T, const SIZE: i32> ArrowSerialize for FixedSizeVec<T, SIZE>
where
    T: ArrowSerialize + ArrowEnableVecForType + 'static,
    <T as ArrowSerialize>::ArrayBuilderType: Default,
{
    type ArrayBuilderType = FixedSizeListBuilder<<T as ArrowSerialize>::ArrayBuilderType>;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::new(<T as ArrowSerialize>::new_array(), SIZE)
            .with_field(<T as ArrowField>::field(DEFAULT_FIELD_NAME))
    }

    fn arrow_serialize(v: &<Self as ArrowField>::Type, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        let values = array.values();
        for i in v.iter() {
            <T as ArrowSerialize>::arrow_serialize(i, values)?;
        }
        array.append(true);
        Ok(())
    }
}

impl<T, const SIZE: usize> ArrowSerialize for [T; SIZE]
where
    T: ArrowSerialize + ArrowEnableVecForType + 'static,
    <T as ArrowSerialize>::ArrayBuilderType: Default,
{
    type ArrayBuilderType = FixedSizeListBuilder<<T as ArrowSerialize>::ArrayBuilderType>;

    #[inline]
    fn new_array() -> Self::ArrayBuilderType {
        Self::ArrayBuilderType::new(<T as ArrowSerialize>::new_array(), SIZE as i32)
            .with_field(<T as ArrowField>::field(DEFAULT_FIELD_NAME))
    }

    fn arrow_serialize(v: &<Self as ArrowField>::Type, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        let values = array.values();
        for i in v.iter() {
            <T as ArrowSerialize>::arrow_serialize(i, values)?;
        }
        array.append(true);
        Ok(())
    }
}

// internal helper method to extend a mutable array
fn arrow_serialize_extend_internal<
    'a,
    A: 'static,
    T: ArrowSerialize + ArrowField<Type = A> + 'static,
    I: IntoIterator<Item = &'a A>,
>(
    into_iter: I,
    array: &mut <T as ArrowSerialize>::ArrayBuilderType,
) -> arrow::error::Result<()> {
    let iter = into_iter.into_iter();
    for i in iter {
        <T as ArrowSerialize>::arrow_serialize(i, array)?;
    }
    Ok(())
}

/// Serializes an iterator into an `arrow::ArrayBuilder`
pub fn arrow_serialize_to_mutable_array<
    'a,
    A: 'static,
    T: ArrowSerialize + ArrowField<Type = A> + 'static,
    I: IntoIterator<Item = &'a A>,
>(
    into_iter: I,
) -> arrow::error::Result<<T as ArrowSerialize>::ArrayBuilderType> {
    let mut arr = <T as ArrowSerialize>::new_array();
    arrow_serialize_extend_internal::<A, T, I>(into_iter, &mut arr)?;
    Ok(arr)
}

/// API to flatten a RecordBatch consisting of an `arrow::array::StructArray` into a `RecordBatch` consisting of `arrow::array::Array`s contained by the `StructArray`
pub trait FlattenRecordBatch {
    /// Convert an `arrow::record_batch::RecordBatch` containing a `arrow::array::StructArray` to an `arrow::record_batch::RecordBatch` consisting of the
    /// `arrow::array::Array`s contained by the `StructArray` by consuming the
    /// original `RecordBatch`. Returns an error if the `RecordBatch` cannot be flattened.
    fn flatten(self) -> Result<RecordBatch, arrow::error::ArrowError>;
}

impl FlattenRecordBatch for RecordBatch {
    fn flatten(self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let arrays = self.columns();

        // we only support flattening of a RecordBatch containing a single StructArray
        if arrays.len() != 1 {
            return Err(arrow::error::ArrowError::InvalidArgumentError(
                "RecordBatch must contain a single Array".to_string(),
            ));
        }

        let array = &arrays[0];

        let data_type = array.as_ref().data_type();
        if !matches!(data_type, arrow::datatypes::DataType::Struct(_)) {
            return Err(arrow::error::ArrowError::InvalidArgumentError(
                "Array in RecordBatch must be of type arrow::datatypes::PhysicalType::Struct".to_string(),
            ));
        }

        let struct_array = array.as_ref().as_any().downcast_ref::<StructArray>().unwrap();
        Ok(RecordBatch::from(struct_array))
    }
}

/// Top-level API to serialize to Arrow
pub trait TryIntoArrow<'a, ArrowArray, Element>
where
    Self: IntoIterator<Item = &'a Element>,
    Element: 'static,
{
    /// Convert from any iterable collection into an `arrow::Array`
    fn try_into_arrow(self) -> arrow::error::Result<ArrowArray>
    where
        Element: ArrowSerialize + ArrowField<Type = Element> + 'static;

    /// Convert from any iterable collection into an `arrow::Array` by coercing the conversion to a specific Arrow type.
    /// This is useful when the same rust type maps to one or more Arrow types for example `LargeString`.
    fn try_into_arrow_as_type<ArrowType>(self) -> arrow::error::Result<ArrowArray>
    where
        ArrowType: ArrowSerialize + ArrowField<Type = Element> + 'static;
}

impl<'a, Element, Collection> TryIntoArrow<'a, ArrayRef, Element> for Collection
where
    Element: 'static,
    Collection: IntoIterator<Item = &'a Element>,
{
    fn try_into_arrow(self) -> arrow::error::Result<ArrayRef>
    where
        Element: ArrowSerialize + ArrowField<Type = Element> + 'static,
    {
        Ok(arrow_serialize_to_mutable_array::<Element, Element, Collection>(self)?.finish())
    }

    fn try_into_arrow_as_type<Field>(self) -> arrow::error::Result<ArrayRef>
    where
        Field: ArrowSerialize + ArrowField<Type = Element> + 'static,
    {
        Ok(arrow_serialize_to_mutable_array::<Element, Field, Collection>(self)?.finish())
    }
}

impl<'a, Element, Collection> TryIntoArrow<'a, RecordBatch, Element> for Collection
where
    Element: 'static,
    Collection: IntoIterator<Item = &'a Element>,
{
    fn try_into_arrow(self) -> arrow::error::Result<RecordBatch>
    where
        Element: ArrowSerialize + ArrowField<Type = Element> + 'static,
    {
        RecordBatch::try_from_iter([(
            "record_batch_item",
            arrow_serialize_to_mutable_array::<Element, Element, Collection>(self)?.finish(),
        )])
    }

    fn try_into_arrow_as_type<Field>(self) -> arrow::error::Result<RecordBatch>
    where
        Field: ArrowSerialize + ArrowField<Type = Element> + 'static,
    {
        RecordBatch::try_from_iter([(
            "record_batch_item",
            arrow_serialize_to_mutable_array::<Element, Field, Collection>(self)?.finish(),
        )])
    }
}
