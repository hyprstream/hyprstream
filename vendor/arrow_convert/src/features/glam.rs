use arrow::datatypes::DataType;

use crate::arrow_enable_vec_for_type;
use crate::deserialize::ArrowDeserialize;
use crate::field::ArrowField;
use crate::serialize::ArrowSerialize;
use arrow::datatypes::Field;

use crate::deserialize::arrow_deserialize_vec_helper;
use arrow::array::ArrayRef;
use arrow::array::{BooleanBuilder, Float32Builder, Float64Builder};
use arrow::array::{FixedSizeListArray, FixedSizeListBuilder};
use std::sync::Arc;

/// This macro implements the `ArrowSerialize` and `ArrowDeserialize` traits for a given `glam` vector or matrix type.
///
/// The macro takes the following parameters:
/// - `$type`: The type of the `glam` vector or matrix to implement the traits for.
/// - `$size`: The size of the vector or matrix (e.g. 2 for `glam::Vec2`, 4 for `glam::Mat4`).
/// - `$dt`: The data type of the elements in the vector or matrix (e.g. `bool`, `f32`).
/// - `$arrow_dt`: The corresponding Arrow data type for the element type.
/// - `$array_builder`: The Arrow array builder type to use for the element type.
/// - `$se`: A closure that serializes the `$type` to a slice of the element type.
/// - `$de`: A closure that deserializes a `Vec` of the element type to the `$type`.
macro_rules! impl_glam_ty {
    ($type:ty, $size:expr, $dt:ident, $arrow_dt:expr, $array_builder:ident, $se:expr, $de:expr) => {
        impl ArrowField for $type {
            type Type = Self;

            fn data_type() -> DataType {
                let field = Field::new("scalar", $arrow_dt, false);
                DataType::FixedSizeList(Arc::new(field), $size)
            }
        }

        arrow_enable_vec_for_type!($type);

        impl ArrowSerialize for $type {
            type ArrayBuilderType = FixedSizeListBuilder<$array_builder>;

            fn new_array() -> Self::ArrayBuilderType {
                let field = Field::new("scalar", $arrow_dt, false);
                Self::ArrayBuilderType::new(<$dt as ArrowSerialize>::new_array(), $size).with_field(field)
            }

            fn arrow_serialize(v: &Self::Type, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
                let v = $se(v);

                array.values().append_slice(v.as_ref());
                array.append(true);
                Ok(())
            }
        }

        impl ArrowDeserialize for $type {
            type ArrayType = FixedSizeListArray;

            fn arrow_deserialize(v: Option<ArrayRef>) -> Option<Self> {
                let v = arrow_deserialize_vec_helper::<$dt>(v)?;
                Some($de(v))
            }
        }
    };
}

/// Implements the `ArrowSerialize` and `ArrowDeserialize` traits for the given `glam::Vec<bool>` type.
macro_rules! impl_glam_vec_bool {
    ($type:ty, $size:expr) => {
        impl_glam_ty!(
            $type,
            $size,
            bool,
            DataType::Boolean,
            BooleanBuilder,
            |v: &$type| <[bool; $size]>::from(*v),
            |v: Vec<bool>| {
                let length = v.len();

                match <[bool; $size]>::try_from(v).ok() {
                    None => panic!(
                        "Expected size of {} deserializing array of type `{}`, got {}",
                        std::any::type_name::<$type>(),
                        $size,
                        length
                    ),
                    Some(array) => Self::from_array(array),
                }
            }
        );
    };
}

/// Implements the `ArrowSerialize` and `ArrowDeserialize` traits for the given `glam::Vec2` type.
macro_rules! impl_glam_vec_f32 {
    ($type:ty, $size:expr) => {
        impl_glam_ty!(
            $type,
            $size,
            f32,
            DataType::Float32,
            Float32Builder,
            |v: &$type| *v,
            |v: Vec<f32>| Self::from_slice(&v)
        );
    };
}

/// Implements the `ArrowSerialize` and `ArrowDeserialize` traits for the given `glam::Mat2`, `glam::Mat3`, and `glam::Mat4` types.
macro_rules! impl_glam_mat_f32 {
    ($type:ty, $size:expr) => {
        impl_glam_ty!(
            $type,
            $size,
            f32,
            DataType::Float32,
            Float32Builder,
            |v: &$type| *v,
            |v: Vec<f32>| Self::from_cols_slice(&v)
        );
    };
}

/// Implements the `ArrowSerialize` and `ArrowDeserialize` traits for the given `glam::DVec2`, `glam::DVec3`, and `glam::DVec4` types.
macro_rules! impl_glam_vec_f64 {
    ($type:ty, $size:expr) => {
        impl_glam_ty!(
            $type,
            $size,
            f64,
            DataType::Float64,
            Float64Builder,
            |v: &$type| *v,
            |v: Vec<f64>| Self::from_slice(&v)
        );
    };
}

/// Implements the `ArrowSerialize` and `ArrowDeserialize` traits for the given `glam::DMat2`, `glam::DMat3`, and `glam::DMat4` types.
macro_rules! impl_glam_mat_f64 {
    ($type:ty, $size:expr) => {
        impl_glam_ty!(
            $type,
            $size,
            f64,
            DataType::Float64,
            Float64Builder,
            |v: &$type| *v,
            |v: Vec<f64>| Self::from_cols_slice(&v)
        );
    };
}

// Boolean vectors
impl_glam_vec_bool!(glam::BVec2, 2);
impl_glam_vec_bool!(glam::BVec3, 3);
impl_glam_vec_bool!(glam::BVec4, 4);

// Float32 vectors and matrices
impl_glam_vec_f32!(glam::Vec2, 2);
impl_glam_vec_f32!(glam::Vec3, 3);
impl_glam_vec_f32!(glam::Vec4, 4);
impl_glam_mat_f32!(glam::Mat2, 4);
impl_glam_mat_f32!(glam::Mat3, 9);
impl_glam_mat_f32!(glam::Mat4, 16);

// Float64 vectors and matrices
impl_glam_vec_f64!(glam::DVec2, 2);
impl_glam_vec_f64!(glam::DVec3, 3);
impl_glam_vec_f64!(glam::DVec4, 4);
impl_glam_mat_f64!(glam::DMat2, 4);
impl_glam_mat_f64!(glam::DMat3, 9);
impl_glam_mat_f64!(glam::DMat4, 16);
