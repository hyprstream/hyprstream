use crate::arrow_enable_vec_for_type;
use crate::deserialize::ArrowDeserialize;
use crate::field::ArrowField;
use crate::serialize::ArrowSerialize;

use arrow::datatypes::{DataType, DECIMAL128_MAX_PRECISION, DECIMAL_DEFAULT_SCALE};
use rust_decimal::Decimal;

use arrow::array::{Decimal128Array, Decimal128Builder};

impl ArrowField for Decimal {
    type Type = Decimal;

    #[inline]
    fn data_type() -> DataType {
        DataType::Decimal128(DECIMAL128_MAX_PRECISION, DECIMAL_DEFAULT_SCALE)
    }
}

arrow_enable_vec_for_type!(Decimal);

impl ArrowSerialize for Decimal {
    type ArrayBuilderType = Decimal128Builder;

    fn new_array() -> Self::ArrayBuilderType {
        Decimal128Builder::new().with_data_type(Self::data_type())
    }

    fn arrow_serialize(v: &Self::Type, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
        array.append_value(decimal_to_scaled_i128(*v));
        Ok(())
    }
}

impl ArrowDeserialize for Decimal {
    type ArrayType = Decimal128Array;

    fn arrow_deserialize(v: Option<i128>) -> Option<Decimal> {
        v.map(|d| Decimal::from_i128_with_scale(d, DECIMAL_DEFAULT_SCALE as _))
    }
}

/// Converts a `Decimal` value to an `i128` representation, adjusting the scale to match the default scale.
fn decimal_to_scaled_i128(decimal: Decimal) -> i128 {
    let m = decimal.mantissa();
    let scale_diff = DECIMAL_DEFAULT_SCALE as i32 - decimal.scale() as i32;

    if scale_diff == 0 {
        m
    } else if scale_diff < 0 {
        m / 10_i128.pow(scale_diff.unsigned_abs())
    } else {
        m * 10_i128.pow(scale_diff as u32)
    }
}
