use arrow_convert::field::LargeBinary;
use arrow_convert::{ArrowDeserialize, ArrowField, ArrowSerialize};

#[derive(Debug, ArrowField, ArrowSerialize, ArrowDeserialize)]
struct Test {
    #[arrow_field(type = "LargeBinary")]
    s: String,
}

fn main() {}
