/// Tests that the macro generated code doesn't assume the presence of additional bindings and uses absolute paths
use arrow_convert_derive::ArrowField;

#[derive(ArrowField)]
#[allow(dead_code)]
struct S {
    int_field: i64,
}
