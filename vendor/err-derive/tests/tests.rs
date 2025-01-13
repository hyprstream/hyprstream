#![cfg(feature = "std")]
use std::error::Error;

#[macro_use]
extern crate err_derive;

#[derive(Debug, Error)]
#[error(display = "An error has occurred.")]
struct UnitError;

#[test]
fn unit_struct() {
    let s = format!("{}", UnitError);
    assert_eq!(&s[..], "An error has occurred.");
}

#[derive(Debug, Error)]
#[error(display = "Error code: {}", code)]
struct RecordError {
    code: u32,
}

#[test]
fn record_struct() {
    let s = format!("{}", RecordError { code: 0 });
    assert_eq!(&s[..], "Error code: 0");
}

#[derive(Debug, Error)]
#[error(display = "Error code: {}", _0)]
struct TupleError(i32);

#[test]
fn tuple_struct() {
    let s = format!("{}", TupleError(2));
    assert_eq!(&s[..], "Error code: 2");
}

#[derive(Debug, Error)]
enum EnumError {
    #[error(display = "Error code: {}", code)]
    StructVariant { code: i32 },
    #[error(display = "Error: {}", _0)]
    TupleVariant(&'static str),
    #[error(display = "An error has occurred.")]
    UnitVariant,
}

#[test]
fn enum_error() {
    let s = format!("{}", EnumError::StructVariant { code: 2 });
    assert_eq!(&s[..], "Error code: 2");
    let s = format!("{}", EnumError::TupleVariant("foobar"));
    assert_eq!(&s[..], "Error: foobar");
    let s = format!("{}", EnumError::UnitVariant);
    assert_eq!(&s[..], "An error has occurred.");
}

#[derive(Debug, Error)]
enum ErrorWithSource {
    #[error(display = "tuple error")]
    A(#[error(source)] TupleError),
}

#[test]
fn error_with_source() {
    let e = ErrorWithSource::A(TupleError(1));

    assert!(e.source().is_some());
}

#[derive(Debug, Error)]
enum EnumFrom {
    #[error(display = "Tuple error: {}", _0)]
    TupleVariant(#[error(source)] UnitError),
    #[error(display = "Struct error: {}", inner)]
    StructVariant {
        #[error(from)]
        inner: EnumError,
    },
}

#[allow(dead_code)]
#[derive(Debug, Error)]
enum EnumNoFromVariant {
    #[error(display = "First error: {}", _0)]
    First(#[error(source)] UnitError),
    #[error(display = "Second error: {}", _0)]
    Second(#[error(source, no_from)] UnitError),
}

#[allow(dead_code)]
#[derive(Debug, Error)]
#[error(no_from)]
enum EnumNoFromStruct {
    #[error(display = "First error: {}", _0)]
    First(#[error(source)] UnitError),
    #[error(display = "Second error: {}", _0)]
    Second(#[error(source, from)] UnitError),
}

#[test]
fn from_inner() {
    let inner = EnumError::StructVariant { code: 12 };
    let outer = EnumFrom::from(inner);

    let s = format!("{}", outer);
    assert_eq!(&s[..], "Struct error: Error code: 12");

    let s = format!("{}", EnumNoFromVariant::from(UnitError));
    assert_eq!(&s[..], "First error: An error has occurred.");

    let s = format!("{}", EnumNoFromStruct::from(UnitError));
    assert_eq!(&s[..], "Second error: An error has occurred.");
}

#[allow(dead_code)]
#[derive(Debug, Error)]
enum EnumAutoBox {
    #[error(display = "Boxed error: {}", _0)]
    First(#[error(source)] Box<UnitError>),
}

#[test]
fn from_enum_autobox() {
    let s = format!("{}", EnumAutoBox::from(UnitError));
    assert_eq!(&s[..], "Boxed error: An error has occurred.");

    let s = format!("{}", EnumAutoBox::from(Box::new(UnitError)));
    assert_eq!(&s[..], "Boxed error: An error has occurred.");
}

#[derive(Debug, Error)]
pub enum TestsNonFieldDisplayValues {
    #[error(display = "{}", STATIC_STR)]
    A,
    #[error(display = "{}", get_static_str())]
    B,
    #[error(display = "{}", Self::some_method(self))]
    C,
    #[error(display = "{}", u8::max_value())]
    D,
    #[error(display = "{}", std::u8::MAX)]
    E,
}

impl TestsNonFieldDisplayValues {
    fn some_method(&self) -> &'static str {
        "some_method"
    }
}

static STATIC_STR: &str = "STATIC_STR";

fn get_static_str() -> &'static str {
    STATIC_STR
}

#[test]
fn non_field_display() {
    let s = format!("{}", TestsNonFieldDisplayValues::A);
    assert_eq!(&s, "STATIC_STR");

    let s = format!("{}", TestsNonFieldDisplayValues::B);
    assert_eq!(&s, "STATIC_STR");

    let s = format!("{}", TestsNonFieldDisplayValues::C);
    assert_eq!(&s, "some_method");

    let s = format!("{}", TestsNonFieldDisplayValues::D);
    assert_eq!(&s, "255");

    let s = format!("{}", TestsNonFieldDisplayValues::E);
    assert_eq!(&s, "255");
}
