# arrow_convert

Provides an API on top of [`arrow-rs`](https://github.com/apache/arrow-rs) to convert between rust types and Arrow. This repository was ported from the directly converted from [`arrow2-convert`](https://github.com/DataEngineeringLabs/arrow2-convert) library for use with `arrow-rs`.

The Arrow ecosystem provides many ways to convert between Arrow and other popular formats across several languages. This project aims to serve the need for rust-centric data pipelines to easily convert to/from Arrow with strong typing and arbitrary nesting.

## Example

The example below performs a round trip conversion of a struct with a single field. 

Please see the [complex_example.rs](https://github.com/Swoorup/arrow-convert/blob/main/arrow_convert/tests/complex_example.rs) for usage of the full functionality.

```rust
/// Simple example

use arrow::array::{Array, ArrayRef};
use arrow_convert::{deserialize::TryIntoCollection, serialize::TryIntoArrow, ArrowField, ArrowSerialize, ArrowDeserialize};

#[derive(Debug, Clone, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
pub struct Foo {
    name: String,
}

fn main() {
    // an item
    let original_array = [
        Foo { name: "hello".to_string() },
        Foo { name: "one more".to_string() },
        Foo { name: "good bye".to_string() },
    ];

    // serialize to an arrow array. try_into_arrow() is enabled by the TryIntoArrow trait
    let arrow_array: ArrayRef = original_array.try_into_arrow().unwrap();

    // which can be cast to an Arrow StructArray and be used for all kinds of IPC, FFI, etc.
    // supported by `arrow`
    let struct_array= arrow_array.as_any().downcast_ref::<arrow::array::StructArray>().unwrap();
    assert_eq!(struct_array.len(), 3);

    // deserialize back to our original vector via TryIntoCollection trait.
    let round_trip_array: Vec<Foo> = arrow_array.try_into_collection().unwrap();
    assert_eq!(round_trip_array, original_array);
}
```

## API

Types that implement the `ArrowField`, `ArrowSerialize` and `ArrowDeserialize` traits can be converted to/from Arrow via the `try_into_arrow` and the `try_into_collection` methods. 

The `ArrowField`, `ArrowSerialize` and `ArrowDeserialize` derive macros can be used to generate implementations of these traits for structs and enums. Custom implementations can also be defined for any type that needs to convert to/from Arrow by manually implementing the traits.

For serializing to arrow, `TryIntoArrow::try_into_arrow` can be used to serialize any iterable into an `arrow::Array` or a `arrow::Chunk`.  `arrow::Array` represents the in-memory Arrow layout. `arrow::Chunk` represents a column group and can be used with `arrow` API for other functionality such converting to parquet and arrow flight RPC.

For deserializing from arrow, the `TryIntoCollection::try_into_collection` can be used to deserialize from an `arrow::Array` representation into any container that implements `FromIterator`.

### Default implementations

Default implementations of the above traits are provided for the following:

- Numeric types
    - [`u8`], [`u16`], [`u32`], [`u64`], [`i8`], [`i16`], [`i32`], [`i64`], [`f32`], [`f64`]
    - [`i128`] is supported via the `type` attribute. Please see the [i128 section](#i128) for more details.
- Other types: 
    - [`bool`], [`String`], [`Binary`]
- Temporal types: 
    - [`chrono::NaiveDate`], [`chrono::NaiveDateTime`]
- Option<T> if T implements `ArrowField`
- Vec<T> if T implements `ArrowField`
- `[T; SIZE]` if T implements `ArrowField`
- Large Arrow types [`LargeBinary`], [`LargeString`], [`LargeList`] are supported via the `type` attribute. Please see the [complex_example.rs](./arrow_convert/tests/complex_example.rs) for usage.
- Fixed size types [`FixedSizeBinary`], [`FixedSizeList`] are supported via the `FixedSizeVec` type override.
    - Note: nesting of [`FixedSizeList`] is not supported.
- `TinyAsciiStr` from the [tinystr](https://github.com/zbraniecki/tinystr) crate (with the `tinystr` feature enabled)
- `Decimal` from the [rust_decimal](https://github.com/paupino/rust-decimal) crate (with the `rust_decimal` feature enabled)
- `Glam` vector and matrix types (with the `glam` feature enabled):
    - `Vec2`, `Vec3`, `Vec4`
    - `DVec2`, `DVec3`, `DVec4`
    - `BVec2`, `BVec3`, `BVec4`
    - `Mat2`, `Mat3`, `Mat4`
    - `DMat2`, `DMat3`, `DMat4`

### Enums

Enums are still an experimental feature and need to be integrated tested. Rust enum arrays are converted to a `Arrow::UnionArray`. Some additional notes on enums:

- Rust unit variants are represented using as the `bool` data type.

### i128

i128 represents a decimal number and requires the precision and scale to be specified to be used as an Arrow data type. The precision and scale can be specified by using a type override via the `I128` type. 

For example to use `i128` as a field in a struct:

```rust
use arrow_convert::field::I128;
use arrow_convert::ArrowField;

#[derive(Debug, ArrowField)]
struct S {
    #[arrow_field(type = "I128<32, 32>")]
    field: i128,
}
```

A `vec<i128>` can be converted. to/from arrow by using the `arrow_serialize_to_mutable_array` and `arrow_array_deserialize_iterator_as_type` methods. 

```rust
use arrow::array::{Array, ArrayBuilder, ArrayRef};
use arrow_convert::serialize::arrow_serialize_to_mutable_array;
use arrow_convert::deserialize::arrow_array_deserialize_iterator_as_type;
use arrow_convert::field::I128;
use std::borrow::Borrow;
use std::sync::Arc;

fn convert_i128() {
    let original_array = vec![1 as i128, 2, 3];
    let b: ArrayRef = Arc::new(arrow_serialize_to_mutable_array::<_, I128<32,32>, _>(
        &original_array).unwrap().finish());
    let round_trip: Vec<i128> = arrow_array_deserialize_iterator_as_type::<_, I128<32,32>>(
        b.borrow()).unwrap().collect();
    assert_eq!(original_array, round_trip);
}

```
### Nested Option Types

Since the Arrow format only supports one level of validity, nested option types such as `Option<Option<T>>`, after serialization to Arrow, will lose any intermediate nesting of None values. For example, `Some(None)` will be serialized to `None`, 

### Missing Features

- Support for generics, slices and reference is currently missing.

This is not an exhaustive list. Please open an issue if you need a feature.
## Memory

Pass-thru conversions perform a single memory copy. Deserialization performs a copy from arrow to the destination. Serialization performs a copy from the source to arrow. In-place deserialization is theoretically possible but currently not supported.

## Internals

### Similarities with Serde

The design is inspired by serde. The `ArrowSerialize` and `ArrowDeserialize` are analogs of serde's `Serialize` and `Deserialize` respectively.

However unlike serde's traits provide an exhaustive and flexible mapping to the serde data model, arrow_convert's traits provide a much more narrower mapping to arrow's data structures.

Specifically, the `ArrowSerialize` trait provides the logic to serialize a type to the corresponding `arrow::array::ArrayBuilder`. The `ArrowDeserialize` trait deserializes a type from the corresponding `arrow::array::ArrowArray`. 

### Workarounds

Features such as partial implementation specialization and generic associated types (currently only available in nightly builds) can greatly simplify the underlying implementation.

For example custom types need to explicitly enable Vec<T> serialization via the `arrow_enable_vec_for_type` macro on the primitive type. This is needed since Vec<u8> is a special type in Arrow, but without implementation specialization there's no way to special-case it.

Availability of generaic associated types would simplify the implementation for large and fixed types, since a generic ArrayBuilder can be defined. Ideally for code reusability, we wouldn’t have to reimplement `ArrowSerialize` and `ArrowDeserialize` for large and fixed size types since the primitive types are the same. However, this requires the trait functions to take a generic bounded mutable array as an argument instead of a single array type. This requires the `ArrowSerialize` and `ArrowDeserialize` implementations to be able to specify the bounds as part of the associated type, which is not possible without generic associated types.

As a result, we’re forced to sacrifice code reusability and introduce a little bit of complexity by providing separate `ArrowSerialize` and `ArrowDeserialize` implementations for large and fixed size types via placeholder structures. This also requires introducing the `Type` associated type to `ArrowField` so that the arrow type can be overriden via a macro field attribute without affecting the actual type.

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
