//! # Serde Rusqlite
//!
//! This crate provides convenience functions to bridge serde and rusqlite. With their help
//! you can "deserialize" rusqlite `Row`'s into serde `Deserialize` types and "serialize" types
//! implementing `Serialize` into bound query arguments (positional or named) that rusqlite expects.
//!
//! Serialization of named bound arguments is only supported from `struct`s and `map`s because other
//! serde types lack column name information. Likewise, serialization of positional bound arguments
//! is only supported from `tuple`s, `sequence`s and primitive non-iterable types. In the latter case
//! the result will be single-element vector. Each serialized field or element must implement
//! `rusqlite::types::ToSql`.
//!
//! For deserialization you can use two families of functions: `from_*()` and `from_*_with_columns()`.
//! The most used one is the former. The latter allows you to specify column names for types that need
//! them, but don't supply them. This includes different `Map` types like `HashMap`. Specifying columns
//! for deserialization into e.g. `struct` doesn't have any effect as the field list of the struct itself
//! will be used in any case.
//!
//! SQLite only supports 5 types: `NULL` (`None`), `INTEGER` (`i64`), `REAL` (`f64`), `TEXT` (`String`)
//! and `BLOB` (`Vec<u8>`). Corresponding rust types are inside brackets.
//!
//! Some types employ non-trivial handling, these are described below:
//!
//! * Serialization of `u64` will fail if it can't be represented by `i64` due to sqlite limitations.
//! * Simple `enum`s will be serialized as strings so:
//!
//!   ```
//!   enum Gender {
//!      M,
//!      F,
//!   }
//!   ```
//!
//!   will have two possible `TEXT` options in the database "M" and "F". Deserialization into `enum`
//!   from `TEXT` is also supported.
//! * `bool`s are serialized as `INTEGER`s 0 or 1, can be deserialized from `INTEGER` and `REAL` where
//!   0 and 0.0 are `false`, anything else is `true`.
//! * `f64` and `f32` values of `NaN` are serialized as `NULL`s. When deserializing such value `Option<f64>`
//!   will have value of `None` and `f64` will have value of `NaN`. The same applies to `f32`.
//! * `Bytes`, `ByteBuf` from `serde_bytes` are supported as optimized way of handling `BLOB`s.
//! * `unit` serializes to `NULL`.
//! * Only `sequence`s of `u8` are serialized and deserialized, `BLOB` database type is used. It's
//!   more optimal though to use `Bytes` and `ByteBuf` from `serde_bytes` for such fields.
//! * `unit_struct` serializes to `struct` name as `TEXT`, when deserializing the check is made to ensure
//!   that `struct` name coincides with the string in the database.
//!
//! # Examples
//! ```
//! use serde_derive::{Deserialize, Serialize};
//! use serde_rusqlite::*;
//!
//! #[derive(Serialize, Deserialize, Debug, PartialEq)]
//! struct Example {
//!    id: i64,
//!    name: String,
//! }
//!
//! let connection = rusqlite::Connection::open_in_memory().unwrap();
//! connection.execute("CREATE TABLE example (id INT, name TEXT)", []).unwrap();
//!
//! // using structure to generate named bound query arguments
//! let row1 = Example { id: 1, name: "first name".into() };
//! connection.execute("INSERT INTO example (id, name) VALUES (:id, :name)", to_params_named(&row1).unwrap().to_slice().as_slice()).unwrap();
//! // and limiting the set of fields that are to be serialized
//! let row2 = Example { id: 10, name: "second name".into() };
//! connection.execute("INSERT INTO example (id, name) VALUES (2, :name)", to_params_named_with_fields(&row2, &["name"]).unwrap().to_slice().as_slice()).unwrap();
//!
//! // using tuple to generate positional bound query arguments
//! let row2 = (3, "third name");
//! connection.execute("INSERT INTO example (id, name) VALUES (?, ?)", to_params(&row2).unwrap()).unwrap();
//!
//! // deserializing using query() and from_rows(), the most efficient way
//! let mut statement = connection.prepare("SELECT * FROM example").unwrap();
//! let mut res = from_rows::<Example>(statement.query([]).unwrap());
//! assert_eq!(res.next().unwrap().unwrap(), row1);
//! assert_eq!(res.next().unwrap().unwrap(), Example { id: 2, name: "second name".into() });
//!
//! // deserializing using query_and_then() and from_row(), incurs extra overhead in from_row() call
//! let mut statement = connection.prepare("SELECT * FROM example").unwrap();
//! let mut rows = statement.query_and_then([], from_row::<Example>).unwrap();
//! assert_eq!(rows.next().unwrap().unwrap(), row1);
//! assert_eq!(rows.next().unwrap().unwrap(), Example { id: 2, name: "second name".into() });
//!
//! // deserializing using query_and_then() and from_row_with_columns(), better performance than from_row()
//! let mut statement = connection.prepare("SELECT * FROM example").unwrap();
//! let columns = columns_from_statement(&statement);
//! let mut rows = statement.query_and_then([], |row| from_row_with_columns::<Example>(row, &columns)).unwrap();
//! assert_eq!(rows.next().unwrap().unwrap(), row1);
//! assert_eq!(rows.next().unwrap().unwrap(), Example { id: 2, name: "second name".into() });
//!
//! // deserializing using query() and from_rows_ref()
//! let mut statement = connection.prepare("SELECT * FROM example").unwrap();
//! let mut rows = statement.query([]).unwrap();
//! {
//!    // only first record is deserialized here
//!    let mut res = from_rows_ref::<Example>(&mut rows);
//!    assert_eq!(res.next().unwrap().unwrap(), row1);
//! }
//! // the second record is deserialized using the original Rows iterator
//! assert_eq!(from_row::<Example>(&rows.next().unwrap().unwrap()).unwrap(), Example { id: 2, name: "second name".into() });
//! ```

pub use rusqlite;
use rusqlite::{params_from_iter, ParamsFromIter};

pub use de::{DeserRows, DeserRowsRef, RowDeserializer};
pub use error::{Error, Result};
pub use ser::{NamedParamSlice, NamedSliceSerializer, PositionalParams, PositionalSliceSerializer};

pub mod de;
pub mod error;
pub mod ser;
#[cfg(test)]
mod tests;

/// Returns column names of the statement the way `from_row_with_columns()` method expects them
///
/// This function is needed because by default `column_names()` returns `Vec<&str>` which
/// ties it to the lifetime of the `rusqlite::Statement`. This way we won't be able to run for example
/// `.query_map()` because it mutably borrows `rusqlite::Statement` and by that time it's already borrowed
/// for columns. So this function owns all column names to detach them from the lifetime of `rusqlite::Statement`.
#[inline]
pub fn columns_from_statement(stmt: &rusqlite::Statement) -> Vec<String> {
	stmt.column_names().into_iter().map(str::to_owned).collect()
}

/// Deserializes an instance of `D: serde::Deserialize` from `rusqlite::Row`
///
/// Calling this function incurs allocation and processing overhead because we need to fetch column names from the row.
/// So use with care when calling this function in a loop or check `from_row_with_columns()` to avoid that overhead.
///
/// You should supply this function to `query_map()`.
#[inline]
pub fn from_row<D: serde::de::DeserializeOwned>(row: &rusqlite::Row) -> Result<D> {
	let columns = row.as_ref().column_names();
	let columns_ref = columns.iter().map(|x| x.to_string()).collect::<Vec<_>>();
	from_row_with_columns(row, &columns_ref)
}

/// Deserializes any instance of `D: serde::Deserialize` from `rusqlite::Row` with specified columns
///
/// Use this function over `from_row()` to avoid allocation and overhead for fetching column names. To get columns names
/// you can use `columns_from_statement()`.
///
/// You should use this function in the closure you supply to `query_map()`.
///
/// Note: `columns` is a slice of owned `String`s to be type compatible with what `columns_from_statement()`
/// returns. Most of the time the result of that function will be used as the argument so it makes little sense
/// to accept something like `&[impl AsRef<str>]` here. It will only make usage of the API less ergonomic. E.g.
/// There will be 2 generic type arguments to the `from_row_with_columns()` instead of one.
#[inline]
pub fn from_row_with_columns<D: serde::de::DeserializeOwned>(row: &rusqlite::Row, columns: &[String]) -> Result<D> {
	D::deserialize(RowDeserializer::from_row_with_columns(row, columns))
}

/// Returns iterator that owns `rusqlite::Rows` and deserializes all records from it into instances of `D: serde::Deserialize`
///
/// Also see `from_row()` for some specific info.
///
/// This function covers most of the use cases and is easier to use than the alternative `from_rows_ref()`.
#[inline]
pub fn from_rows<D: serde::de::DeserializeOwned>(rows: rusqlite::Rows) -> DeserRows<D> {
	DeserRows::new(rows)
}

/// Returns iterator that borrows `rusqlite::Rows` and deserializes records from it into instances of `D: serde::Deserialize`
///
/// Use this function instead of `from_rows()` when you still need iterator with the remaining rows after deserializing some
/// of them.
#[inline]
pub fn from_rows_ref<'rows, 'stmt, D: serde::de::DeserializeOwned>(
	rows: &'rows mut rusqlite::Rows<'stmt>,
) -> DeserRowsRef<'rows, 'stmt, D> {
	DeserRowsRef::new(rows)
}

/// Serializes an instance of `S: serde::Serialize` into structure for positional bound query arguments
///
/// To get the slice suitable for supplying to `query()` or `execute()` call `to_slice()` on the `Ok` result and
/// borrow it.
#[inline]
pub fn to_params<S: serde::Serialize>(obj: S) -> Result<ParamsFromIter<PositionalParams>> {
	obj.serialize(PositionalSliceSerializer::default()).map(params_from_iter)
}

/// Serializes an instance of `S: serde::Serialize` into structure for named bound query arguments
///
/// To get the slice suitable for supplying to `query_named()` or `execute_named()` call `to_slice()` on the `Ok` result
/// and borrow it.
#[inline]
pub fn to_params_named<S: serde::Serialize>(obj: S) -> Result<NamedParamSlice> {
	obj.serialize(NamedSliceSerializer::default())
}

/// Serializes only the specified `fields` of an instance of `S: serde::Serialize` into structure
/// for named bound query arguments
///
/// To get the slice suitable for supplying to `query_named()` or `execute_named()` call `to_slice()` on the `Ok` result
/// and borrow it.
#[inline]
pub fn to_params_named_with_fields<S: serde::Serialize>(obj: S, fields: &[&str]) -> Result<NamedParamSlice> {
	obj.serialize(NamedSliceSerializer::with_only_fields(fields))
}
