use std::{f32, f64};

use rusqlite::types::{FromSql, Value};
use rusqlite::Row;
use serde::de::{DeserializeSeed, EnumAccess, IntoDeserializer, MapAccess, SeqAccess, VariantAccess, Visitor};
use serde::{forward_to_deserialize_any, Deserializer};

pub use iter::{DeserRows, DeserRowsRef};

use crate::{Error, Result};

mod iter;

macro_rules! forward_to_row_value_deserializer {
	($($fun:ident)*) => {
		$(
			fn $fun<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
				self.row_value().$fun(visitor)
			}
		)*
	}
}

/// Deserializer for `rusqlite::Row`
///
/// You shouldn't use it directly, but via the crate's `from_row()` function. Check the crate documentation for example.
pub struct RowDeserializer<'row, 'stmt, 'cols> {
	row: &'row Row<'stmt>,
	columns: &'cols [String],
}

impl<'row, 'stmt, 'cols> RowDeserializer<'row, 'stmt, 'cols> {
	pub fn from_row_with_columns(row: &'row Row<'stmt>, columns: &'cols [String]) -> Self {
		Self { row, columns }
	}

	fn row_value(&self) -> RowValue<'row, 'stmt> {
		RowValue { row: self.row, idx: 0 }
	}
}

impl<'de> Deserializer<'de> for RowDeserializer<'de, '_, '_> {
	type Error = Error;

	fn deserialize_unit_struct<V: Visitor<'de>>(self, name: &'static str, visitor: V) -> Result<V::Value> {
		self.row_value().deserialize_unit_struct(name, visitor)
	}

	fn deserialize_newtype_struct<V: Visitor<'de>>(self, _name: &'static str, visitor: V) -> Result<V::Value> {
		visitor.visit_newtype_struct(self.row_value())
	}

	fn deserialize_tuple<V: Visitor<'de>>(self, _len: usize, visitor: V) -> Result<V::Value> {
		visitor.visit_seq(RowSeqAccess { idx: 0, de: self })
	}

	fn deserialize_map<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
		visitor.visit_map(RowMapAccess { idx: 0, de: self })
	}

	fn deserialize_struct<V: Visitor<'de>>(
		self,
		_name: &'static str,
		_fields: &'static [&'static str],
		visitor: V,
	) -> Result<V::Value> {
		self.deserialize_map(visitor)
	}

	fn deserialize_enum<V: Visitor<'de>>(
		self,
		name: &'static str,
		variants: &'static [&'static str],
		visitor: V,
	) -> Result<V::Value> {
		self.row_value().deserialize_enum(name, variants, visitor)
	}

	forward_to_row_value_deserializer! {
		deserialize_bool
		deserialize_f32
		deserialize_f64
		deserialize_option
		deserialize_unit
		deserialize_any
		deserialize_byte_buf
	}

	forward_to_deserialize_any! {
		i8 i16 i32 i64 u8 u16 u32 u64 char str string bytes
		seq tuple_struct identifier ignored_any
	}
}

struct RowValue<'row, 'stmt> {
	idx: usize,
	row: &'row Row<'stmt>,
}

impl<'row> RowValue<'row, '_> {
	fn value<T: FromSql>(&self) -> Result<T> {
		self.row.get(self.idx).map_err(Error::from)
	}

	fn deserialize_any_helper<V: Visitor<'row>>(self, visitor: V, value: Value) -> Result<V::Value> {
		match value {
			Value::Null => visitor.visit_none(),
			Value::Integer(val) => visitor.visit_i64(val),
			Value::Real(val) => visitor.visit_f64(val),
			Value::Text(val) => visitor.visit_string(val),
			Value::Blob(val) => visitor.visit_seq(val.into_deserializer()),
		}
	}
}

impl<'de> Deserializer<'de> for RowValue<'de, '_> {
	type Error = Error;

	fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
		let val = self.value()?;
		self.deserialize_any_helper(visitor, val)
	}

	fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
		match self.value()? {
			Value::Integer(val) => visitor.visit_bool(val != 0),
			Value::Real(val) => visitor.visit_bool(val != 0.),
			val => self.deserialize_any_helper(visitor, val),
		}
	}

	fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
		match self.value()? {
			Value::Null => visitor.visit_f32(f32::NAN),
			val => self.deserialize_any_helper(visitor, val),
		}
	}

	fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
		match self.value()? {
			Value::Null => visitor.visit_f64(f64::NAN),
			val => self.deserialize_any_helper(visitor, val),
		}
	}

	fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
		visitor.visit_byte_buf(self.value()?)
	}

	fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
		match self.value()? {
			Value::Null => visitor.visit_none(),
			_ => visitor.visit_some(self),
		}
	}

	fn deserialize_unit<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
		match self.value()? {
			Value::Null => visitor.visit_unit(),
			val => self.deserialize_any_helper(visitor, val),
		}
	}

	fn deserialize_unit_struct<V: Visitor<'de>>(self, name: &'static str, visitor: V) -> Result<V::Value> {
		match self.value()? {
			Value::Text(ref val) if val == name => visitor.visit_unit(),
			val => self.deserialize_any_helper(visitor, val),
		}
	}

	fn deserialize_enum<V: Visitor<'de>>(
		self,
		_name: &'static str,
		_variants: &'static [&'static str],
		visitor: V,
	) -> Result<V::Value> {
		visitor.visit_enum(RowEnumAccess(self.value()?))
	}

	forward_to_deserialize_any! {
		i8 i16 i32 i64 u8 u16 u32 u64 char str string bytes
		newtype_struct seq tuple
		tuple_struct map struct identifier ignored_any
	}
}

struct RowMapAccess<'row, 'stmt, 'cols> {
	idx: usize,
	de: RowDeserializer<'row, 'stmt, 'cols>,
}

impl<'de> MapAccess<'de> for RowMapAccess<'de, '_, '_> {
	type Error = Error;

	fn next_key_seed<K: DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<K::Value>> {
		if self.idx >= self.de.columns.len() {
			Ok(None)
		} else {
			let column = self.de.columns[self.idx].as_str();
			seed
				.deserialize(column.into_deserializer())
				.map(Some)
				.map_err(|e| add_field_to_error(e, column))
		}
	}

	fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value> {
		let out = seed
			.deserialize(RowValue {
				idx: self.idx,
				row: self.de.row,
			})
			.map_err(|e| add_field_to_error(e, &self.de.columns[self.idx]));
		self.idx += 1;
		out
	}
}

struct RowSeqAccess<'row, 'stmt, 'cols> {
	idx: usize,
	de: RowDeserializer<'row, 'stmt, 'cols>,
}

impl<'de> SeqAccess<'de> for RowSeqAccess<'de, '_, '_> {
	type Error = Error;

	fn next_element_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>> {
		let out = seed
			.deserialize(RowValue {
				idx: self.idx,
				row: self.de.row,
			})
			.map(Some)
			.map_err(|e| add_field_to_error(e, &self.de.columns[self.idx]));
		self.idx += 1;
		out
	}
}

struct RowEnumAccess(String);

impl<'de> EnumAccess<'de> for RowEnumAccess {
	type Error = Error;
	type Variant = RowVariantAccess;

	fn variant_seed<V: DeserializeSeed<'de>>(self, seed: V) -> Result<(V::Value, Self::Variant)> {
		seed.deserialize(self.0.into_deserializer()).map(|v| (v, RowVariantAccess))
	}
}

struct RowVariantAccess;

impl<'de> VariantAccess<'de> for RowVariantAccess {
	type Error = Error;

	fn unit_variant(self) -> Result<()> {
		Ok(())
	}

	fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, _seed: T) -> Result<T::Value> {
		Err(Error::de_unsupported("newtype_variant"))
	}
	fn tuple_variant<V: Visitor<'de>>(self, _len: usize, _visitor: V) -> Result<V::Value> {
		Err(Error::de_unsupported("tuple_variant"))
	}
	fn struct_variant<V: Visitor<'de>>(self, _fields: &'static [&'static str], _visitor: V) -> Result<V::Value> {
		Err(Error::de_unsupported("struct_variant"))
	}
}

fn add_field_to_error(mut error: Error, error_column: &str) -> Error {
	if let Error::Deserialization { column, .. } = &mut error {
		*column = Some(error_column.to_string());
	}
	error
}
