use serde::ser;

use crate::{Error, NamedParamSlice, Result};

use super::tosql::ToSqlSerializer;

/// Serializer into `NamedParamSlice`
///
/// You shouldn't use it directly, but via the crate's `to_params_named()` function. Check the crate documentation for example.
#[derive(Default)]
pub struct NamedSliceSerializer<'f> {
	pub result: NamedParamSlice,
	entry_key: Option<String>,
	only_fields: &'f [&'f str],
}

impl<'f> NamedSliceSerializer<'f> {
	pub fn with_only_fields(only_fields: &'f [&'f str]) -> Self {
		Self {
			result: NamedParamSlice::default(),
			entry_key: None,
			only_fields,
		}
	}

	#[inline]
	fn add_entry(&mut self, key: &str, value: impl serde::Serialize) -> Result<()> {
		if self.only_fields.is_empty() || self.only_fields.contains(&key) {
			self.result.push((format!(":{}", key), value.serialize(ToSqlSerializer)?));
		}
		Ok(())
	}
}

impl ser::Serializer for NamedSliceSerializer<'_> {
	type Ok = NamedParamSlice;
	type Error = Error;
	type SerializeSeq = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeTuple = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeTupleStruct = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeTupleVariant = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeMap = Self;
	type SerializeStruct = Self;
	type SerializeStructVariant = Self;

	fn serialize_none(self) -> Result<Self::Ok> {
		Err(Error::ser_unsupported("None"))
	}

	fn serialize_some<T: ?Sized + serde::Serialize>(self, value: &T) -> Result<Self::Ok> {
		value.serialize(self)
	}

	fn serialize_unit(self) -> Result<Self::Ok> {
		Err(Error::ser_unsupported("()"))
	}

	fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok> {
		Err(Error::ser_unsupported("unit_struct"))
	}

	fn serialize_unit_variant(self, _name: &'static str, _variant_index: u32, variant: &'static str) -> Result<Self::Ok> {
		self.serialize_str(variant)
	}

	fn serialize_newtype_struct<T: ?Sized + serde::Serialize>(self, _name: &'static str, value: &T) -> Result<Self::Ok> {
		value.serialize(self)
	}

	fn serialize_newtype_variant<T: ?Sized + serde::Serialize>(
		self,
		_name: &'static str,
		_variant_index: u32,
		_variant: &'static str,
		value: &T,
	) -> Result<Self::Ok> {
		value.serialize(self)
	}

	ser_unimpl!(serialize_bool, bool);
	ser_unimpl!(serialize_i8, i8);
	ser_unimpl!(serialize_i16, i16);
	ser_unimpl!(serialize_i32, i32);
	ser_unimpl!(serialize_i64, i64);
	ser_unimpl!(serialize_u8, u8);
	ser_unimpl!(serialize_u16, u16);
	ser_unimpl!(serialize_u32, u32);
	ser_unimpl!(serialize_u64, u64);
	ser_unimpl!(serialize_f32, f32);
	ser_unimpl!(serialize_f64, f64);
	ser_unimpl!(serialize_str, &str);
	ser_unimpl!(serialize_char, char);
	ser_unimpl!(serialize_bytes, &[u8]);

	fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
		Err(Error::ser_unsupported("seq"))
	}
	fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
		Err(Error::ser_unsupported("tuple"))
	}
	fn serialize_tuple_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeTupleStruct> {
		Err(Error::ser_unsupported("tuple_struct"))
	}
	fn serialize_tuple_variant(
		self,
		_name: &'static str,
		_variant_index: u32,
		_variant: &'static str,
		_len: usize,
	) -> Result<Self::SerializeTupleVariant> {
		Err(Error::ser_unsupported("tuple_variant"))
	}
	fn serialize_map(mut self, len: Option<usize>) -> Result<Self::SerializeMap> {
		if let Some(len) = len {
			self.result.reserve_exact(len);
		}
		Ok(self)
	}
	fn serialize_struct(mut self, _name: &'static str, len: usize) -> Result<Self::SerializeStruct> {
		self.result.reserve_exact(len);
		Ok(self)
	}
	fn serialize_struct_variant(
		mut self,
		_name: &'static str,
		_variant_index: u32,
		_variant: &'static str,
		len: usize,
	) -> Result<Self::SerializeStructVariant> {
		self.result.reserve_exact(len);
		Ok(self)
	}
}

impl ser::SerializeMap for NamedSliceSerializer<'_> {
	type Ok = NamedParamSlice;
	type Error = Error;

	fn serialize_key<T: ?Sized + serde::Serialize>(&mut self, key: &T) -> Result<()> {
		self.entry_key = Some(key.serialize(ColumNameSerializer)?);
		Ok(())
	}

	fn serialize_value<T: ?Sized + serde::Serialize>(&mut self, value: &T) -> Result<()> {
		if let Some(column_name) = self.entry_key.take() {
			self.add_entry(&column_name, value)?;
		}
		Ok(())
	}

	fn end(self) -> Result<Self::Ok> {
		Ok(self.result)
	}
}

impl ser::SerializeStruct for NamedSliceSerializer<'_> {
	type Ok = NamedParamSlice;
	type Error = Error;

	fn serialize_field<T: ?Sized + serde::Serialize>(&mut self, key: &'static str, value: &T) -> Result<()> {
		self.add_entry(key, value)
	}

	fn end(self) -> Result<Self::Ok> {
		Ok(self.result)
	}
}

impl ser::SerializeStructVariant for NamedSliceSerializer<'_> {
	type Ok = NamedParamSlice;
	type Error = Error;

	fn serialize_field<T: ?Sized + serde::Serialize>(&mut self, key: &'static str, value: &T) -> Result<()> {
		self.add_entry(key, value)
	}

	fn end(self) -> Result<Self::Ok> {
		Ok(self.result)
	}
}

struct ColumNameSerializer;

impl ser::Serializer for ColumNameSerializer {
	type Ok = String;
	type Error = Error;
	type SerializeSeq = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeTuple = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeTupleStruct = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeTupleVariant = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeMap = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeStruct = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeStructVariant = ser::Impossible<Self::Ok, Self::Error>;

	fn serialize_char(self, v: char) -> Result<Self::Ok> {
		Ok(v.to_string())
	}

	fn serialize_str(self, v: &str) -> Result<Self::Ok> {
		Ok(v.into())
	}

	ser_unimpl!(serialize_bool, bool);
	ser_unimpl!(serialize_i8, i8);
	ser_unimpl!(serialize_i16, i16);
	ser_unimpl!(serialize_i32, i32);
	ser_unimpl!(serialize_i64, i64);
	ser_unimpl!(serialize_u8, u8);
	ser_unimpl!(serialize_u16, u16);
	ser_unimpl!(serialize_u32, u32);
	ser_unimpl!(serialize_u64, u64);
	ser_unimpl!(serialize_f32, f32);
	ser_unimpl!(serialize_f64, f64);
	ser_unimpl!(serialize_bytes, &[u8]);

	fn serialize_none(self) -> Result<Self::Ok> {
		Err(Error::ser_unsupported("None"))
	}
	fn serialize_some<T: ?Sized + serde::Serialize>(self, _value: &T) -> Result<Self::Ok> {
		Err(Error::ser_unsupported("Some"))
	}
	fn serialize_unit(self) -> Result<Self::Ok> {
		Err(Error::ser_unsupported("()"))
	}
	fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok> {
		Err(Error::ser_unsupported("unit_struct"))
	}
	fn serialize_unit_variant(self, _name: &'static str, _variant_index: u32, _variant: &'static str) -> Result<Self::Ok> {
		Err(Error::ser_unsupported("unit_variant"))
	}
	fn serialize_newtype_struct<T: ?Sized + serde::Serialize>(self, _name: &'static str, _value: &T) -> Result<Self::Ok> {
		Err(Error::ser_unsupported("newtype_struct"))
	}
	fn serialize_newtype_variant<T: ?Sized + serde::Serialize>(
		self,
		_name: &'static str,
		_variant_index: u32,
		_variant: &'static str,
		_value: &T,
	) -> Result<Self::Ok> {
		Err(Error::ser_unsupported("newtype_variant"))
	}
	fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
		Err(Error::ser_unsupported("seq"))
	}
	fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
		Err(Error::ser_unsupported("tuple"))
	}
	fn serialize_tuple_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeTupleStruct> {
		Err(Error::ser_unsupported("tuple_struct"))
	}
	fn serialize_tuple_variant(
		self,
		_name: &'static str,
		_variant_index: u32,
		_variant: &'static str,
		_len: usize,
	) -> Result<Self::SerializeTupleVariant> {
		Err(Error::ser_unsupported("type_variant"))
	}
	fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
		Err(Error::ser_unsupported("map"))
	}
	fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeStruct> {
		Err(Error::ser_unsupported("struct"))
	}
	fn serialize_struct_variant(
		self,
		_name: &'static str,
		_variant_index: u32,
		_variant: &'static str,
		_len: usize,
	) -> Result<Self::SerializeStructVariant> {
		Err(Error::ser_unsupported("struct_variant"))
	}
}
