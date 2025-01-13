use rusqlite::types::{ToSql, Value};
use serde::ser;

use crate::{Error, Result};

use super::blob::BlobSerializer;

macro_rules! tosql_ser {
	($fun:ident, &$type:ty) => {
		fn $fun(self, v: &$type) -> Result<Self::Ok> {
			Ok(Box::new(v.to_owned()))
		}
	};
	($fun:ident, $type:ty) => {
		fn $fun(self, v: $type) -> Result<Self::Ok> {
			Ok(Box::new(v))
		}
	};
}

pub struct ToSqlSerializer;

impl ser::Serializer for ToSqlSerializer {
	type Ok = Box<dyn ToSql>;
	type Error = Error;
	type SerializeSeq = BlobSerializer;
	type SerializeTuple = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeTupleStruct = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeTupleVariant = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeMap = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeStruct = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeStructVariant = ser::Impossible<Self::Ok, Self::Error>;

	tosql_ser!(serialize_bool, bool);
	tosql_ser!(serialize_i8, i8);
	tosql_ser!(serialize_i16, i16);
	tosql_ser!(serialize_i32, i32);
	tosql_ser!(serialize_i64, i64);
	tosql_ser!(serialize_u8, u8);
	tosql_ser!(serialize_u16, u16);
	tosql_ser!(serialize_u32, u32);
	tosql_ser!(serialize_f64, f64);
	tosql_ser!(serialize_str, &str);
	tosql_ser!(serialize_bytes, &[u8]);

	fn serialize_u64(self, v: u64) -> Result<Self::Ok> {
		if v > i64::MAX as u64 {
			Err(Error::ValueTooLarge(format!("Value is too large to fit into i64: {}", v)))
		} else {
			self.serialize_i64(v as i64)
		}
	}

	fn serialize_f32(self, v: f32) -> Result<Self::Ok> {
		self.serialize_f64(f64::from(v))
	}

	fn serialize_char(self, v: char) -> Result<Self::Ok> {
		let mut char_bytes = [0; 4];
		self.serialize_str(v.encode_utf8(&mut char_bytes))
	}

	fn serialize_none(self) -> Result<Self::Ok> {
		Ok(Box::new(Value::Null))
	}

	fn serialize_some<T: ?Sized + serde::Serialize>(self, value: &T) -> Result<Self::Ok> {
		value.serialize(self)
	}

	fn serialize_unit(self) -> Result<Self::Ok> {
		self.serialize_none()
	}

	fn serialize_unit_struct(self, name: &'static str) -> Result<Self::Ok> {
		self.serialize_str(name)
	}

	fn serialize_unit_variant(self, _name: &'static str, _variant_index: u32, variant: &'static str) -> Result<Self::Ok> {
		self.serialize_str(variant)
	}

	fn serialize_newtype_struct<T: ?Sized + serde::Serialize>(self, _name: &'static str, value: &T) -> Result<Self::Ok> {
		value.serialize(self)
	}

	fn serialize_newtype_variant<T: ?Sized + serde::Serialize>(
		self,
		name: &'static str,
		_variant_index: u32,
		_variant: &'static str,
		value: &T,
	) -> Result<Self::Ok> {
		self.serialize_newtype_struct(name, value)
	}

	fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq> {
		Ok(BlobSerializer {
			buf: Vec::with_capacity(len.unwrap_or(0)),
		})
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
