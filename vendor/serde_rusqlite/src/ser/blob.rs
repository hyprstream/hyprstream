use serde::ser;

use crate::{Error, Result};

pub struct BlobSerializer {
	pub buf: Vec<u8>,
}

impl ser::SerializeSeq for BlobSerializer {
	type Ok = Box<dyn rusqlite::types::ToSql>;
	type Error = Error;

	fn serialize_element<T: ?Sized + serde::Serialize>(&mut self, value: &T) -> Result<()> {
		self.buf.push(value.serialize(U8Serializer)?);
		Ok(())
	}

	fn end(self) -> Result<Self::Ok> {
		Ok(Box::new(self.buf))
	}
}

pub struct U8Serializer;

impl ser::Serializer for U8Serializer {
	type Ok = u8;
	type Error = Error;
	type SerializeSeq = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeTuple = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeTupleStruct = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeTupleVariant = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeMap = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeStruct = ser::Impossible<Self::Ok, Self::Error>;
	type SerializeStructVariant = ser::Impossible<Self::Ok, Self::Error>;

	fn serialize_u8(self, v: u8) -> Result<Self::Ok> {
		Ok(v)
	}

	ser_unimpl!(serialize_bool, bool);
	ser_unimpl!(serialize_i8, i8);
	ser_unimpl!(serialize_i16, i16);
	ser_unimpl!(serialize_i32, i32);
	ser_unimpl!(serialize_i64, i64);
	ser_unimpl!(serialize_u16, u16);
	ser_unimpl!(serialize_u32, u32);
	ser_unimpl!(serialize_u64, u64);
	ser_unimpl!(serialize_f32, f32);
	ser_unimpl!(serialize_f64, f64);
	ser_unimpl!(serialize_char, char);
	ser_unimpl!(serialize_str, &str);
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
		self.serialize_unit()
	}
	fn serialize_unit_variant(self, _name: &'static str, _variant_index: u32, _variant: &'static str) -> Result<Self::Ok> {
		self.serialize_unit()
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
