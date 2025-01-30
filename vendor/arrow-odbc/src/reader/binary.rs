use std::{convert::TryInto, sync::Arc};

use arrow::array::{ArrayRef, BinaryBuilder, FixedSizeBinaryBuilder};
use odbc_api::buffers::{AnySlice, BufferDesc};

use super::{MappingError, ReadStrategy};

pub struct Binary {
    /// Maximum length in bytes of elements
    max_len: usize,
}

impl Binary {
    pub fn new(max_len: usize) -> Self {
        Self { max_len }
    }
}

impl ReadStrategy for Binary {
    fn buffer_desc(&self) -> BufferDesc {
        BufferDesc::Binary {
            length: self.max_len,
        }
    }

    fn fill_arrow_array(&self, column_view: AnySlice) -> Result<ArrayRef, MappingError> {
        let view = column_view.as_bin_view().unwrap();
        let mut builder = BinaryBuilder::new();
        for value in view.iter() {
            if let Some(bytes) = value {
                builder.append_value(bytes);
            } else {
                builder.append_null();
            }
        }
        Ok(Arc::new(builder.finish()))
    }
}

pub struct FixedSizedBinary {
    /// Length in bytes of elements
    len: u32,
}

impl FixedSizedBinary {
    pub fn new(len: u32) -> Self {
        Self { len }
    }
}

impl ReadStrategy for FixedSizedBinary {
    fn buffer_desc(&self) -> BufferDesc {
        BufferDesc::Binary {
            length: self.len as usize,
        }
    }

    fn fill_arrow_array(&self, column_view: AnySlice) -> Result<ArrayRef, MappingError> {
        let view = column_view.as_bin_view().unwrap();
        let mut builder = FixedSizeBinaryBuilder::new(self.len.try_into().unwrap());
        for value in view.iter() {
            if let Some(bytes) = value {
                builder.append_value(bytes).unwrap();
            } else {
                builder.append_null();
            }
        }
        Ok(Arc::new(builder.finish()))
    }
}
