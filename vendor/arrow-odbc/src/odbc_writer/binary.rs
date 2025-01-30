use arrow::array::{Array, BinaryArray};
use odbc_api::buffers::{AnySliceMut, BufferDesc};

use super::{WriteStrategy, WriterError};

pub struct VariadicBinary {
    buffer_start_size: usize,
}

impl VariadicBinary {
    pub fn new(buffer_start_size: usize) -> Self {
        VariadicBinary { buffer_start_size }
    }
}

impl WriteStrategy for VariadicBinary {
    fn buffer_desc(&self) -> BufferDesc {
        BufferDesc::Binary {
            length: self.buffer_start_size,
        }
    }

    fn write_rows(
        &self,
        param_offset: usize,
        to: AnySliceMut<'_>,
        from: &dyn Array,
    ) -> Result<(), WriterError> {
        let from = from.as_any().downcast_ref::<BinaryArray>().unwrap();
        let mut to = to.as_bin_view().unwrap();
        for (row_index, element) in from.iter().enumerate() {
            if let Some(bytes) = element {
                to.ensure_max_element_length(bytes.len(), row_index)
                    .map_err(WriterError::RebindBuffer)?;
                to.set_cell(param_offset + row_index, Some(bytes))
            } else {
                to.set_cell(param_offset + row_index, None);
            }
        }
        Ok(())
    }
}
