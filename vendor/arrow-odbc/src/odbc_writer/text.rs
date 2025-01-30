use arrow::array::{Array, LargeStringArray, StringArray};
use odbc_api::buffers::{AnySliceMut, BufferDesc, TextColumnSliceMut};

use super::{WriteStrategy, WriterError};

#[cfg(not(target_os = "windows"))]
pub type Utf8ToNativeText = Utf8ToNarrow;

#[cfg(target_os = "windows")]
pub type Utf8ToNativeText = Utf8ToWide;

#[cfg(not(target_os = "windows"))]
pub type LargeUtf8ToNativeText = LargeUtf8ToNarrow;

#[cfg(target_os = "windows")]
pub type LargeUtf8ToNativeText = LargeUtf8ToWide;

pub struct Utf8ToNarrow;

impl WriteStrategy for Utf8ToNarrow {
    fn buffer_desc(&self) -> BufferDesc {
        BufferDesc::Text { max_str_len: 1 }
    }

    fn write_rows(
        &self,
        param_offset: usize,
        to: AnySliceMut<'_>,
        from: &dyn Array,
    ) -> Result<(), WriterError> {
        let from = from.as_any().downcast_ref::<StringArray>().unwrap();
        let to = to.as_text_view().unwrap();
        insert_into_narrow_slice(from.iter(), to, param_offset)?;
        Ok(())
    }
}

pub struct LargeUtf8ToNarrow;

impl WriteStrategy for LargeUtf8ToNarrow {
    fn buffer_desc(&self) -> BufferDesc {
        BufferDesc::Text { max_str_len: 1 }
    }

    fn write_rows(
        &self,
        param_offset: usize,
        to: AnySliceMut<'_>,
        from: &dyn Array,
    ) -> Result<(), WriterError> {
        let from = from.as_any().downcast_ref::<LargeStringArray>().unwrap();
        let to = to.as_text_view().unwrap();
        insert_into_narrow_slice(from.iter(), to, param_offset)?;
        Ok(())
    }
}

fn insert_into_narrow_slice<'a>(
    from: impl Iterator<Item = Option<&'a str>>,
    mut to: TextColumnSliceMut<u8>,
    param_offset: usize,
) -> Result<(), WriterError> {
    for (row_index, element) in from.enumerate() {
        // Total number of rows written into the inserter (`to`). This includes the values from the
        // current batch (`row_index`), as well as the ones from the previous batches
        // (`param_offset`). In case of reallocation, we need to copy all these values. Also, this
        // is the index of the element we currently want to write.
        let num_rows_written_so_far = param_offset + row_index;
        if let Some(text) = element {
            to.ensure_max_element_length(text.len(), num_rows_written_so_far)
                .map_err(WriterError::RebindBuffer)?;
            to.set_cell(num_rows_written_so_far, Some(text.as_bytes()))
        } else {
            to.set_cell(num_rows_written_so_far, None);
        }
    }
    Ok(())
}

pub struct Utf8ToWide;

impl WriteStrategy for Utf8ToWide {
    fn buffer_desc(&self) -> BufferDesc {
        BufferDesc::WText { max_str_len: 1 }
    }

    fn write_rows(
        &self,
        param_offset: usize,
        to: AnySliceMut<'_>,
        from: &dyn Array,
    ) -> Result<(), WriterError> {
        let from = from.as_any().downcast_ref::<StringArray>().unwrap();
        let to = to.as_w_text_view().unwrap();
        insert_into_wide_slice(from.iter(), to, param_offset)?;
        Ok(())
    }
}

pub struct LargeUtf8ToWide;

impl WriteStrategy for LargeUtf8ToWide {
    fn buffer_desc(&self) -> BufferDesc {
        BufferDesc::WText { max_str_len: 1 }
    }

    fn write_rows(
        &self,
        param_offset: usize,
        to: AnySliceMut<'_>,
        from: &dyn Array,
    ) -> Result<(), WriterError> {
        let from = from.as_any().downcast_ref::<LargeStringArray>().unwrap();
        let to = to.as_w_text_view().unwrap();
        insert_into_wide_slice(from.iter(), to, param_offset)?;
        Ok(())
    }
}

fn insert_into_wide_slice<'a>(
    from: impl Iterator<Item = Option<&'a str>>,
    mut to: TextColumnSliceMut<u16>,
    at: usize,
) -> Result<(), WriterError> {
    // We must first encode the utf8 input to utf16. We reuse this buffer for that in order to avoid
    // allocations.
    let mut utf_16 = Vec::new();
    for (row_index, element) in from.enumerate() {
        // Total number of rows written into the inserter (`to`). This includes the values from the
        // current batch (`row_index`), as well as the ones from the previous batches (`at`). In
        // case of reallocation, we need to copy all these values. Also, this is the index of the
        // element we currently want to write.
        let num_rows_written_so_far = at + row_index;
        if let Some(text) = element {
            utf_16.extend(text.encode_utf16());
            to.ensure_max_element_length(utf_16.len(), num_rows_written_so_far)
                .map_err(WriterError::RebindBuffer)?;
            to.set_cell(num_rows_written_so_far, Some(&utf_16));
            utf_16.clear();
        } else {
            to.set_cell(num_rows_written_so_far, None);
        }
    }
    Ok(())
}
