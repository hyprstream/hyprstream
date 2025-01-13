use com::Interface;

use crate::intellisense::ffi::*;
use crate::os::{CoTaskMemFree, BSTR, LPSTR};
use crate::utils::Result;
use crate::wrapper::Dxc;
use std::ffi::CString;
use std::mem::ManuallyDrop;

pub struct DxcIntellisense {
    inner: IDxcIntelliSense,
}

impl DxcIntellisense {
    fn new(inner: IDxcIntelliSense) -> Self {
        Self { inner }
    }

    pub fn get_default_editing_tu_options(&self) -> Result<DxcTranslationUnitFlags> {
        let mut options: DxcTranslationUnitFlags = DxcTranslationUnitFlags::NONE;
        unsafe { self.inner.get_default_editing_tu_options(&mut options) }
            .result_with_success(options)
    }

    pub fn create_index(&self) -> Result<DxcIndex> {
        let mut index = None;
        unsafe { self.inner.create_index(&mut index) }.result()?;
        Ok(DxcIndex::new(index.unwrap()))
    }

    pub fn create_unsaved_file(&self, file_name: &str, contents: &str) -> Result<DxcUnsavedFile> {
        let c_file_name = CString::new(file_name).expect("Failed to convert `file_name`");
        let c_contents = CString::new(contents).expect("Failed to convert `contents`");

        let mut file = None;
        unsafe {
            self.inner.create_unsaved_file(
                c_file_name.as_ptr(),
                c_contents.as_ptr(),
                contents.len() as u32,
                &mut file,
            )
        }
        .result()?;
        Ok(DxcUnsavedFile::new(file.unwrap()))
    }
}

pub struct DxcIndex {
    inner: IDxcIndex,
}

impl DxcIndex {
    fn new(inner: IDxcIndex) -> Self {
        Self { inner }
    }
}

impl DxcIndex {
    pub fn parse_translation_unit(
        &self,
        source_filename: &str,
        args: &[&str],
        unsaved_files: &[&DxcUnsavedFile],
        options: DxcTranslationUnitFlags,
    ) -> Result<DxcTranslationUnit> {
        let c_source_filename =
            CString::new(source_filename).expect("Failed to convert `source_filename`");

        let uf = unsaved_files
            .iter()
            .map(|unsaved_file| unsaved_file.inner.clone())
            .collect::<Vec<_>>();

        let mut c_args: Vec<CString> = vec![];
        let mut cliargs = vec![];

        for arg in args.iter() {
            let c_arg = CString::new(*arg).expect("Failed to convert `arg`");
            cliargs.push(c_arg.as_ptr().cast());
            c_args.push(c_arg);
        }

        let mut tu = None;

        unsafe {
            self.inner.parse_translation_unit(
                c_source_filename.as_ptr().cast(),
                cliargs.as_ptr(),
                cliargs.len() as i32,
                uf.as_ptr(),
                uf.len() as u32,
                options,
                &mut tu,
            )
        }
        .result()?;
        Ok(DxcTranslationUnit::new(tu.unwrap()))
    }
}

pub struct DxcUnsavedFile {
    inner: IDxcUnsavedFile,
}

impl DxcUnsavedFile {
    pub fn get_length(&self) -> Result<u32> {
        let mut length: u32 = 0;
        unsafe { self.inner.get_length(&mut length) }.result_with_success(length)
    }

    fn new(inner: IDxcUnsavedFile) -> Self {
        DxcUnsavedFile { inner }
    }
}

pub struct DxcTranslationUnit {
    inner: IDxcTranslationUnit,
}

impl DxcTranslationUnit {
    fn new(inner: IDxcTranslationUnit) -> Self {
        DxcTranslationUnit { inner }
    }

    pub fn get_file(&self, name: &[u8]) -> Result<DxcFile> {
        let mut file = None;
        unsafe { self.inner.get_file(name.as_ptr(), &mut file) }.result()?;
        Ok(DxcFile::new(file.unwrap()))
    }

    pub fn get_cursor(&self) -> Result<DxcCursor> {
        let mut cursor = None;
        unsafe { self.inner.get_cursor(&mut cursor) }.result()?;
        Ok(DxcCursor::new(cursor.unwrap()))
    }
}

pub struct DxcCursor {
    inner: IDxcCursor,
}

impl DxcCursor {
    fn new(inner: IDxcCursor) -> Self {
        DxcCursor { inner }
    }

    pub fn get_children(&self, skip: u32, max_count: u32) -> Result<Vec<DxcCursor>> {
        let mut result: *mut IDxcCursor = std::ptr::null_mut();
        let mut result_length: u32 = 0;

        unsafe {
            self.inner
                .get_children(skip, max_count, &mut result_length, &mut result)
        }
        .result()?;

        // get_children allocates a buffer to pass the result in.
        // Create a vector so that we get ownership of the `IDxcCursor(s) (received from get_children), instead of
        // having to clone (copy is intentionally not implemented) them and leaving unowned COM references alive.
        // It is wrapped in ManuallyDrop to free the underlying pointer by hand using CoTaskMemFree.
        // TODO: switch to Vec::from_raw_parts_in with custom deallocator when this is stabilized
        let child_cursors = ManuallyDrop::new(unsafe {
            Vec::from_raw_parts(result, result_length as usize, result_length as usize)
        })
        .drain(..)
        .map(DxcCursor::new)
        .collect::<Vec<_>>();

        unsafe { CoTaskMemFree(result.cast()) };
        Ok(child_cursors)
    }

    pub fn get_all_children(&self) -> Result<Vec<DxcCursor>> {
        const MAX_CHILDREN_PER_CHUNK: u32 = 10;
        let mut children = vec![];

        loop {
            let res = self.get_children(children.len() as u32, MAX_CHILDREN_PER_CHUNK)?;
            let res_len = res.len();
            children.extend(res);
            if res_len < MAX_CHILDREN_PER_CHUNK as usize {
                break Ok(children);
            }
        }
    }

    pub fn get_extent(&self) -> Result<DxcSourceRange> {
        let mut range = None;
        unsafe { self.inner.get_extent(&mut range) }.result()?;
        Ok(DxcSourceRange::new(range.unwrap()))
    }

    pub fn get_location(&self) -> Result<DxcSourceLocation> {
        let mut location = None;
        unsafe { self.inner.get_location(&mut location) }.result()?;
        Ok(DxcSourceLocation::new(location.unwrap()))
    }

    pub fn get_display_name(&self) -> Result<String> {
        let mut name: BSTR = std::ptr::null_mut();
        unsafe { self.inner.get_display_name(&mut name) }.result()?;
        Ok(crate::utils::from_bstr(name))
    }

    pub fn get_formatted_name(&self, formatting: DxcCursorFormatting) -> Result<String> {
        let mut name: BSTR = std::ptr::null_mut();
        unsafe { self.inner.get_formatted_name(formatting, &mut name) }.result()?;
        Ok(crate::utils::from_bstr(name))
    }

    pub fn get_qualified_name(&self, include_template_args: bool) -> Result<String> {
        let mut name: BSTR = std::ptr::null_mut();
        unsafe {
            self.inner
                .get_qualified_name(include_template_args, &mut name)
        }
        .result()?;
        Ok(crate::utils::from_bstr(name))
    }

    pub fn get_kind(&self) -> Result<DxcCursorKind> {
        let mut cursor_kind: DxcCursorKind = DxcCursorKind::UNEXPOSED_DECL;
        unsafe { self.inner.get_kind(&mut cursor_kind) }.result_with_success(cursor_kind)
    }

    pub fn get_kind_flags(&self) -> Result<DxcCursorKindFlags> {
        let mut cursor_kind_flags: DxcCursorKindFlags = DxcCursorKindFlags::NONE;
        unsafe { self.inner.get_kind_flags(&mut cursor_kind_flags) }
            .result_with_success(cursor_kind_flags)
    }

    pub fn get_semantic_parent(&self) -> Result<DxcCursor> {
        let mut inner = None;
        unsafe { self.inner.get_semantic_parent(&mut inner) }.result()?;
        Ok(DxcCursor::new(inner.unwrap()))
    }

    pub fn get_lexical_parent(&self) -> Result<DxcCursor> {
        let mut inner = None;
        unsafe { self.inner.get_lexical_parent(&mut inner) }.result()?;
        Ok(DxcCursor::new(inner.unwrap()))
    }

    pub fn get_cursor_type(&self) -> Result<DxcType> {
        let mut inner = None;
        unsafe { self.inner.get_cursor_type(&mut inner) }.result()?;
        Ok(DxcType::new(inner.unwrap()))
    }

    pub fn get_num_arguments(&self) -> Result<i32> {
        let mut result: i32 = 0;

        unsafe { self.inner.get_num_arguments(&mut result) }.result_with_success(result)
    }

    pub fn get_argument_at(&self, index: i32) -> Result<DxcCursor> {
        let mut inner = None;
        unsafe { self.inner.get_argument_at(index, &mut inner) }.result()?;
        Ok(DxcCursor::new(inner.unwrap()))
    }

    pub fn get_referenced_cursor(&self) -> Result<DxcCursor> {
        let mut inner = None;
        unsafe { self.inner.get_referenced_cursor(&mut inner) }.result()?;
        Ok(DxcCursor::new(inner.unwrap()))
    }

    pub fn get_definition_cursor(&self) -> Result<DxcCursor> {
        let mut inner = None;
        unsafe { self.inner.get_definition_cursor(&mut inner) }.result()?;
        Ok(DxcCursor::new(inner.unwrap()))
    }

    pub fn find_references_in_file(
        &self,
        file: &DxcFile,
        skip: u32,
        top: u32,
    ) -> Result<Vec<DxcCursor>> {
        let mut result: *mut IDxcCursor = std::ptr::null_mut();
        let mut result_length: u32 = 0;

        unsafe {
            self.inner.find_references_in_file(
                &file.inner,
                skip,
                top,
                &mut result_length,
                &mut result,
            )
        }
        .result()?;

        // find_references_in_file allocates a buffer to pass the result in.
        // Create a vector so that we get ownership of the `IDxcCursor(s) (received from find_references_in_file), instead
        // of having to clone (copy is intentionally not implemented) them and leaving unowned COM references alive.
        // It is wrapped in ManuallyDrop to free the underlying pointer by hand using CoTaskMemFree.
        // TODO: switch to Vec::from_raw_parts_in with custom deallocator when this is stabilized
        let child_cursors = ManuallyDrop::new(unsafe {
            Vec::from_raw_parts(result, result_length as usize, result_length as usize)
        })
        .drain(..)
        .map(DxcCursor::new)
        .collect::<Vec<_>>();

        unsafe { CoTaskMemFree(result.cast()) };
        Ok(child_cursors)
    }

    pub fn get_spelling(&self) -> Result<String> {
        let mut spelling: LPSTR = std::ptr::null_mut();
        unsafe { self.inner.get_spelling(&mut spelling) }.result()?;
        Ok(crate::utils::from_lpstr(spelling))
    }

    pub fn is_equal_to(&self, other: &DxcCursor) -> Result<bool> {
        let mut result: bool = false;
        unsafe { self.inner.is_equal_to(&other.inner, &mut result) }.result_with_success(result)
    }

    pub fn is_null(&mut self) -> Result<bool> {
        let mut result: bool = false;
        unsafe { IDxcCursor::is_null(&self.inner, &mut result) }.result_with_success(result)
    }

    pub fn is_definition(&self) -> Result<bool> {
        let mut result: bool = false;
        unsafe { self.inner.is_definition(&mut result) }.result_with_success(result)
    }

    pub fn get_snapped_child(&self, location: &DxcSourceLocation) -> Result<DxcCursor> {
        let mut inner = None;
        unsafe { self.inner.get_snapped_child(&location.inner, &mut inner) }.result()?;
        Ok(DxcCursor::new(inner.unwrap()))
    }

    pub fn get_source<'a>(&self, source: &'a str) -> Result<&'a str> {
        let range = self.get_extent()?;

        let DxcSourceOffsets {
            start_offset,
            end_offset,
        } = range.get_offsets()?;

        let source_range = (start_offset as usize)..(end_offset as usize);

        Ok(&source[source_range])
    }
}

pub struct DxcType {
    inner: IDxcType,
}

impl DxcType {
    fn new(inner: IDxcType) -> Self {
        DxcType { inner }
    }

    pub fn get_spelling(&self) -> Result<String> {
        let mut spelling: LPSTR = std::ptr::null_mut();
        unsafe { self.inner.get_spelling(&mut spelling) }
            .result_with_success(crate::utils::from_lpstr(spelling))
    }
}

pub struct DxcSourceLocation {
    inner: IDxcSourceLocation,
}

impl std::fmt::Debug for DxcSourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DxcSourceLocation")
            .field("inner", &self.inner)
            .finish()
    }
}

impl DxcSourceLocation {
    fn new(inner: IDxcSourceLocation) -> Self {
        DxcSourceLocation { inner }
    }
}

#[derive(Debug)]
pub struct DxcSourceOffsets {
    pub start_offset: u32,
    pub end_offset: u32,
}

pub struct DxcSourceRange {
    inner: IDxcSourceRange,
}

impl std::fmt::Debug for DxcSourceRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DxcSourceRange")
            .field("inner", &self.inner)
            .finish()
    }
}

impl DxcSourceRange {
    pub fn get_offsets(&self) -> Result<DxcSourceOffsets> {
        let mut start_offset: u32 = 0;
        let mut end_offset: u32 = 0;
        unsafe { self.inner.get_offsets(&mut start_offset, &mut end_offset) }.result_with_success(
            DxcSourceOffsets {
                start_offset,
                end_offset,
            },
        )
    }
}

impl DxcSourceRange {
    fn new(inner: IDxcSourceRange) -> Self {
        DxcSourceRange { inner }
    }
}

pub struct DxcFile {
    inner: IDxcFile,
}

impl DxcFile {
    fn new(inner: IDxcFile) -> Self {
        DxcFile { inner }
    }
}

impl Dxc {
    pub fn create_intellisense(&self) -> Result<DxcIntellisense> {
        let mut intellisense = None;

        self.get_dxc_create_instance()?(
            &CLSID_DxcIntelliSense,
            &IDxcIntelliSense::IID,
            &mut intellisense,
        )
        .result()?;
        Ok(DxcIntellisense::new(intellisense.unwrap()))
    }
}
