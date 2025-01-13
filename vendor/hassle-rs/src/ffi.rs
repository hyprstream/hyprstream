#![allow(clippy::transmute_ptr_to_ptr)]
#![allow(clippy::too_many_arguments)]

use crate::os::{HRESULT, LPCWSTR, LPWSTR};
use com::{interfaces, interfaces::IUnknown, IID};
use std::ffi::c_void;

pub type DxcCreateInstanceProc<T> =
    extern "system" fn(rclsid: &IID, riid: &IID, ppv: *mut Option<T>) -> HRESULT;

pub type DxcCreateInstanceProc2 = extern "system" fn(
    malloc: /* IMalloc */ *const c_void,
    rclsid: &IID,
    riid: &IID,
    ppv: *mut *mut c_void,
) -> HRESULT;

pub const DFCC_DXIL: u32 = u32::from_le_bytes([b'D', b'X', b'I', b'L']);

interfaces! {
    #[uuid("8ba5fb08-5195-40e2-ac58-0d989c3a0102")]
    pub(crate) unsafe interface IDxcBlob: IUnknown {
        pub(crate) fn get_buffer_pointer(&self) -> *mut c_void;
        pub(crate) fn get_buffer_size(&self) -> usize;
    }

    #[uuid("7241d424-2646-4191-97c0-98e96e42fc68")]
    pub(crate) unsafe interface IDxcBlobEncoding: IDxcBlob {
        pub(crate) fn get_encoding(&self, known: *mut u32, code_page: *mut u32) -> HRESULT;
    }

    #[uuid("e5204dc7-d18c-4c3c-bdfb-851673980fe7")]
    pub(crate) unsafe interface IDxcLibrary: IUnknown {
        pub(crate) fn set_malloc(&self, malloc: *const c_void) -> HRESULT;
        pub(crate) fn create_blob_from_blob(
            &self,
            blob: IDxcBlob,
            offset: u32,
            length: u32,
            result_blob: *mut Option<IDxcBlob>,
        ) -> HRESULT;
        pub(crate) fn create_blob_from_file(
            &self,
            filename: LPCWSTR,
            code_page: *const u32,
            blob_encoding: *mut Option<IDxcBlobEncoding>,
        ) -> HRESULT;
        pub(crate) fn create_blob_with_encoding_from_pinned(
            &self,
            text: *const c_void,
            size: u32,
            code_page: u32,
            blob_encoding: *mut Option<IDxcBlobEncoding>,
        ) -> HRESULT;
        pub(crate) fn create_blob_with_encoding_on_heap_copy(
            &self,
            text: *const c_void,
            size: u32,
            code_page: u32,
            blob_encoding: *mut Option<IDxcBlobEncoding>,
        ) -> HRESULT;
        pub(crate) fn create_blob_with_encoding_on_malloc(
            &self,
            text: *const c_void,
            malloc: *const /* IMalloc */ c_void,
            size: u32,
            code_page: u32,
            blob_encoding: *mut Option<IDxcBlobEncoding>,
        ) -> HRESULT;
        pub(crate) fn create_include_handler(
            &self,
            include_handler: *mut Option<IDxcIncludeHandler>,
        ) -> HRESULT;
        pub(crate) fn create_stream_from_blob_read_only(
            &self,
            blob: IDxcBlob,
            stream: *mut *mut /* IStream */ c_void,
        ) -> HRESULT;
        pub(crate) fn get_blob_as_utf8(
            &self,
            blob: IDxcBlob,
            blob_encoding: *mut Option<IDxcBlobEncoding>,
        ) -> HRESULT;
        pub(crate) fn get_blob_as_utf16(
            &self,
            blob: IDxcBlob,
            blob_encoding: *mut Option<IDxcBlobEncoding>,
        ) -> HRESULT;
    }

    #[uuid("cedb484a-d4e9-445a-b991-ca21ca157dc2")]
    pub(crate) unsafe interface IDxcOperationResult: IUnknown {
        pub(crate) fn get_status(&self, status: *mut u32) -> HRESULT;
        pub(crate) fn get_result(&self, result: *mut Option<IDxcBlob>) -> HRESULT;
        pub(crate) fn get_error_buffer(&self, errors: *mut Option<IDxcBlobEncoding>) -> HRESULT;
    }

    #[uuid("7f61fc7d-950d-467f-b3e3-3c02fb49187c")]
    pub(crate) unsafe interface IDxcIncludeHandler: IUnknown {
        pub(crate) fn load_source(
            &self,
            filename: LPCWSTR,
            include_source: *mut Option<IDxcBlob>,
        ) -> HRESULT;
    }
}

#[repr(C)]
pub struct DxcDefine {
    pub name: LPCWSTR,
    pub value: LPCWSTR,
}

interfaces! {
    #[uuid("8c210bf3-011f-4422-8d70-6f9acb8db617")]
    pub(crate) unsafe interface IDxcCompiler: IUnknown {
        pub(crate) fn compile(
            &self,
            blob: IDxcBlob,
            source_name: LPCWSTR,
            entry_point: LPCWSTR,
            target_profile: LPCWSTR,
            arguments: *const LPCWSTR,
            arg_count: u32,
            defines: *const DxcDefine,
            def_count: u32,
            include_handler: Option<IDxcIncludeHandler>,
            result: *mut Option<IDxcOperationResult>,
        ) -> HRESULT;

        pub(crate) fn preprocess(
            &self,
            blob: IDxcBlob,
            source_name: LPCWSTR,
            arguments: *const LPCWSTR,
            arg_count: u32,
            defines: *const DxcDefine,
            def_count: u32,
            include_handler: Option<IDxcIncludeHandler>,
            result: *mut Option<IDxcOperationResult>,
        ) -> HRESULT;

        pub(crate) fn disassemble(
            &self,
            blob: IDxcBlob,
            disassembly: *mut Option<IDxcBlobEncoding>,
        ) -> HRESULT;
    }

    #[uuid("a005a9d9-b8bb-4594-b5c9-0e633bec4d37")]
    pub(crate) unsafe interface IDxcCompiler2: IDxcCompiler {
        pub(crate) fn compile_with_debug(
            &self,
            blob: IDxcBlob,
            source_name: LPCWSTR,
            entry_point: LPCWSTR,
            target_profile: LPCWSTR,
            arguments: *const LPCWSTR,
            arg_count: u32,
            defines: *const DxcDefine,
            def_count: u32,
            include_handler: Option<IDxcIncludeHandler>,
            result: *mut Option<IDxcOperationResult>,
            debug_blob_name: *mut LPWSTR,
            debug_blob: *mut Option<IDxcBlob>,
        ) -> HRESULT;
    }

    #[uuid("f1b5be2a-62dd-4327-a1c2-42ac1e1e78e6")]
    pub(crate) unsafe interface IDxcLinker: IUnknown {
        pub(crate) fn register_library(&self, lib_name: LPCWSTR, lib: IDxcBlob) -> HRESULT;

        pub(crate) fn link(
            &self,
            entry_name: LPCWSTR,
            target_profile: LPCWSTR,
            lib_names: *const LPCWSTR,
            lib_count: u32,
            arguments: *const LPCWSTR,
            arg_count: u32,
            result: *mut Option<IDxcOperationResult>,
        ) -> HRESULT;
    }
}

pub const DXC_VALIDATOR_FLAGS_DEFAULT: u32 = 0;
pub const DXC_VALIDATOR_FLAGS_IN_PLACE_EDIT: u32 = 1; // Validator is allowed to update shader blob in-place.
pub const DXC_VALIDATOR_FLAGS_ROOT_SIGNATURE_ONLY: u32 = 2;
pub const DXC_VALIDATOR_FLAGS_MODULE_ONLY: u32 = 4;
pub const DXC_VALIDATOR_FLAGS_VALID_MASK: u32 = 0x7;

interfaces! {
    #[uuid("a6e82bd2-1fd7-4826-9811-2857e797f49a")]
    pub(crate) unsafe interface IDxcValidator: IUnknown {
        pub(crate) fn validate(
            &self,
            shader: IDxcBlob,
            flags: u32,
            result: *mut Option<IDxcOperationResult>,
        ) -> HRESULT;
    }

    #[uuid("334b1f50-2292-4b35-99a1-25588d8c17fe")]
    pub(crate) unsafe interface IDxcContainerBuilder: IUnknown {
        pub(crate) fn load(&self, dxil_container_header: IDxcBlob) -> HRESULT;
        pub(crate) fn add_part(&self, four_cc: u32, source: IDxcBlob) -> HRESULT;
        pub(crate) fn remove_part(&self, four_cc: u32) -> HRESULT;
        pub(crate) fn seralize_container(
            &self,
            result: *mut Option<IDxcOperationResult>,
        ) -> HRESULT;
    }

    #[uuid("091f7a26-1c1f-4948-904b-e6e3a8a771d5")]
    pub(crate) unsafe interface IDxcAssembler: IUnknown {
        pub(crate) fn assemble_to_container(
            &self,
            shader: IDxcBlob,
            result: *mut Option<IDxcOperationResult>,
        ) -> HRESULT;
    }

    #[uuid("d2c21b26-8350-4bdc-976a-331ce6f4c54c")]
    pub(crate) unsafe interface IDxcContainerReflection: IUnknown {
        pub(crate) fn load(&self, container: IDxcBlob) -> HRESULT;
        pub(crate) fn get_part_count(&self, result: *mut u32) -> HRESULT;
        pub(crate) fn get_part_kind(&self, idx: u32, result: *mut u32) -> HRESULT;
        pub(crate) fn get_part_content(&self, idx: u32, result: *mut Option<IDxcBlob>) -> HRESULT;
        pub(crate) fn find_first_part_kind(&self, kind: u32, result: *mut u32) -> HRESULT;
        pub(crate) fn get_part_reflection(
            &self,
            idx: u32,
            iid: *const IID,
            object: *mut Option<IUnknown>,
        ) -> HRESULT;
    }

    #[uuid("5a58797d-a72c-478d-8ba2-efc6b0efe88e")]
    pub(crate) unsafe interface ID3D12ShaderReflection: IUnknown {
        pub(crate) fn get_desc(&self, p_desc: *mut c_void) -> HRESULT;
        pub(crate) fn get_constant_buffer_by_index(&self, index: u32) -> *mut c_void;
        pub(crate) fn get_constant_buffer_by_name(&self, name: *const c_void) -> *mut c_void;
        pub(crate) fn get_resource_binding_desc(
            &self,
            resource_index: u32,
            p_desc: *mut c_void,
        ) -> HRESULT;
        pub(crate) fn get_input_parameter_desc(
            &self,
            parameter_index: u32,
            p_desc: *mut c_void,
        ) -> HRESULT;
        pub(crate) fn get_output_parameter_desc(
            &self,
            parameter_index: u32,
            p_desc: *mut c_void,
        ) -> HRESULT;
        pub(crate) fn get_patch_constant_parameter_desc(
            &self,
            parameter_index: u32,
            p_desc: *mut c_void,
        ) -> HRESULT;
        pub(crate) fn get_variable_by_name(&self, name: *const c_void) -> *mut c_void;
        pub(crate) fn get_resource_binding_desc_by_name(
            &self,
            name: *const c_void,
            p_desc: *mut c_void,
        ) -> HRESULT;
        pub(crate) fn get_mov_instruction_count(&self) -> u32;
        pub(crate) fn get_movc_instruction_count(&self) -> u32;
        pub(crate) fn get_conversion_instruction_count(&self) -> u32;
        pub(crate) fn get_bitwise_instruction_count(&self) -> u32;
        pub(crate) fn get_gs_input_primitive(&self) -> u32;
        pub(crate) fn is_sample_frequency_shader(&self) -> bool;
        pub(crate) fn get_num_interface_slots(&self) -> u32;
        pub(crate) fn get_min_feature_level(&self, p_level: *mut c_void) -> HRESULT;
        pub(crate) fn get_thread_group_size(
            &self,
            size_x: *mut u32,
            size_y: *mut u32,
            size_z: *mut u32,
        ) -> u32;
        pub(crate) fn get_requires_flags(&self) -> u64;
    }

    #[uuid("ae2cd79f-cc22-453f-9b6b-b124e7a5204c")]
    pub(crate) unsafe interface IDxcOptimizerPass: IUnknown {
        pub(crate) fn get_option_name(&self, result: *mut LPWSTR) -> HRESULT;
        pub(crate) fn get_description(&self, result: *mut LPWSTR) -> HRESULT;
        pub(crate) fn get_option_arg_count(&self, count: *mut u32) -> HRESULT;
        pub(crate) fn get_option_arg_name(&self, arg_idx: u32, result: *mut LPWSTR) -> HRESULT;
        pub(crate) fn get_option_arg_description(
            &self,
            arg_idx: u32,
            result: *mut LPWSTR,
        ) -> HRESULT;
    }

    #[uuid("25740e2e-9cba-401b-9119-4fb42f39f270")]
    pub(crate) unsafe interface IDxcOptimizer: IUnknown {
        pub(crate) fn get_available_pass_count(&self, count: *mut u32) -> HRESULT;
        pub(crate) fn get_available_pass(
            &self,
            index: u32,
            result: *mut Option<IDxcOptimizerPass>,
        ) -> HRESULT;
        pub(crate) fn run_optimizer(
            &self,
            blob: IDxcBlob,
            options: *const LPCWSTR,
            option_count: u32,
            output_module: *mut Option<IDxcBlob>,
            output_text: *mut Option<IDxcBlobEncoding>,
        ) -> HRESULT;
    }
}

pub const DXC_VERSION_INFO_FLAGS_NONE: u32 = 0;
pub const DXC_VERSION_INFO_FLAGS_DEBUG: u32 = 1; // Matches VS_FF_DEBUG
pub const DXC_VERSION_INFO_FLAGS_INTERNAL: u32 = 2; // Internal Validator (non-signing)

interfaces! {
    #[uuid("b04f5b50-2059-4f12-a8ff-a1e0cde1cc7e")]
    pub(crate) unsafe interface IDxcVersionInfo: IUnknown {
        pub(crate) fn get_version(&self, major: *mut u32, minor: *mut u32) -> HRESULT;
        pub(crate) fn get_flags(&self, flags: *mut u32) -> HRESULT;
    }

    #[uuid("fb6904c4-42f0-4b62-9c46-983af7da7c83")]
    pub(crate) unsafe interface IDxcVersionInfo2: IUnknown {
        pub(crate) fn get_commit_info(
            &self,
            commit_count: *mut u32,
            commit_hash: *mut *mut u8,
        ) -> HRESULT;
    }
}

pub const CLSID_DxcCompiler: IID = IID {
    data1: 0x73e22d93,
    data2: 0xe6ce,
    data3: 0x47f3,
    data4: [0xb5, 0xbf, 0xf0, 0x66, 0x4f, 0x39, 0xc1, 0xb0],
};
pub const CLSID_DxcLinker: IID = IID {
    data1: 0xef6a8087,
    data2: 0xb0ea,
    data3: 0x4d56,
    data4: [0x9e, 0x45, 0xd0, 0x7e, 0x1a, 0x8b, 0x78, 0x6],
};
pub const CLSID_DxcDiaDataSource: IID = IID {
    data1: 0xcd1f6b73,
    data2: 0x2ab0,
    data3: 0x484d,
    data4: [0x8e, 0xdc, 0xeb, 0xe7, 0xa4, 0x3c, 0xa0, 0x9f],
};
pub const CLSID_DxcLibrary: IID = IID {
    data1: 0x6245d6af,
    data2: 0x66e0,
    data3: 0x48fd,
    data4: [0x80, 0xb4, 0x4d, 0x27, 0x17, 0x96, 0x74, 0x8c],
};
pub const CLSID_DxcValidator: IID = IID {
    data1: 0x8ca3e215,
    data2: 0xf728,
    data3: 0x4cf3,
    data4: [0x8c, 0xdd, 0x88, 0xaf, 0x91, 0x75, 0x87, 0xa1],
};
pub const CLSID_DxcAssembler: IID = IID {
    data1: 0xd728db68,
    data2: 0xf903,
    data3: 0x4f80,
    data4: [0x94, 0xcd, 0xdc, 0xcf, 0x76, 0xec, 0x71, 0x51],
};
pub const CLSID_DxcContainerReflection: IID = IID {
    data1: 0xb9f54489,
    data2: 0x55b8,
    data3: 0x400c,
    data4: [0xba, 0x3a, 0x16, 0x75, 0xe4, 0x72, 0x8b, 0x91],
};
pub const CLSID_DxcOptimizer: IID = IID {
    data1: 0xae2cd79f,
    data2: 0xcc22,
    data3: 0x453f,
    data4: [0x9b, 0x6b, 0xb1, 0x24, 0xe7, 0xa5, 0x20, 0x4c],
};
pub const CLSID_DxcContainerBuilder: IID = IID {
    data1: 0x94134294,
    data2: 0x411f,
    data3: 0x4574,
    data4: [0xb4, 0xd0, 0x87, 0x41, 0xe2, 0x52, 0x40, 0xd2],
};
