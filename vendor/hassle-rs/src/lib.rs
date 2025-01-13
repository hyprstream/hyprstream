#![allow(dead_code)]
#![allow(non_upper_case_globals)]
#![allow(
    clippy::transmute_ptr_to_ptr, // Introduced by com-rs
    clippy::too_many_arguments, // We're wrapping and API outside of our control
    clippy::uninlined_format_args, // Unfavourable format; implies unneeded MSRV bump
)]

//! # Hassle
//!
//! This crate provides an FFI layer and idiomatic rust wrappers for the new [DirectXShaderCompiler](https://github.com/Microsoft/DirectXShaderCompiler) library.
//!
//! ## Simple example
//!
//! ```rust
//! use hassle_rs::compile_hlsl;
//!
//! let code = "
//!     Texture2D<float4> g_input    : register(t0, space0);
//!     RWTexture2D<float4> g_output : register(u0, space0);
//!
//!     [numthreads(8, 8, 1)]
//!     void copyCs(uint3 dispatchThreadId : SV_DispatchThreadID)
//!     {
//!         g_output[dispatchThreadId.xy] = g_input[dispatchThreadId.xy];
//!     }";
//!
//! let ir = compile_hlsl(
//!     "shader_filename.hlsl",
//!     code,
//!     "copyCs",
//!     "cs_6_1",
//!     &vec!["-spirv"],
//!     &vec![
//!         ("MY_DEFINE", Some("Value")),
//!         ("OTHER_DEFINE", None)
//!     ],
//! );
//! ```

pub mod fake_sign;
pub mod ffi;
pub mod os;
pub mod utils;
pub mod wrapper;

pub mod intellisense;

pub use crate::ffi::*;
pub use crate::utils::{compile_hlsl, fake_sign_dxil_in_place, validate_dxil, HassleError, Result};
pub use crate::wrapper::*;
