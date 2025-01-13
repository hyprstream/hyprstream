use proc_macro_error2::{abort, proc_macro_error};

mod derive_enum;
mod derive_struct;
mod input;

use input::*;

/// Derive macro for arrow fields
#[proc_macro_error]
#[proc_macro_derive(ArrowField, attributes(arrow_field))]
pub fn arrow_convert_derive_field(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    match &ast.data {
        syn::Data::Enum(e) => derive_enum::expand_field(DeriveEnum::from_ast(&ast, e)).into(),
        syn::Data::Struct(s) => derive_struct::expand_field(DeriveStruct::from_ast(&ast, s)).into(),
        _ => {
            abort!(ast.ident.span(), "Only structs and enums supported");
        }
    }
}

/// Derive macro for arrow serialize
#[proc_macro_error]
#[proc_macro_derive(ArrowSerialize, attributes(arrow_field))]
pub fn arrow_convert_derive_serialize(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    match &ast.data {
        syn::Data::Enum(e) => derive_enum::expand_serialize(DeriveEnum::from_ast(&ast, e)).into(),
        syn::Data::Struct(s) => derive_struct::expand_serialize(DeriveStruct::from_ast(&ast, s)).into(),
        _ => {
            abort!(ast.ident.span(), "Only structs and enums supported");
        }
    }
}

/// Derive macro for arrow deserialize
#[proc_macro_error]
#[proc_macro_derive(ArrowDeserialize, attributes(arrow_field))]
pub fn arrow_convert_derive_deserialize(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    match &ast.data {
        syn::Data::Enum(e) => derive_enum::expand_deserialize(DeriveEnum::from_ast(&ast, e)).into(),
        syn::Data::Struct(s) => derive_struct::expand_deserialize(DeriveStruct::from_ast(&ast, s)).into(),
        _ => {
            abort!(ast.ident.span(), "Only structs and enums supported");
        }
    }
}
