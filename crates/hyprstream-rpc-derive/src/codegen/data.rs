//! Data struct and enum type generation.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::schema::collect_list_struct_types;
use crate::schema::types::*;
use crate::util::*;

/// Generate Rust data structs for structs used in List() and response contexts, and enum types.
pub fn generate_data_structs(schema: &ParsedSchema) -> TokenStream {
    let mut tokens = TokenStream::new();

    // Generate enum types
    for e in &schema.enums {
        tokens.extend(generate_enum_type(e));
    }

    // Generate data structs
    let list_struct_types = collect_list_struct_types(schema);
    for type_name in &list_struct_types {
        if let Some(s) = schema.structs.iter().find(|s| s.name == *type_name) {
            tokens.extend(generate_data_struct(type_name, s, &schema.enums, &schema.structs));
        }
    }

    tokens
}

fn generate_enum_type(e: &EnumDef) -> TokenStream {
    let enum_name = format_ident!("{}Enum", e.name);
    let doc = format!("Generated from Cap'n Proto enum {}", e.name);

    let variants: Vec<TokenStream> = e
        .variants
        .iter()
        .enumerate()
        .map(|(i, (variant_name, _ordinal))| {
            let variant_ident = format_ident!("{}", to_pascal_case(variant_name));
            if i == 0 {
                quote! { #[default] #variant_ident }
            } else {
                quote! { #variant_ident }
            }
        })
        .collect();

    quote! {
        #[doc = #doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
        pub enum #enum_name {
            #(#variants,)*
        }
    }
}

fn generate_data_struct(
    type_name: &str,
    s: &StructDef,
    enums: &[EnumDef],
    structs: &[StructDef],
) -> TokenStream {
    let data_name = format_ident!("{}Data", type_name);
    let doc = format!("Generated from Cap'n Proto struct {type_name}");

    let fields: Vec<TokenStream> = s
        .fields
        .iter()
        .map(|field| {
            let rust_name = format_ident!("{}", to_snake_case(&field.name));
            let rust_type =
                rust_type_tokens(&CapnpType::classify(&field.type_name, structs, enums).rust_owned_type());
            quote! { pub #rust_name: #rust_type }
        })
        .collect();

    quote! {
        #[doc = #doc]
        #[derive(Debug, Clone, serde::Serialize)]
        pub struct #data_name {
            #(#fields,)*
        }
    }
}
