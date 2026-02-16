//! Data struct and enum type generation with ToCapnp/FromCapnp trait impls.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::resolve::ResolvedSchema;
use crate::schema::collect_list_struct_types;
use crate::schema::types::*;
use crate::util::*;

/// Resolve the capnp module path for a type based on its origin file.
fn resolve_capnp_mod(
    origin_file: Option<&str>,
    service_name: &str,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let mod_name = origin_file.unwrap_or(service_name);
    let capnp_mod_ident = format_ident!("{}_capnp", mod_name);
    match types_crate {
        Some(tc) => quote! { #tc::#capnp_mod_ident },
        None => quote! { crate::#capnp_mod_ident },
    }
}

/// Generate Rust data structs with ToCapnp/FromCapnp impls, and enum types.
pub fn generate_data_structs(
    resolved: &ResolvedSchema,
    service_name: &str,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let mut tokens = TokenStream::new();

    // Generate enum types
    for e in &resolved.raw.enums {
        tokens.extend(generate_enum_type(e, resolved));
    }

    // Generate data structs with trait impls
    let list_struct_types = collect_list_struct_types(resolved.raw);
    for type_name in &list_struct_types {
        if let Some(s) = resolved.find_struct(type_name) {
            let capnp_mod = resolve_capnp_mod(
                s.origin_file.as_deref(),
                service_name,
                types_crate,
            );
            tokens.extend(generate_data_struct(type_name, s, resolved));
            tokens.extend(generate_to_capnp_impl(type_name, s, resolved, &capnp_mod, service_name, types_crate));
            tokens.extend(generate_from_capnp_impl(type_name, s, resolved, &capnp_mod, service_name, types_crate));
        }
    }

    tokens
}

fn generate_enum_type(e: &EnumDef, resolved: &ResolvedSchema) -> TokenStream {
    let enum_name = format_ident!("{}Enum", e.name);
    let doc = format!("Generated from Cap'n Proto enum {}", e.name);

    let variants: Vec<TokenStream> = e
        .variants
        .iter()
        .enumerate()
        .map(|(i, (variant_name, _ordinal))| {
            let variant_ident = resolved.name(variant_name).pascal_ident.clone();
            if i == 0 {
                quote! { #[default] #variant_ident }
            } else {
                quote! { #variant_ident }
            }
        })
        .collect();

    quote! {
        #[doc = #doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
        pub enum #enum_name {
            #(#variants,)*
        }
    }
}

fn generate_data_struct(
    type_name: &str,
    s: &StructDef,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let data_name = format_ident!("{}", type_name);
    let doc = format!("Generated from Cap'n Proto struct {type_name}");

    let fields: Vec<TokenStream> = s
        .fields
        .iter()
        .map(|field| {
            let rust_name = resolved.name(&field.name).snake_ident.clone();
            let rust_type = if field.type_name == "Data" && field.fixed_size.is_some() {
                let n = field.fixed_size.unwrap_or(0) as usize;
                let ts: TokenStream = format!("[u8; {n}]").parse().unwrap_or_else(|_| quote! { Vec<u8> });
                ts
            } else {
                rust_type_tokens(&resolved.resolve_type(&field.type_name).rust_owned)
            };
            quote! { pub #rust_name: #rust_type }
        })
        .collect();

    quote! {
        #[doc = #doc]
        #[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
        pub struct #data_name {
            #(#fields,)*
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ToCapnp impl generation
// ─────────────────────────────────────────────────────────────────────────────

fn generate_to_capnp_impl(
    type_name: &str,
    s: &StructDef,
    resolved: &ResolvedSchema,
    capnp_mod: &TokenStream,
    service_name: &str,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let data_name = format_ident!("{}", type_name);
    let capnp_struct = format_ident!("{}", to_capnp_module_name(type_name));

    let field_setters: Vec<TokenStream> = s.fields.iter().map(|field| {
        generate_data_field_setter(field, resolved, service_name, types_crate)
    }).collect();

    quote! {
        impl hyprstream_rpc::capnp::ToCapnp for #data_name {
            type Builder<'a> = #capnp_mod::#capnp_struct::Builder<'a>;

            #[allow(unused_variables, unused_mut)]
            fn write_to(&self, builder: &mut Self::Builder<'_>) {
                #(#field_setters)*
            }
        }
    }
}

/// Look up a referenced type's origin_file from the structs/enums lists.
fn lookup_origin_file<'a>(
    type_name: &str,
    resolved: &'a ResolvedSchema,
) -> Option<&'a str> {
    if let Some(s) = resolved.find_struct(type_name) {
        return s.origin_file.as_deref();
    }
    if let Some(e) = resolved.find_enum(type_name) {
        return e.origin_file.as_deref();
    }
    None
}

fn generate_data_field_setter(
    field: &FieldDef,
    resolved: &ResolvedSchema,
    service_name: &str,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let rust_name = resolved.name(&field.name).snake_ident.clone();
    let field_snake = &resolved.name(&field.name).snake;
    let setter_name = format_ident!("set_{}", field_snake);
    let ct = resolved.resolve_type(&field.type_name).capnp_type.clone();

    match ct {
        CapnpType::Text | CapnpType::Data => {
            quote! { builder.#setter_name(&self.#rust_name); }
        }
        CapnpType::Bool | CapnpType::UInt8 | CapnpType::UInt16 | CapnpType::UInt32 | CapnpType::UInt64
        | CapnpType::Int8 | CapnpType::Int16 | CapnpType::Int32 | CapnpType::Int64
        | CapnpType::Float32 | CapnpType::Float64 => {
            quote! { builder.#setter_name(self.#rust_name); }
        }
        CapnpType::Enum(ref name) => {
            if let Some(e) = resolved.find_enum(name) {
                let field_capnp_mod = resolve_capnp_mod(
                    lookup_origin_file(name, resolved),
                    service_name,
                    types_crate,
                );
                let enum_rust_name = format_ident!("{}Enum", name);
                let type_ident = format_ident!("{}", name);
                let match_arms: Vec<TokenStream> = e.variants.iter().map(|(vname, _)| {
                    let v_pascal = resolved.name(vname).pascal_ident.clone();
                    quote! { #enum_rust_name::#v_pascal => #field_capnp_mod::#type_ident::#v_pascal }
                }).collect();
                quote! {
                    builder.#setter_name(match self.#rust_name {
                        #(#match_arms,)*
                    });
                }
            } else {
                TokenStream::new()
            }
        }
        CapnpType::Struct(_) => {
            let init_name = format_ident!("init_{}", field_snake);
            quote! {
                hyprstream_rpc::capnp::ToCapnp::write_to(&self.#rust_name, &mut builder.reborrow().#init_name());
            }
        }
        CapnpType::ListText => {
            let init_name = format_ident!("init_{}", field_snake);
            quote! {
                {
                    let mut list = builder.reborrow().#init_name(self.#rust_name.len() as u32);
                    for (i, item) in self.#rust_name.iter().enumerate() {
                        list.set(i as u32, item.as_str());
                    }
                }
            }
        }
        CapnpType::ListData => {
            let init_name = format_ident!("init_{}", field_snake);
            quote! {
                {
                    let mut list = builder.reborrow().#init_name(self.#rust_name.len() as u32);
                    for (i, item) in self.#rust_name.iter().enumerate() {
                        list.set(i as u32, item.as_slice());
                    }
                }
            }
        }
        CapnpType::ListPrimitive(_) => {
            let init_name = format_ident!("init_{}", field_snake);
            quote! {
                {
                    let mut list = builder.reborrow().#init_name(self.#rust_name.len() as u32);
                    for (i, item) in self.#rust_name.iter().enumerate() {
                        list.set(i as u32, *item);
                    }
                }
            }
        }
        CapnpType::ListStruct(_) => {
            let init_name = format_ident!("init_{}", field_snake);
            quote! {
                {
                    let mut list = builder.reborrow().#init_name(self.#rust_name.len() as u32);
                    for (i, item) in self.#rust_name.iter().enumerate() {
                        hyprstream_rpc::capnp::ToCapnp::write_to(item, &mut list.reborrow().get(i as u32));
                    }
                }
            }
        }
        _ => TokenStream::new(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FromCapnp impl generation
// ─────────────────────────────────────────────────────────────────────────────

fn generate_from_capnp_impl(
    type_name: &str,
    s: &StructDef,
    resolved: &ResolvedSchema,
    capnp_mod: &TokenStream,
    service_name: &str,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let data_name = format_ident!("{}", type_name);
    let capnp_struct = format_ident!("{}", to_capnp_module_name(type_name));

    let field_readers: Vec<TokenStream> = s.fields.iter().map(|field| {
        generate_data_field_reader(field, resolved, service_name, types_crate)
    }).collect();

    quote! {
        impl hyprstream_rpc::capnp::FromCapnp for #data_name {
            type Reader<'a> = #capnp_mod::#capnp_struct::Reader<'a>;

            #[allow(unused_variables)]
            fn read_from(reader: Self::Reader<'_>) -> anyhow::Result<Self> {
                Ok(Self {
                    #(#field_readers)*
                })
            }
        }
    }
}

fn generate_data_field_reader(
    field: &FieldDef,
    resolved: &ResolvedSchema,
    service_name: &str,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let rust_name = resolved.name(&field.name).snake_ident.clone();
    let getter_name = format_ident!("get_{}", resolved.name(&field.name).snake);
    let ct = resolved.resolve_type(&field.type_name).capnp_type.clone();

    match ct {
        CapnpType::Void => quote! { #rust_name: (), },
        CapnpType::Text => quote! { #rust_name: reader.#getter_name()?.to_str()?.to_string(), },
        CapnpType::Data if field.fixed_size.is_some() => {
            let n = field.fixed_size.unwrap_or(0) as usize;
            let n_lit = proc_macro2::Literal::usize_unsuffixed(n);
            let field_name_str = field.name.as_str();
            quote! {
                #rust_name: {
                    let data = reader.#getter_name()?;
                    if data.len() != #n_lit {
                        anyhow::bail!(
                            "{}: expected {} bytes, got {}",
                            #field_name_str, #n_lit, data.len()
                        );
                    }
                    let mut arr = [0u8; #n_lit];
                    arr.copy_from_slice(data);
                    arr
                },
            }
        }
        CapnpType::Data => quote! { #rust_name: reader.#getter_name()?.to_vec(), },
        CapnpType::Bool | CapnpType::UInt8 | CapnpType::UInt16 | CapnpType::UInt32 | CapnpType::UInt64
        | CapnpType::Int8 | CapnpType::Int16 | CapnpType::Int32 | CapnpType::Int64
        | CapnpType::Float32 | CapnpType::Float64 => {
            quote! { #rust_name: reader.#getter_name(), }
        }
        CapnpType::Enum(ref name) => {
            if let Some(e) = resolved.find_enum(name) {
                let field_capnp_mod = resolve_capnp_mod(
                    lookup_origin_file(name, resolved),
                    service_name,
                    types_crate,
                );
                let enum_rust_name = format_ident!("{}Enum", name);
                let type_ident = format_ident!("{}", name);
                let match_arms: Vec<TokenStream> = e.variants.iter().map(|(vname, _)| {
                    let v_pascal = resolved.name(vname).pascal_ident.clone();
                    quote! { #field_capnp_mod::#type_ident::#v_pascal => #enum_rust_name::#v_pascal }
                }).collect();
                quote! {
                    #rust_name: match reader.#getter_name()? {
                        #(#match_arms,)*
                        _ => Default::default(),
                    },
                }
            } else {
                quote! { #rust_name: Default::default(), }
            }
        }
        CapnpType::Struct(_) => {
            quote! {
                #rust_name: hyprstream_rpc::capnp::FromCapnp::read_from(reader.#getter_name()?)?,
            }
        }
        CapnpType::ListText => {
            quote! {
                #rust_name: {
                    let list = reader.#getter_name()?;
                    let mut result = Vec::with_capacity(list.len() as usize);
                    for i in 0..list.len() {
                        result.push(list.get(i)?.to_str()?.to_string());
                    }
                    result
                },
            }
        }
        CapnpType::ListData => {
            quote! {
                #rust_name: {
                    let list = reader.#getter_name()?;
                    let mut result = Vec::with_capacity(list.len() as usize);
                    for i in 0..list.len() {
                        result.push(list.get(i)?.to_vec());
                    }
                    result
                },
            }
        }
        CapnpType::ListPrimitive(_) => {
            quote! {
                #rust_name: reader.#getter_name()?.iter().collect(),
            }
        }
        CapnpType::ListStruct(_) => {
            quote! {
                #rust_name: {
                    let list = reader.#getter_name()?;
                    let mut result = Vec::with_capacity(list.len() as usize);
                    for i in 0..list.len() {
                        result.push(hyprstream_rpc::capnp::FromCapnp::read_from(list.get(i))?);
                    }
                    result
                },
            }
        }
        _ => quote! { #rust_name: Default::default(), },
    }
}
