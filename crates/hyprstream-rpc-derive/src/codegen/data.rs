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

    // Known service names that have their own `_client` module.
    // Types imported from one of these are emitted as `pub type Foo = super::{origin}_client::Foo;`
    // instead of full struct + trait impls, avoiding duplicate definitions.
    const SERVICE_MODULES: &[&str] = &["inference", "model", "registry", "policy", "worker", "mcp"];

    // For data-only schemas (no request/response unions), generate ALL structs.
    // For service schemas, only generate types referenced by variants.
    let is_data_only = resolved.raw.request_variants.is_empty();
    let list_struct_types = if is_data_only {
        resolved.raw.structs.iter().map(|s| s.name.clone()).collect()
    } else {
        collect_list_struct_types(resolved.raw)
    };
    for type_name in &list_struct_types {
        if let Some(s) = resolved.find_struct(type_name) {
            // Skip Option* wrapper structs — they become Option<T> fields, not standalone types
            if s.option_inner_type().is_some() {
                continue;
            }

            // If this type originates from a different service's schema, emit a `pub type` alias
            // instead of a full struct definition. This avoids duplicate types and the need for
            // hand-written `From` impls between structurally identical generated types.
            if let Some(origin) = s.origin_file.as_deref() {
                if origin != service_name && SERVICE_MODULES.contains(&origin) {
                    let data_name = format_ident!("{}", type_name);
                    let origin_mod = format_ident!("{}_client", origin);
                    tokens.extend(quote! {
                        pub type #data_name = super::#origin_mod::#data_name;
                    });
                    continue;
                }

                // Types originating from RPC infrastructure schemas (e.g. streaming.capnp)
                // alias to the canonical type in hyprstream_rpc instead of generating duplicates.
                const RPC_CRATE_MODULES: &[(&str, &str)] = &[
                    ("streaming", "hyprstream_rpc::streaming"),
                    ("common", "hyprstream_rpc::common_types"),
                ];
                if origin != service_name {
                    if let Some((_, rpc_mod)) = RPC_CRATE_MODULES.iter().find(|(name, _)| *name == origin) {
                        let data_name = format_ident!("{}", type_name);
                        #[allow(clippy::expect_used)] // compile-time constant path
                        let rpc_path: syn::Path = syn::parse_str(rpc_mod).expect("invalid RPC_CRATE_MODULES path");
                        tokens.extend(quote! {
                            pub type #data_name = #rpc_path::#data_name;
                        });
                        continue;
                    }
                }
            }

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
    let enum_name = format_ident!("{}", e.name);
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
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
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
        .non_union_fields()
        .map(|field| {
            let rust_name = resolved.name(&field.name).snake_ident.clone();
            // Check if field type is an Option* wrapper struct (e.g. OptionFloat32 → Option<f32>)
            let option_inner = resolved.find_struct(&field.type_name)
                .and_then(|s| s.option_inner_type())
                .map(|inner_name| rust_type_tokens(&resolved.resolve_type(inner_name).rust_owned));

            let inner_type = if field.type_name == "Data" && field.fixed_size.is_some() {
                let n = field.fixed_size.unwrap_or(0);
                // Arrays > 32 don't implement Default/Serialize/Deserialize via derive,
                // so use Vec<u8> in the struct. ToCapnp/FromCapnp handle length validation.
                if n <= 32 {
                    let n_usize = n as usize;
                    let ts: TokenStream = format!("[u8; {n_usize}]").parse().unwrap_or_else(|_| quote! { Vec<u8> });
                    ts
                } else {
                    quote! { Vec<u8> }
                }
            } else {
                rust_type_tokens(&resolved.resolve_type(&field.type_name).rust_owned)
            };
            let is_option = field.optional || option_inner.is_some();
            let rust_type = if let Some(ref inner) = option_inner {
                quote! { Option<#inner> }
            } else if field.optional {
                quote! { Option<#inner_type> }
            } else {
                inner_type
            };
            let serde_attr = if let Some(ref rename) = field.serde_rename {
                if is_option {
                    quote! { #[serde(rename = #rename, skip_serializing_if = "Option::is_none", default)] }
                } else {
                    quote! { #[serde(rename = #rename)] }
                }
            } else if is_option {
                quote! { #[serde(skip_serializing_if = "Option::is_none", default)] }
            } else {
                TokenStream::new()
            };
            quote! { #serde_attr pub #rust_name: #rust_type }
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

    let field_setters: Vec<TokenStream> = s.non_union_fields().map(|field| {
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
    // Check for Option* wrapper struct field
    if let Some(inner_type_name) = resolved.find_struct(&field.type_name)
        .and_then(|s| s.option_inner_type())
    {
        let rust_name = resolved.name(&field.name).snake_ident.clone();
        let field_snake = &resolved.name(&field.name).snake;
        let init_name = format_ident!("init_{}", field_snake);
        let ct = resolved.resolve_type(inner_type_name).capnp_type.clone();
        let set_val = match ct {
            CapnpType::Bool | CapnpType::UInt8 | CapnpType::UInt16 | CapnpType::UInt32 | CapnpType::UInt64
            | CapnpType::Int8 | CapnpType::Int16 | CapnpType::Int32 | CapnpType::Int64
            | CapnpType::Float32 | CapnpType::Float64 => {
                quote! { opt.set_some(*__v); }
            }
            CapnpType::Text => {
                quote! { opt.set_some(__v.as_str()); }
            }
            CapnpType::Data => {
                quote! { opt.set_some(__v.as_slice()); }
            }
            _ => quote! { let _ = opt; }, // fallback: leave as none
        };
        return quote! {
            if let Some(ref __v) = self.#rust_name {
                let mut opt = builder.reborrow().#init_name();
                #set_val
            }
        };
    }

    let inner = generate_data_field_setter_inner(field, resolved, service_name, types_crate);
    if field.optional {
        let rust_name = resolved.name(&field.name).snake_ident.clone();
        // For optional fields, unwrap `self.field` into `__val` and use that in the setter.
        // The inner setter is generated with `__val` as the value source.
        quote! { if let Some(ref __val) = self.#rust_name { #inner } }
    } else {
        inner
    }
}

fn generate_data_field_setter_inner(
    field: &FieldDef,
    resolved: &ResolvedSchema,
    service_name: &str,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let rust_name = resolved.name(&field.name).snake_ident.clone();
    let field_snake = &resolved.name(&field.name).snake;
    let setter_name = format_ident!("set_{}", field_snake);
    let ct = resolved.resolve_type(&field.type_name).capnp_type.clone();

    // For optional fields, `__val` is `&T` (from `ref __val` in the if-let).
    // For non-optional, use `self.field` directly.
    let (val_owned, val_borrowed) = if field.optional {
        // __val is &T, so *__val gives T (for Copy), &**__val or __val.as_str() for refs
        (quote! { *__val }, quote! { __val })
    } else {
        (quote! { self.#rust_name }, quote! { &self.#rust_name })
    };

    match ct {
        CapnpType::Text | CapnpType::Data => {
            quote! { builder.#setter_name(#val_borrowed); }
        }
        CapnpType::Bool | CapnpType::UInt8 | CapnpType::UInt16 | CapnpType::UInt32 | CapnpType::UInt64
        | CapnpType::Int8 | CapnpType::Int16 | CapnpType::Int32 | CapnpType::Int64
        | CapnpType::Float32 | CapnpType::Float64 => {
            quote! { builder.#setter_name(#val_owned); }
        }
        CapnpType::Enum(ref name) => {
            if let Some(e) = resolved.find_enum(name) {
                let field_capnp_mod = resolve_capnp_mod(
                    lookup_origin_file(name, resolved),
                    service_name,
                    types_crate,
                );
                let enum_rust_name = format_ident!("{}", name);
                let type_ident = format_ident!("{}", name);
                // For optional fields, __val is &EnumType, so dereference with *
                let match_val = if field.optional {
                    quote! { *__val }
                } else {
                    quote! { self.#rust_name }
                };
                let match_arms: Vec<TokenStream> = e.variants.iter().map(|(vname, _)| {
                    let v_pascal = resolved.name(vname).pascal_ident.clone();
                    quote! { #enum_rust_name::#v_pascal => #field_capnp_mod::#type_ident::#v_pascal }
                }).collect();
                quote! {
                    builder.#setter_name(match #match_val {
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
                hyprstream_rpc::capnp::ToCapnp::write_to(#val_borrowed, &mut builder.reborrow().#init_name());
            }
        }
        CapnpType::ListText => {
            let init_name = format_ident!("init_{}", field_snake);
            quote! {
                {
                    let __list_val = #val_borrowed;
                    let mut list = builder.reborrow().#init_name(__list_val.len() as u32);
                    for (i, item) in __list_val.iter().enumerate() {
                        list.set(i as u32, item.as_str());
                    }
                }
            }
        }
        CapnpType::ListData => {
            let init_name = format_ident!("init_{}", field_snake);
            quote! {
                {
                    let __list_val = #val_borrowed;
                    let mut list = builder.reborrow().#init_name(__list_val.len() as u32);
                    for (i, item) in __list_val.iter().enumerate() {
                        list.set(i as u32, item.as_slice());
                    }
                }
            }
        }
        CapnpType::ListPrimitive(ref inner) if inner.is_list() => {
            // Nested list: List(List(Float32)) → Vec<Vec<f32>>
            let init_name = format_ident!("init_{}", field_snake);
            quote! {
                {
                    let __list_val = #val_borrowed;
                    let mut outer = builder.reborrow().#init_name(__list_val.len() as u32);
                    for (i, inner_vec) in __list_val.iter().enumerate() {
                        let mut inner_list = outer.reborrow().init(i as u32, inner_vec.len() as u32);
                        for (j, val) in inner_vec.iter().enumerate() {
                            inner_list.set(j as u32, *val);
                        }
                    }
                }
            }
        }
        CapnpType::ListPrimitive(_) => {
            let init_name = format_ident!("init_{}", field_snake);
            quote! {
                {
                    let __list_val = #val_borrowed;
                    let mut list = builder.reborrow().#init_name(__list_val.len() as u32);
                    for (i, item) in __list_val.iter().enumerate() {
                        list.set(i as u32, *item);
                    }
                }
            }
        }
        CapnpType::ListStruct(_) => {
            let init_name = format_ident!("init_{}", field_snake);
            quote! {
                {
                    let __list_val = #val_borrowed;
                    let mut list = builder.reborrow().#init_name(__list_val.len() as u32);
                    for (i, item) in __list_val.iter().enumerate() {
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

    let field_readers: Vec<TokenStream> = s.non_union_fields().map(|field| {
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
    let inner = generate_data_field_reader_inner(field, resolved, service_name, types_crate);
    if field.optional {
        // Wrap in Option: non-default → Some, default → None.
        // Only Text and List types use the $optional sentinel convention.
        // Numeric/float/enum types use Option* union structs (handled in generate_data_field_reader_inner).
        let rust_name = resolved.name(&field.name).snake_ident.clone();
        let getter_name = format_ident!("get_{}", resolved.name(&field.name).snake);
        let ct = resolved.resolve_type(&field.type_name).capnp_type.clone();
        match ct {
            CapnpType::Text => {
                quote! { #rust_name: { let v = reader.#getter_name()?.to_str()?; if v.is_empty() { None } else { Some(v.to_string()) } }, }
            }
            CapnpType::Data => {
                if let Some(n) = field.fixed_size {
                    let n_usize = n as usize;
                    if n <= 32 {
                        quote! { #rust_name: {
                            let v = reader.#getter_name()?;
                            if v.is_empty() { None } else {
                                let mut arr = [0u8; #n_usize];
                                arr.copy_from_slice(&v[..#n_usize]);
                                Some(arr)
                            }
                        }, }
                    } else {
                        quote! { #rust_name: {
                            let v = reader.#getter_name()?;
                            if v.is_empty() { None } else { Some(v.to_vec()) }
                        }, }
                    }
                } else {
                    quote! { #rust_name: { let v = reader.#getter_name()?; if v.is_empty() { None } else { Some(v.to_vec()) } }, }
                }
            }
            CapnpType::ListText => {
                quote! {
                    #rust_name: {
                        let list = reader.#getter_name()?;
                        if list.len() == 0 {
                            None
                        } else {
                            let mut result = Vec::with_capacity(list.len() as usize);
                            for i in 0..list.len() {
                                result.push(list.get(i)?.to_str()?.to_string());
                            }
                            Some(result)
                        }
                    },
                }
            }
            CapnpType::ListData => {
                quote! {
                    #rust_name: {
                        let list = reader.#getter_name()?;
                        if list.len() == 0 {
                            None
                        } else {
                            let mut result = Vec::with_capacity(list.len() as usize);
                            for i in 0..list.len() {
                                result.push(list.get(i)?.to_vec());
                            }
                            Some(result)
                        }
                    },
                }
            }
            CapnpType::Bool => {
                // Bool $optional: false = None, true = Some(true)
                // (Cap'n Proto default for Bool is false, so absent field = false)
                let getter_name = format_ident!("get_{}", resolved.name(&field.name).snake);
                quote! { #rust_name: { let v = reader.#getter_name(); if v { Some(v) } else { None } }, }
            }
            CapnpType::Float32 | CapnpType::Float64 => {
                // Float $optional: 0.0 = None, non-zero = Some(val)
                let getter_name = format_ident!("get_{}", resolved.name(&field.name).snake);
                quote! { #rust_name: { let v = reader.#getter_name(); if v != 0.0 { Some(v) } else { None } }, }
            }
            CapnpType::UInt8 | CapnpType::UInt16 | CapnpType::UInt32 | CapnpType::UInt64
            | CapnpType::Int8 | CapnpType::Int16 | CapnpType::Int32 | CapnpType::Int64 => {
                // Integer $optional: 0 = None, non-zero = Some(val)
                let getter_name = format_ident!("get_{}", resolved.name(&field.name).snake);
                quote! { #rust_name: { let v = reader.#getter_name(); if v != 0 { Some(v) } else { None } }, }
            }
            CapnpType::Struct(_) | CapnpType::Unknown(_) => {
                // Struct $optional: read and wrap in Some unconditionally.
                // Cap'n Proto struct fields always have a reader, so absence is represented as Default.
                quote! { #rust_name: Some(hyprstream_rpc::capnp::FromCapnp::read_from(reader.#getter_name()?)?), }
            }
            _ => inner, // fall through
        }
    } else {
        inner
    }
}

fn generate_data_field_reader_inner(
    field: &FieldDef,
    resolved: &ResolvedSchema,
    service_name: &str,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let rust_name = resolved.name(&field.name).snake_ident.clone();
    let getter_name = format_ident!("get_{}", resolved.name(&field.name).snake);
    let ct = resolved.resolve_type(&field.type_name).capnp_type.clone();

    // Check for Option* wrapper struct field
    if let Some(inner_type_name) = resolved.find_struct(&field.type_name)
        .and_then(|s| s.option_inner_type())
    {
        let capnp_opt_mod = resolve_capnp_mod(
            resolved.find_struct(&field.type_name).and_then(|s| s.origin_file.as_deref()),
            service_name,
            types_crate,
        );
        let type_mod = format_ident!("{}", to_capnp_module_name(&field.type_name));
        let inner_ct = resolved.resolve_type(inner_type_name).capnp_type.clone();
        // For Text inner type, getter returns Result<text::Reader>
        let some_expr = match inner_ct {
            CapnpType::Text => quote! { v.to_str()?.to_string() },
            CapnpType::Enum(ref enum_name) => {
                // Convert capnp enum value to Rust enum
                if let Some(e) = resolved.find_enum(enum_name) {
                    let enum_type_ident = format_ident!("{}", enum_name);
                    let rust_enum_ident = format_ident!("{}", enum_name);
                    let field_capnp_mod = resolve_capnp_mod(
                        lookup_origin_file(enum_name, resolved),
                        service_name,
                        types_crate,
                    );
                    let match_arms: Vec<TokenStream> = e.variants.iter().map(|(vname, _)| {
                        let v_pascal = resolved.name(vname).pascal_ident.clone();
                        quote! { #field_capnp_mod::#enum_type_ident::#v_pascal => #rust_enum_ident::#v_pascal }
                    }).collect();
                    // v is Result<CapnpEnum, NotInSchema> — unwrap with `?` before matching
                    quote! { match v? { #(#match_arms,)* _ => Default::default() } }
                } else {
                    quote! { Default::default() }
                }
            },
            _ => quote! { v },
        };
        return quote! {
            #rust_name: {
                let opt = reader.#getter_name()?;
                match opt.which()? {
                    #capnp_opt_mod::#type_mod::Which::None(()) => None,
                    #capnp_opt_mod::#type_mod::Which::Some(v) => Some(#some_expr),
                }
            },
        };
    }

    match ct {
        CapnpType::Void => quote! { #rust_name: (), },
        CapnpType::Text => quote! { #rust_name: reader.#getter_name()?.to_str()?.to_string(), },
        CapnpType::Data if field.fixed_size.is_some() => {
            let n = field.fixed_size.unwrap_or(0);
            let n_usize = n as usize;
            let n_lit = proc_macro2::Literal::usize_unsuffixed(n_usize);
            let field_name_str = field.name.as_str();
            if n <= 32 {
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
            } else {
                // N > 32: struct uses Vec<u8>, validate length
                quote! {
                    #rust_name: {
                        let data = reader.#getter_name()?;
                        if data.len() != #n_lit {
                            anyhow::bail!(
                                "{}: expected {} bytes, got {}",
                                #field_name_str, #n_lit, data.len()
                            );
                        }
                        data.to_vec()
                    },
                }
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
                let enum_rust_name = format_ident!("{}", name);
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
        CapnpType::ListPrimitive(ref inner) if inner.is_list() => {
            // Nested list: List(List(Float32)) → Vec<Vec<f32>>
            quote! {
                #rust_name: {
                    let outer_list = reader.#getter_name()?;
                    let mut result = Vec::with_capacity(outer_list.len() as usize);
                    for i in 0..outer_list.len() {
                        let inner_list = outer_list.get(i)?;
                        result.push(inner_list.iter().collect());
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
