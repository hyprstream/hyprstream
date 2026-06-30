//! Data struct and enum type generation with ToCapnp/FromCapnp trait impls.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::resolve::ResolvedSchema;
use crate::schema::types::collect_list_struct_types;
use crate::schema::types::*;
use crate::util::*;

/// Resolve a field-level `$domainType("path::Type")` annotation to its Rust type tokens.
///
/// Unlike the struct-level `resolve_domain_type` (which prepends the generated crate prefix
/// for return-type wrappers), field newtypes live in shared crates and are referenced by their
/// absolute path verbatim (e.g. `hyprstream_rpc::identity::Did`), so the path resolves
/// identically from every generated module regardless of `types_crate`.
fn resolve_field_domain_type(domain_path: &str) -> TokenStream {
    domain_path.parse().unwrap_or_else(|_| quote! { String })
}

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
        // Synthetic capnp group nodes are named `Parent.armName` (e.g.
        // `Ordering.unordered`). They are not standalone Rust types — their leaf
        // fields are folded into the parent union's enum variant — and the dotted
        // name is not a valid Rust identifier, so skip them here.
        if type_name.contains('.') {
            continue;
        }
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
                    ("streaming", "hyprstream_rpc::stream_info"),
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

            // Pure tagged-union structs (a capnp `union { ... }` with no sibling
            // non-union fields) become native Rust enums — one variant per arm —
            // instead of an (empty) struct. Option<T> wrappers are handled above.
            if s.is_pure_union() && s.option_inner_type().is_none() {
                tokens.extend(generate_union_enum(type_name, s, resolved));
                tokens.extend(generate_union_to_capnp_impl(type_name, s, resolved, &capnp_mod, service_name, types_crate));
                tokens.extend(generate_union_from_capnp_impl(type_name, s, resolved, &capnp_mod, service_name, types_crate));
                continue;
            }

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

            let inner_type = if let Some(dt) = field.domain_type.as_deref() {
                // Field-level `$domainType`: use the newtype (wire type unchanged).
                resolve_field_domain_type(dt)
            } else if field.type_name == "Data" && field.fixed_size.is_some() {
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

    // Mixed struct (non-union fields + an anonymous union): fold the union into
    // a companion enum `{Type}Content` carried as a `content` field so the union
    // arms round-trip. Pure unions are handled by `generate_union_enum`; structs
    // with no union skip this entirely.
    let (content_field, content_enum) = if s.has_union && !s.is_pure_union() {
        let content_enum_name = format_ident!("{}Content", type_name);
        let enum_def = generate_mixed_union_enum(type_name, s, resolved);
        (
            quote! { pub content: #content_enum_name, },
            enum_def,
        )
    } else {
        (TokenStream::new(), TokenStream::new())
    };

    quote! {
        #content_enum
        #[doc = #doc]
        #[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        pub struct #data_name {
            #(#fields,)*
            #content_field
        }
    }
}

/// Generate the companion `{Type}Content` enum for a mixed struct's anonymous
/// union. Mirrors `generate_union_enum` but emits a distinctly-named enum so it
/// can be carried as a field of the owning struct.
fn generate_mixed_union_enum(
    type_name: &str,
    s: &StructDef,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let enum_name = format_ident!("{}Content", type_name);
    let doc = format!("Anonymous union of Cap'n Proto struct {type_name}");

    let default_arm = s.union_arms.iter().find(|a| a.discriminant_value == 0);
    let default_is_unit = matches!(default_arm.map(|a| &a.payload), Some(ArmPayload::Void));

    let variants: Vec<TokenStream> = s
        .union_arms
        .iter()
        .map(|arm| {
            let variant_ident = resolved.name(&arm.name).pascal_ident.clone();
            let default_attr = if arm.discriminant_value == 0 && default_is_unit {
                quote! { #[default] }
            } else {
                TokenStream::new()
            };
            match &arm.payload {
                ArmPayload::Void => quote! { #default_attr #variant_ident },
                ArmPayload::Type(t) => {
                    let ty = arm_payload_type_tokens(t, resolved);
                    quote! { #default_attr #variant_ident(#ty) }
                }
                ArmPayload::Group(leaves) => {
                    let fields: Vec<TokenStream> = leaves
                        .iter()
                        .map(|leaf| {
                            let leaf_name = resolved.name(&leaf.name).snake_ident.clone();
                            let leaf_ty = arm_payload_type_tokens(&leaf.type_name, resolved);
                            quote! { #leaf_name: #leaf_ty }
                        })
                        .collect();
                    quote! { #default_attr #variant_ident { #(#fields,)* } }
                }
            }
        })
        .collect();

    let (derive_default, manual_default) = if default_is_unit {
        (quote! { Default, }, TokenStream::new())
    } else {
        let manual = default_arm.map(|arm| {
            let variant_ident = resolved.name(&arm.name).pascal_ident.clone();
            let ctor = match &arm.payload {
                ArmPayload::Void => quote! { Self::#variant_ident },
                ArmPayload::Type(_) => quote! { Self::#variant_ident(Default::default()) },
                ArmPayload::Group(leaves) => {
                    let fields: Vec<TokenStream> = leaves.iter().map(|leaf| {
                        let leaf_name = resolved.name(&leaf.name).snake_ident.clone();
                        quote! { #leaf_name: Default::default() }
                    }).collect();
                    quote! { Self::#variant_ident { #(#fields,)* } }
                }
            };
            quote! {
                impl Default for #enum_name {
                    fn default() -> Self { #ctor }
                }
            }
        }).unwrap_or_default();
        (TokenStream::new(), manual)
    };

    quote! {
        #[doc = #doc]
        #[derive(Debug, Clone, #derive_default PartialEq, serde::Serialize, serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        pub enum #enum_name {
            #(#variants,)*
        }
        #manual_default
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tagged-union (pure-union struct) → Rust enum generation
// ─────────────────────────────────────────────────────────────────────────────

/// Owned Rust type tokens for a single-type union-arm payload.
fn arm_payload_type_tokens(type_name: &str, resolved: &ResolvedSchema) -> TokenStream {
    rust_type_tokens(&resolved.resolve_type(type_name).rust_owned)
}

/// Generate a Rust `enum` for a pure-union capnp struct.
///
/// Each arm maps to a variant: Void → unit, single-type → tuple, group → struct.
/// The discriminant-0 arm carries `#[default]`.
fn generate_union_enum(
    type_name: &str,
    s: &StructDef,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let enum_name = format_ident!("{}", type_name);
    let doc = format!("Generated from Cap'n Proto union {type_name}");

    // The discriminant-0 arm is the capnp zero-fill default. If it's a Void arm
    // we can use `#[derive(Default)]` + `#[default]` (requires a unit variant);
    // otherwise we drop `Default` from the derive and emit a manual `impl Default`
    // that builds the @0 variant with default-valued payload.
    let default_arm = s.union_arms.iter().find(|a| a.discriminant_value == 0);
    let default_is_unit = matches!(default_arm.map(|a| &a.payload), Some(ArmPayload::Void));

    let variants: Vec<TokenStream> = s
        .union_arms
        .iter()
        .map(|arm| {
            let variant_ident = resolved.name(&arm.name).pascal_ident.clone();
            let default_attr = if arm.discriminant_value == 0 && default_is_unit {
                quote! { #[default] }
            } else {
                TokenStream::new()
            };
            match &arm.payload {
                ArmPayload::Void => quote! { #default_attr #variant_ident },
                ArmPayload::Type(t) => {
                    let ty = arm_payload_type_tokens(t, resolved);
                    quote! { #default_attr #variant_ident(#ty) }
                }
                ArmPayload::Group(leaves) => {
                    let fields: Vec<TokenStream> = leaves
                        .iter()
                        .map(|leaf| {
                            let leaf_name = resolved.name(&leaf.name).snake_ident.clone();
                            let leaf_ty = arm_payload_type_tokens(&leaf.type_name, resolved);
                            quote! { #leaf_name: #leaf_ty }
                        })
                        .collect();
                    quote! { #default_attr #variant_ident { #(#fields,)* } }
                }
            }
        })
        .collect();

    let (derive_default, manual_default) = if default_is_unit {
        (quote! { Default, }, TokenStream::new())
    } else {
        // Manual Default for a data-carrying @0 arm.
        let manual = default_arm.map(|arm| {
            let variant_ident = resolved.name(&arm.name).pascal_ident.clone();
            let ctor = match &arm.payload {
                ArmPayload::Void => quote! { Self::#variant_ident },
                ArmPayload::Type(_) => quote! { Self::#variant_ident(Default::default()) },
                ArmPayload::Group(leaves) => {
                    let fields: Vec<TokenStream> = leaves.iter().map(|leaf| {
                        let leaf_name = resolved.name(&leaf.name).snake_ident.clone();
                        quote! { #leaf_name: Default::default() }
                    }).collect();
                    quote! { Self::#variant_ident { #(#fields,)* } }
                }
            };
            quote! {
                impl Default for #enum_name {
                    fn default() -> Self { #ctor }
                }
            }
        }).unwrap_or_default();
        (TokenStream::new(), manual)
    };

    quote! {
        #[doc = #doc]
        #[derive(Debug, Clone, #derive_default PartialEq, serde::Serialize, serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        pub enum #enum_name {
            #(#variants,)*
        }
        #manual_default
    }
}

/// Generate the value-setter for a scalar/Text/Data/enum/struct union-arm payload.
///
/// `value` is a binding (e.g. `v`) holding `&T` for the arm's owned Rust type.
/// `setter`/`init` are the capnp builder method idents for this arm.
fn union_arm_value_setter(
    type_name: &str,
    value: &TokenStream,
    setter: &proc_macro2::Ident,
    init: &proc_macro2::Ident,
    resolved: &ResolvedSchema,
    capnp_mod: &TokenStream,
    service_name: &str,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let ct = resolved.resolve_type(type_name).capnp_type.clone();
    match ct {
        CapnpType::Bool | CapnpType::UInt8 | CapnpType::UInt16 | CapnpType::UInt32 | CapnpType::UInt64
        | CapnpType::Int8 | CapnpType::Int16 | CapnpType::Int32 | CapnpType::Int64
        | CapnpType::Float32 | CapnpType::Float64 => {
            quote! { builder.#setter(*#value); }
        }
        CapnpType::Text => quote! { builder.#setter(#value.as_str()); },
        CapnpType::Data => quote! { builder.#setter(#value.as_slice()); },
        CapnpType::Enum(ref name) => {
            if let Some(e) = resolved.find_enum(name) {
                let field_capnp_mod = resolve_capnp_mod(
                    lookup_origin_file(name, resolved),
                    service_name,
                    types_crate,
                );
                let enum_rust = format_ident!("{}", name);
                let type_ident = format_ident!("{}", name);
                let arms: Vec<TokenStream> = e.variants.iter().map(|(vname, _)| {
                    let vp = resolved.name(vname).pascal_ident.clone();
                    quote! { #enum_rust::#vp => #field_capnp_mod::#type_ident::#vp }
                }).collect();
                quote! { builder.#setter(match #value { #(#arms,)* }); }
            } else {
                TokenStream::new()
            }
        }
        CapnpType::Struct(_) => {
            quote! {
                let mut __arm = builder.reborrow().#init();
                hyprstream_rpc::capnp::ToCapnp::write_to(#value, &mut __arm);
            }
        }
        _ => {
            let _ = capnp_mod;
            TokenStream::new()
        }
    }
}

/// Generate `ToCapnp` for a pure-union enum.
fn generate_union_to_capnp_impl(
    type_name: &str,
    s: &StructDef,
    resolved: &ResolvedSchema,
    capnp_mod: &TokenStream,
    service_name: &str,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let enum_name = format_ident!("{}", type_name);
    let capnp_struct = format_ident!("{}", to_capnp_module_name(type_name));

    let arms: Vec<TokenStream> = s.union_arms.iter().map(|arm| {
        let variant = resolved.name(&arm.name).pascal_ident.clone();
        let arm_snake = &resolved.name(&arm.name).snake;
        let setter = format_ident!("set_{}", arm_snake);
        let init = format_ident!("init_{}", arm_snake);
        match &arm.payload {
            ArmPayload::Void => {
                quote! { Self::#variant => builder.#setter(()), }
            }
            ArmPayload::Type(t) => {
                let value = quote! { v };
                let set = union_arm_value_setter(t, &value, &setter, &init, resolved, capnp_mod, service_name, types_crate);
                quote! { Self::#variant(v) => { #set } }
            }
            ArmPayload::Group(leaves) => {
                let leaf_binds: Vec<TokenStream> = leaves.iter().map(|leaf| {
                    let leaf_name = resolved.name(&leaf.name).snake_ident.clone();
                    quote! { #leaf_name }
                }).collect();
                let leaf_sets: Vec<TokenStream> = leaves.iter().map(|leaf| {
                    let leaf_snake = &resolved.name(&leaf.name).snake;
                    let leaf_name = resolved.name(&leaf.name).snake_ident.clone();
                    let leaf_setter = format_ident!("set_{}", leaf_snake);
                    let ct = resolved.resolve_type(&leaf.type_name).capnp_type.clone();
                    match ct {
                        CapnpType::Text => quote! { __g.#leaf_setter(#leaf_name.as_str()); },
                        CapnpType::Data => quote! { __g.#leaf_setter(#leaf_name.as_slice()); },
                        CapnpType::Struct(_) => quote! {
                            hyprstream_rpc::capnp::ToCapnp::write_to(#leaf_name, &mut __g.reborrow().#leaf_setter());
                        },
                        _ => quote! { __g.#leaf_setter(*#leaf_name); },
                    }
                }).collect();
                quote! {
                    Self::#variant { #(#leaf_binds,)* } => {
                        let mut __g = builder.reborrow().#init();
                        #(#leaf_sets)*
                    }
                }
            }
        }
    }).collect();

    quote! {
        impl hyprstream_rpc::capnp::ToCapnp for #enum_name {
            type Builder<'a> = #capnp_mod::#capnp_struct::Builder<'a>;

            #[allow(unused_variables, unused_mut)]
            fn write_to(&self, builder: &mut Self::Builder<'_>) {
                match self {
                    #(#arms)*
                }
            }
        }
    }
}

/// Generate `FromCapnp` for a pure-union enum.
fn generate_union_from_capnp_impl(
    type_name: &str,
    s: &StructDef,
    resolved: &ResolvedSchema,
    capnp_mod: &TokenStream,
    service_name: &str,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let enum_name = format_ident!("{}", type_name);
    let capnp_struct = format_ident!("{}", to_capnp_module_name(type_name));
    let union_mod = format_ident!("{}", to_capnp_module_name(type_name));

    let arms: Vec<TokenStream> = s.union_arms.iter().map(|arm| {
        let variant = resolved.name(&arm.name).pascal_ident.clone();
        match &arm.payload {
            ArmPayload::Void => {
                quote! { #capnp_mod::#union_mod::Which::#variant(()) => Self::#variant, }
            }
            ArmPayload::Type(t) => {
                let ct = resolved.resolve_type(t).capnp_type.clone();
                match ct {
                    // Scalars bind the value directly; no Result.
                    CapnpType::Bool | CapnpType::UInt8 | CapnpType::UInt16 | CapnpType::UInt32 | CapnpType::UInt64
                    | CapnpType::Int8 | CapnpType::Int16 | CapnpType::Int32 | CapnpType::Int64
                    | CapnpType::Float32 | CapnpType::Float64 => {
                        quote! { #capnp_mod::#union_mod::Which::#variant(v) => Self::#variant(v), }
                    }
                    // Text / Data / Struct bind a Result<Reader> — unwrap with `?`.
                    CapnpType::Text => {
                        quote! { #capnp_mod::#union_mod::Which::#variant(v) => Self::#variant(v?.to_str()?.to_string()), }
                    }
                    CapnpType::Data => {
                        quote! { #capnp_mod::#union_mod::Which::#variant(v) => Self::#variant(v?.to_vec()), }
                    }
                    CapnpType::Enum(ref name) => {
                        if let Some(e) = resolved.find_enum(name) {
                            let field_capnp_mod = resolve_capnp_mod(
                                lookup_origin_file(name, resolved),
                                service_name,
                                types_crate,
                            );
                            let enum_rust = format_ident!("{}", name);
                            let type_ident = format_ident!("{}", name);
                            let match_arms: Vec<TokenStream> = e.variants.iter().map(|(vname, _)| {
                                let vp = resolved.name(vname).pascal_ident.clone();
                                quote! { #field_capnp_mod::#type_ident::#vp => #enum_rust::#vp }
                            }).collect();
                            quote! { #capnp_mod::#union_mod::Which::#variant(v) => Self::#variant(match v? { #(#match_arms,)* }), }
                        } else {
                            quote! { #capnp_mod::#union_mod::Which::#variant(_) => Self::#variant(Default::default()), }
                        }
                    }
                    CapnpType::Struct(_) => {
                        quote! { #capnp_mod::#union_mod::Which::#variant(r) => Self::#variant(hyprstream_rpc::capnp::FromCapnp::read_from(r?)?), }
                    }
                    _ => quote! { #capnp_mod::#union_mod::Which::#variant(_) => Self::#variant(Default::default()), },
                }
            }
            ArmPayload::Group(leaves) => {
                let leaf_reads: Vec<TokenStream> = leaves.iter().map(|leaf| {
                    let leaf_name = resolved.name(&leaf.name).snake_ident.clone();
                    let leaf_getter = format_ident!("get_{}", resolved.name(&leaf.name).snake);
                    let ct = resolved.resolve_type(&leaf.type_name).capnp_type.clone();
                    match ct {
                        CapnpType::Text => quote! { #leaf_name: g.#leaf_getter()?.to_str()?.to_string() },
                        CapnpType::Data => quote! { #leaf_name: g.#leaf_getter()?.to_vec() },
                        CapnpType::Struct(_) => quote! { #leaf_name: hyprstream_rpc::capnp::FromCapnp::read_from(g.#leaf_getter()?)? },
                        _ => quote! { #leaf_name: g.#leaf_getter() },
                    }
                }).collect();
                quote! { #capnp_mod::#union_mod::Which::#variant(g) => Self::#variant { #(#leaf_reads,)* }, }
            }
        }
    }).collect();

    quote! {
        impl hyprstream_rpc::capnp::FromCapnp for #enum_name {
            type Reader<'a> = #capnp_mod::#capnp_struct::Reader<'a>;

            #[allow(unused_variables)]
            fn read_from(reader: Self::Reader<'_>) -> anyhow::Result<Self> {
                Ok(match reader.which()? {
                    #(#arms)*
                })
            }
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

    // Mixed struct: emit the anonymous-union setters by matching on `self.content`.
    let content_setter = if s.has_union && !s.is_pure_union() {
        let content_enum = format_ident!("{}Content", type_name);
        let arms: Vec<TokenStream> = s.union_arms.iter().map(|arm| {
            let variant = resolved.name(&arm.name).pascal_ident.clone();
            let arm_snake = &resolved.name(&arm.name).snake;
            let setter = format_ident!("set_{}", arm_snake);
            let init = format_ident!("init_{}", arm_snake);
            match &arm.payload {
                ArmPayload::Void => quote! { #content_enum::#variant => builder.#setter(()), },
                ArmPayload::Type(t) => {
                    let value = quote! { v };
                    let set = union_arm_value_setter(t, &value, &setter, &init, resolved, capnp_mod, service_name, types_crate);
                    quote! { #content_enum::#variant(v) => { #set } }
                }
                ArmPayload::Group(leaves) => {
                    let leaf_binds: Vec<TokenStream> = leaves.iter().map(|leaf| {
                        let leaf_name = resolved.name(&leaf.name).snake_ident.clone();
                        quote! { #leaf_name }
                    }).collect();
                    let leaf_sets: Vec<TokenStream> = leaves.iter().map(|leaf| {
                        let leaf_snake = &resolved.name(&leaf.name).snake;
                        let leaf_name = resolved.name(&leaf.name).snake_ident.clone();
                        let leaf_setter = format_ident!("set_{}", leaf_snake);
                        let ct = resolved.resolve_type(&leaf.type_name).capnp_type.clone();
                        match ct {
                            CapnpType::Text => quote! { __g.#leaf_setter(#leaf_name.as_str()); },
                            CapnpType::Data => quote! { __g.#leaf_setter(#leaf_name.as_slice()); },
                            CapnpType::Struct(_) => quote! {
                                hyprstream_rpc::capnp::ToCapnp::write_to(#leaf_name, &mut __g.reborrow().#leaf_setter());
                            },
                            _ => quote! { __g.#leaf_setter(*#leaf_name); },
                        }
                    }).collect();
                    quote! {
                        #content_enum::#variant { #(#leaf_binds,)* } => {
                            let mut __g = builder.reborrow().#init();
                            #(#leaf_sets)*
                        }
                    }
                }
            }
        }).collect();
        quote! {
            match &self.content {
                #(#arms)*
            }
        }
    } else {
        TokenStream::new()
    };

    quote! {
        impl hyprstream_rpc::capnp::ToCapnp for #data_name {
            type Builder<'a> = #capnp_mod::#capnp_struct::Builder<'a>;

            #[allow(unused_variables, unused_mut)]
            fn write_to(&self, builder: &mut Self::Builder<'_>) {
                #(#field_setters)*
                #content_setter
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
        // Field-level `$domainType` newtype over `Text`: write via `.as_str()`.
        CapnpType::Text if field.domain_type.is_some() => {
            // Accessor without a leading `&`: optional → `__val` (already `&T`),
            // non-optional → `self.field`; `.as_str()` borrows in both cases.
            let val_accessor = if field.optional {
                quote! { __val }
            } else {
                quote! { self.#rust_name }
            };
            quote! { builder.#setter_name(#val_accessor.as_str()); }
        }
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

    // Mixed struct: read the anonymous union via `which()` into `content`.
    let content_reader = if s.has_union && !s.is_pure_union() {
        let content_enum = format_ident!("{}Content", type_name);
        let union_mod = format_ident!("{}", to_capnp_module_name(type_name));
        let arms: Vec<TokenStream> = s.union_arms.iter().map(|arm| {
            let variant = resolved.name(&arm.name).pascal_ident.clone();
            match &arm.payload {
                ArmPayload::Void => {
                    quote! { #capnp_mod::#union_mod::Which::#variant(()) => #content_enum::#variant, }
                }
                ArmPayload::Type(t) => {
                    let ct = resolved.resolve_type(t).capnp_type.clone();
                    match ct {
                        CapnpType::Bool | CapnpType::UInt8 | CapnpType::UInt16 | CapnpType::UInt32 | CapnpType::UInt64
                        | CapnpType::Int8 | CapnpType::Int16 | CapnpType::Int32 | CapnpType::Int64
                        | CapnpType::Float32 | CapnpType::Float64 => {
                            quote! { #capnp_mod::#union_mod::Which::#variant(v) => #content_enum::#variant(v), }
                        }
                        CapnpType::Text => {
                            quote! { #capnp_mod::#union_mod::Which::#variant(v) => #content_enum::#variant(v?.to_str()?.to_string()), }
                        }
                        CapnpType::Data => {
                            quote! { #capnp_mod::#union_mod::Which::#variant(v) => #content_enum::#variant(v?.to_vec()), }
                        }
                        CapnpType::Enum(ref name) => {
                            if let Some(e) = resolved.find_enum(name) {
                                let field_capnp_mod = resolve_capnp_mod(
                                    lookup_origin_file(name, resolved),
                                    service_name,
                                    types_crate,
                                );
                                let enum_rust = format_ident!("{}", name);
                                let type_ident = format_ident!("{}", name);
                                let match_arms: Vec<TokenStream> = e.variants.iter().map(|(vname, _)| {
                                    let vp = resolved.name(vname).pascal_ident.clone();
                                    quote! { #field_capnp_mod::#type_ident::#vp => #enum_rust::#vp }
                                }).collect();
                                quote! { #capnp_mod::#union_mod::Which::#variant(v) => #content_enum::#variant(match v? { #(#match_arms,)* }), }
                            } else {
                                quote! { #capnp_mod::#union_mod::Which::#variant(_) => #content_enum::#variant(Default::default()), }
                            }
                        }
                        CapnpType::Struct(_) => {
                            quote! { #capnp_mod::#union_mod::Which::#variant(r) => #content_enum::#variant(hyprstream_rpc::capnp::FromCapnp::read_from(r?)?), }
                        }
                        _ => quote! { #capnp_mod::#union_mod::Which::#variant(_) => #content_enum::#variant(Default::default()), },
                    }
                }
                ArmPayload::Group(leaves) => {
                    let leaf_reads: Vec<TokenStream> = leaves.iter().map(|leaf| {
                        let leaf_name = resolved.name(&leaf.name).snake_ident.clone();
                        let leaf_getter = format_ident!("get_{}", resolved.name(&leaf.name).snake);
                        let ct = resolved.resolve_type(&leaf.type_name).capnp_type.clone();
                        match ct {
                            CapnpType::Text => quote! { #leaf_name: g.#leaf_getter()?.to_str()?.to_string() },
                            CapnpType::Data => quote! { #leaf_name: g.#leaf_getter()?.to_vec() },
                            CapnpType::Struct(_) => quote! { #leaf_name: hyprstream_rpc::capnp::FromCapnp::read_from(g.#leaf_getter()?)? },
                            _ => quote! { #leaf_name: g.#leaf_getter() },
                        }
                    }).collect();
                    quote! { #capnp_mod::#union_mod::Which::#variant(g) => #content_enum::#variant { #(#leaf_reads,)* }, }
                }
            }
        }).collect();
        quote! {
            content: match reader.which()? {
                #(#arms)*
            },
        }
    } else {
        TokenStream::new()
    };

    quote! {
        impl hyprstream_rpc::capnp::FromCapnp for #data_name {
            type Reader<'a> = #capnp_mod::#capnp_struct::Reader<'a>;

            #[allow(unused_variables)]
            fn read_from(reader: Self::Reader<'_>) -> anyhow::Result<Self> {
                Ok(Self {
                    #(#field_readers)*
                    #content_reader
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
            // Optional field-level `$domainType` newtype over `Text`: empty → None, else `Some(Type::new(..))`.
            CapnpType::Text if field.domain_type.is_some() => {
                let dt = resolve_field_domain_type(field.domain_type.as_deref().unwrap_or_default());
                quote! { #rust_name: { let v = reader.#getter_name()?.to_str()?; if v.is_empty() { None } else { Some(#dt::new(v.to_string())) } }, }
            }
            CapnpType::Text => {
                quote! { #rust_name: { let v = reader.#getter_name()?.to_str()?; if v.is_empty() { None } else { Some(v.to_string()) } }, }
            }
            CapnpType::Data => {
                if let Some(n) = field.fixed_size {
                    let n_usize = n as usize;
                    if n <= 32 {
                        quote! { #rust_name: {
                            let v = reader.#getter_name()?;
                            if v.is_empty() {
                                None
                            } else if v.len() == #n_usize {
                                let mut arr = [0u8; #n_usize];
                                arr.copy_from_slice(&v[..#n_usize]);
                                Some(arr)
                            } else {
                                anyhow::bail!("optional fixed-size field: expected {} bytes, got {}", #n_usize, v.len());
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
        // Field-level `$domainType` newtype over `Text`: read via `Type::new(String)`.
        CapnpType::Text if field.domain_type.is_some() => {
            let dt = resolve_field_domain_type(field.domain_type.as_deref().unwrap_or_default());
            quote! { #rust_name: #dt::new(reader.#getter_name()?.to_str()?.to_string()), }
        }
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

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod field_domain_type_tests {
    use super::*;
    use crate::resolve::ResolvedSchema;
    use crate::schema::types::{FieldDef, FieldSection, ParsedSchema, StructDef};

    fn text_field(name: &str, domain_type: Option<&str>) -> FieldDef {
        FieldDef {
            name: name.into(),
            type_name: "Text".into(),
            description: String::new(),
            fixed_size: None,
            optional: false,
            slot_offset: 0,
            section: FieldSection::Pointer,
            discriminant_value: 0xFFFF,
            serde_rename: None,
            domain_type: domain_type.map(String::from),
        }
    }

    fn data_only_schema(s: StructDef) -> ParsedSchema {
        ParsedSchema {
            request_variants: vec![],
            response_variants: vec![],
            structs: vec![s],
            scoped_clients: vec![],
            enums: vec![],
            request_struct: None,
            response_struct: None,
        }
    }

    /// A field carrying `$domainType("hyprstream_rpc::identity::Did")` must emit the newtype
    /// as the field's Rust type, write via `.as_str()`, and read via `Did::new(...)` — while a
    /// sibling plain `Text` field stays a `String`.
    #[test]
    fn field_level_domain_type_emits_newtype() {
        let s = StructDef {
            name: "Sample".into(),
            fields: vec![
                text_field("serviceDid", Some("hyprstream_rpc::identity::Did")),
                text_field("plain", None),
            ],
            has_union: false,
            domain_type: None,
            origin_file: None,
            data_words: 0,
            pointer_words: 2,
            discriminant_count: 0,
            discriminant_offset: 0,
            union_arms: vec![],
        };
        let schema = data_only_schema(s);
        let resolved = ResolvedSchema::from(&schema);
        let out = generate_data_structs(&resolved, "discovery", None).to_string();

        // Struct field uses the newtype, not String (TokenStream spaces out `::`).
        assert!(
            out.contains("service_did : hyprstream_rpc :: identity :: Did"),
            "expected Did-typed field, got:\n{out}"
        );
        // ToCapnp writes the newtype via `.as_str()`.
        assert!(
            out.contains("set_service_did") && out.contains("as_str ()"),
            "expected `.as_str()` write for domain field, got:\n{out}"
        );
        // FromCapnp reads via `Did::new(...)`.
        assert!(
            out.contains("hyprstream_rpc :: identity :: Did :: new"),
            "expected `Did::new(...)` read for domain field, got:\n{out}"
        );
        // The sibling plain Text field stays a String.
        assert!(
            out.contains("plain : String"),
            "expected plain Text field to remain String, got:\n{out}"
        );
    }

    /// Struct-level `$domainType` must keep its existing behavior (it does not turn the
    /// struct's own fields into newtypes — only field-level annotations do).
    #[test]
    fn struct_level_domain_type_unaffected() {
        let s = StructDef {
            name: "Wrapped".into(),
            fields: vec![text_field("value", None)],
            has_union: false,
            domain_type: Some("some::DomainType".into()),
            origin_file: None,
            data_words: 0,
            pointer_words: 1,
            discriminant_count: 0,
            discriminant_offset: 0,
            union_arms: vec![],
        };
        let schema = data_only_schema(s);
        let resolved = ResolvedSchema::from(&schema);
        let out = generate_data_structs(&resolved, "discovery", None).to_string();
        // Field without a field-level annotation stays a String regardless of struct-level $domainType.
        assert!(out.contains("value : String"), "got:\n{out}");
    }
}
