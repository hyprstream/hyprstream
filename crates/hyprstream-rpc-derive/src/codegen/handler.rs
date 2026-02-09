//! Handler trait, dispatch function, and response serializer generation.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::schema::types::*;
use crate::util::*;

/// Generate handler trait + dispatch function + response serializer.
pub fn generate_handler(service_name: &str, schema: &ParsedSchema) -> TokenStream {
    let pascal = to_pascal_case(service_name);
    let capnp_mod = format_ident!("{}_capnp", service_name);

    let handler_trait = generate_handler_trait(
        &pascal,
        &schema.request_variants,
        &schema.structs,
        &schema.enums,
        &schema.scoped_clients,
    );

    let dispatch_fn = generate_dispatch_fn(
        &pascal,
        &capnp_mod,
        &schema.request_variants,
        &schema.structs,
        &schema.enums,
        &schema.scoped_clients,
    );

    let serializer = generate_response_serializer(
        &pascal,
        &capnp_mod,
        &schema.response_variants,
        &schema.structs,
        &schema.enums,
    );

    quote! {
        #handler_trait
        #dispatch_fn
        #serializer
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler Trait
// ─────────────────────────────────────────────────────────────────────────────

fn generate_handler_trait(
    pascal: &str,
    request_variants: &[UnionVariant],
    structs: &[StructDef],
    enums: &[EnumDef],
    scoped_clients: &[ScopedClient],
) -> TokenStream {
    let trait_name = format_ident!("{}Handler", pascal);
    let response_type = format_ident!("{}ResponseVariant", pascal);
    let doc = format!("Generated handler trait for the {pascal} service.");
    let scoped_names: Vec<&str> = scoped_clients
        .iter()
        .map(|sc| sc.factory_name.as_str())
        .collect();

    let methods: Vec<TokenStream> = request_variants
        .iter()
        .map(|v| {
            let method_name = format_ident!("handle_{}", to_snake_case(&v.name));

            if scoped_names.contains(&v.name.as_str()) {
                let doc_str = format!("Handle scoped `{}` request (raw bytes for inner dispatch).", v.name);
                return quote! {
                    #[doc = #doc_str]
                    fn #method_name(&self, ctx: &crate::services::EnvelopeContext, request_id: u64, payload: &[u8]) -> anyhow::Result<Vec<u8>>;
                };
            }

            let doc_str = format!("Handle `{}` request.", v.name);
            let params = handler_method_params(v, structs, enums);

            quote! {
                #[doc = #doc_str]
                fn #method_name(&self, ctx: &crate::services::EnvelopeContext, request_id: u64 #(, #params)*) -> anyhow::Result<#response_type>;
            }
        })
        .collect();

    quote! {
        #[doc = #doc]
        pub trait #trait_name {
            #(#methods)*
        }
    }
}

fn handler_method_params(
    v: &UnionVariant,
    structs: &[StructDef],
    enums: &[EnumDef],
) -> Vec<TokenStream> {
    let ct = CapnpType::classify(&v.type_name, structs, enums);
    match ct {
        CapnpType::Void => vec![],
        CapnpType::Text => vec![quote! { value: &str }],
        CapnpType::Data => vec![quote! { data: &[u8] }],
        _ if ct.is_numeric() => {
            let ty = rust_type_tokens(&ct.rust_owned_type());
            vec![quote! { value: #ty }]
        }
        CapnpType::Struct(ref name) => {
            if let Some(sdef) = structs.iter().find(|s| s.name == *name) {
                sdef.fields
                    .iter()
                    .map(|f| {
                        let name = format_ident!("{}", to_snake_case(&f.name));
                        let ty = rust_type_tokens(&CapnpType::classify(&f.type_name, structs, enums).rust_param_type());
                        quote! { #name: #ty }
                    })
                    .collect()
            } else {
                vec![quote! { data: &[u8] }]
            }
        }
        _ => vec![quote! { data: &[u8] }],
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch Function
// ─────────────────────────────────────────────────────────────────────────────

fn generate_dispatch_fn(
    pascal: &str,
    capnp_mod: &syn::Ident,
    request_variants: &[UnionVariant],
    structs: &[StructDef],
    enums: &[EnumDef],
    scoped_clients: &[ScopedClient],
) -> TokenStream {
    let fn_name = format_ident!("dispatch_{}", to_snake_case(pascal));
    let trait_name = format_ident!("{}Handler", pascal);
    let req_snake = format_ident!("{}", to_snake_case(&format!("{pascal}Request")));
    let doc = format!("Dispatch a {pascal} request: deserialize, call handler, serialize response.");
    let scoped_names: Vec<&str> = scoped_clients
        .iter()
        .map(|sc| sc.factory_name.as_str())
        .collect();

    let match_arms: Vec<TokenStream> = request_variants
        .iter()
        .map(|v| {
            let variant_pascal = format_ident!("{}", to_pascal_case(&v.name));
            let handler_method = format_ident!("handle_{}", to_snake_case(&v.name));

            if scoped_names.contains(&v.name.as_str()) {
                return quote! {
                    Which::#variant_pascal(_) => {
                        return handler.#handler_method(ctx, request_id, payload);
                    }
                };
            }

            let ct = CapnpType::classify(&v.type_name, structs, enums);
            match ct {
                CapnpType::Void => quote! {
                    Which::#variant_pascal(()) => handler.#handler_method(ctx, request_id),
                },
                CapnpType::Text => quote! {
                    Which::#variant_pascal(v) => handler.#handler_method(ctx, request_id, v?.to_str()?),
                },
                CapnpType::Data => quote! {
                    Which::#variant_pascal(v) => handler.#handler_method(ctx, request_id, v?),
                },
                _ if ct.is_numeric() => quote! {
                    Which::#variant_pascal(v) => handler.#handler_method(ctx, request_id, v),
                },
                CapnpType::Struct(ref name) => {
                    if let Some(sdef) = structs.iter().find(|s| s.name == *name) {
                        let field_reads: Vec<TokenStream> = sdef.fields.iter().map(|f| {
                            let fname = format_ident!("{}", to_snake_case(&f.name));
                            let getter = format_ident!("get_{}", to_snake_case(&f.name));
                            let read_expr = dispatch_field_read_expr(&getter, &f.type_name, enums, structs);
                            quote! { let #fname = #read_expr; }
                        }).collect();

                        let args: Vec<TokenStream> = sdef.fields.iter().map(|f| {
                            let fname = format_ident!("{}", to_snake_case(&f.name));
                            let fct = CapnpType::classify(&f.type_name, structs, enums);
                            if fct.is_by_ref() {
                                quote! { &#fname }
                            } else {
                                quote! { #fname }
                            }
                        }).collect();

                        quote! {
                            Which::#variant_pascal(v) => {
                                let v = v?;
                                #(#field_reads)*
                                handler.#handler_method(ctx, request_id, #(#args),*)
                            }
                        }
                    } else {
                        let name_str = v.name.clone();
                        quote! {
                            Which::#variant_pascal(_) => {
                                anyhow::bail!("Unhandled variant: {}", #name_str);
                            }
                        }
                    }
                }
                _ => {
                    let name_str = v.name.clone();
                    quote! {
                        Which::#variant_pascal(_) => {
                            anyhow::bail!("Unhandled variant: {}", #name_str);
                        }
                    }
                }
            }
        })
        .collect();

    quote! {
        #[doc = #doc]
        pub fn #fn_name<H: #trait_name>(handler: &H, ctx: &crate::services::EnvelopeContext, payload: &[u8]) -> anyhow::Result<Vec<u8>> {
            let reader = capnp::serialize::read_message(
                &mut std::io::Cursor::new(payload),
                capnp::message::ReaderOptions::new(),
            )?;
            let req = reader.get_root::<crate::#capnp_mod::#req_snake::Reader>()?;
            let request_id = req.get_id();
            use crate::#capnp_mod::#req_snake::Which;
            let result = match req.which()? {
                #(#match_arms)*
                #[allow(unreachable_patterns)]
                _ => anyhow::bail!("Unknown request variant (not in schema)"),
            }?;
            serialize_response(request_id, &result)
        }
    }
}

/// Generate a field read expression for dispatch deserialization.
fn dispatch_field_read_expr(
    getter: &syn::Ident,
    type_name: &str,
    enums: &[EnumDef],
    structs: &[StructDef],
) -> TokenStream {
    let ct = CapnpType::classify(type_name, structs, enums);
    match ct {
        CapnpType::Text => quote! { v.#getter()?.to_str()?.to_string() },
        CapnpType::Data => quote! { v.#getter()?.to_vec() },
        _ if ct.is_numeric() => quote! { v.#getter() },
        CapnpType::ListText => quote! {
            v.#getter()?.iter().map(|s| Ok::<_, anyhow::Error>(s?.to_str()?.to_string())).collect::<Result<Vec<_>, _>>()?
        },
        CapnpType::ListData => quote! {
            v.#getter()?.iter().map(|d| Ok::<_, anyhow::Error>(d?.to_vec())).collect::<Result<Vec<_>, _>>()?
        },
        CapnpType::ListStruct(ref inner) => {
            if let Some(sdef) = structs.iter().find(|s| s.name == *inner) {
                let data_name = format_ident!("{}Data", inner);
                let field_reads: Vec<TokenStream> = sdef.fields.iter().map(|f| {
                    let f_name = format_ident!("{}", to_snake_case(&f.name));
                    let f_getter = format_ident!("get_{}", to_snake_case(&f.name));
                    let inner_read = dispatch_field_read_expr(&f_getter, &f.type_name, enums, structs);
                    quote! { #f_name: { let v = &item; #inner_read } }
                }).collect();

                quote! {
                    {
                        let list = v.#getter()?;
                        let mut result = Vec::with_capacity(list.len() as usize);
                        for item in list.iter() {
                            result.push(#data_name { #(#field_reads,)* });
                        }
                        result
                    }
                }
            } else {
                quote! { Vec::new() }
            }
        }
        CapnpType::Enum(_) => {
            quote! { format!("{:?}", v.#getter()?) }
        }
        _ => quote! { Default::default() },
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Response Serializer
// ─────────────────────────────────────────────────────────────────────────────

fn generate_response_serializer(
    pascal: &str,
    capnp_mod: &syn::Ident,
    response_variants: &[UnionVariant],
    structs: &[StructDef],
    enums: &[EnumDef],
) -> TokenStream {
    let response_type = format_ident!("{}ResponseVariant", pascal);
    let resp_snake = format_ident!("{}", to_snake_case(&format!("{pascal}Response")));
    let doc = format!("Serialize a {pascal}ResponseVariant to Cap'n Proto bytes.");

    let match_arms: Vec<TokenStream> = response_variants
        .iter()
        .map(|v| generate_serializer_arm(pascal, capnp_mod, v, structs, enums))
        .collect();

    quote! {
        #[doc = #doc]
        fn serialize_response(request_id: u64, variant: &#response_type) -> anyhow::Result<Vec<u8>> {
            hyprstream_rpc::serialize_message(|msg| {
                let mut resp = msg.init_root::<crate::#capnp_mod::#resp_snake::Builder>();
                resp.set_request_id(request_id);
                match variant {
                    #(#match_arms)*
                }
            })
        }
    }
}

fn generate_serializer_arm(
    pascal: &str,
    capnp_mod: &syn::Ident,
    v: &UnionVariant,
    structs: &[StructDef],
    enums: &[EnumDef],
) -> TokenStream {
    let response_type = format_ident!("{}ResponseVariant", pascal);
    let variant_pascal = format_ident!("{}", to_pascal_case(&v.name));
    let setter_name = format_ident!("set_{}", to_snake_case(&v.name));
    let init_name = format_ident!("init_{}", to_snake_case(&v.name));

    let ct = CapnpType::classify(&v.type_name, structs, enums);
    match ct {
        CapnpType::Void => quote! {
            #response_type::#variant_pascal => {
                resp.#setter_name(());
            }
        },
        CapnpType::Text | CapnpType::Data => quote! {
            #response_type::#variant_pascal(v) => {
                resp.#setter_name(v);
            }
        },
        _ if ct.is_numeric() => quote! {
            #response_type::#variant_pascal(v) => {
                resp.#setter_name(*v);
            }
        },
        CapnpType::Struct(ref name) => {
            if let Some(sdef) = structs.iter().find(|s| s.name == *name) {
                let field_names: Vec<syn::Ident> = sdef
                    .fields
                    .iter()
                    .map(|f| format_ident!("{}", to_snake_case(&f.name)))
                    .collect();
                let pattern = &field_names;

                if sdef.fields.is_empty() {
                    quote! {
                        #response_type::#variant_pascal { #(#pattern,)* } => {
                            resp.#init_name();
                        }
                    }
                } else {
                    let field_setters: Vec<TokenStream> = sdef
                        .fields
                        .iter()
                        .map(|f| {
                            let f_name = format_ident!("{}", to_snake_case(&f.name));
                            let f_setter = format_ident!("set_{}", to_snake_case(&f.name));
                            generate_response_field_setter_token(&f_name, &f_setter, &f.type_name, enums, structs, capnp_mod)
                        })
                        .collect();

                    quote! {
                        #response_type::#variant_pascal { #(#pattern,)* } => {
                            let mut inner = resp.#init_name();
                            #(#field_setters)*
                        }
                    }
                }
            } else {
                quote! {
                    #response_type::#variant_pascal { .. } => {
                        // Unknown struct type
                    }
                }
            }
        }
        _ => quote! {
            #response_type::#variant_pascal { .. } => {
                // Unhandled type
            }
        },
    }
}

fn generate_response_field_setter_token(
    f_name: &syn::Ident,
    f_setter: &syn::Ident,
    type_name: &str,
    enums: &[EnumDef],
    structs: &[StructDef],
    capnp_mod: &syn::Ident,
) -> TokenStream {
    let ct = CapnpType::classify(type_name, structs, enums);
    let init_list_name = || {
        format_ident!(
            "init_{}",
            f_setter
                .to_string()
                .strip_prefix("set_")
                .unwrap_or(&f_setter.to_string())
        )
    };

    match ct {
        CapnpType::Text | CapnpType::Data => quote! { inner.#f_setter(#f_name); },
        _ if ct.is_numeric() => quote! { inner.#f_setter(*#f_name); },
        CapnpType::ListText => {
            let init_name = init_list_name();
            quote! {
                {
                    let mut list = inner.reborrow().#init_name(#f_name.len() as u32);
                    for (i, item) in #f_name.iter().enumerate() {
                        list.set(i as u32, item.as_str());
                    }
                }
            }
        }
        CapnpType::ListData => {
            let init_name = init_list_name();
            quote! {
                {
                    let mut list = inner.reborrow().#init_name(#f_name.len() as u32);
                    for (i, item) in #f_name.iter().enumerate() {
                        list.set(i as u32, item);
                    }
                }
            }
        }
        CapnpType::ListStruct(ref inner_type) => {
            if let Some(sdef) = structs.iter().find(|s| s.name == *inner_type) {
                let init_name = init_list_name();
                let field_setters: Vec<TokenStream> = sdef.fields.iter().map(|field| {
                    let inner_f = format_ident!("{}", to_snake_case(&field.name));
                    let inner_setter = format_ident!("set_{}", to_snake_case(&field.name));
                    let fct = CapnpType::classify(&field.type_name, structs, enums);
                    match fct {
                        CapnpType::Text | CapnpType::Data => quote! { entry.#inner_setter(&item.#inner_f); },
                        _ => quote! { entry.#inner_setter(item.#inner_f); },
                    }
                }).collect();

                quote! {
                    {
                        let mut list = inner.reborrow().#init_name(#f_name.len() as u32);
                        for (i, item) in #f_name.iter().enumerate() {
                            let mut entry = list.reborrow().get(i as u32);
                            #(#field_setters)*
                        }
                    }
                }
            } else {
                quote! {}
            }
        }
        CapnpType::Enum(ref enum_name) => {
            if let Some(enum_def) = enums.iter().find(|e| e.name == *enum_name) {
                let enum_rust_name = format_ident!("{}Enum", enum_name);
                let type_ident = format_ident!("{}", enum_name);
                let match_arms: Vec<TokenStream> = enum_def.variants.iter().map(|(vname, _)| {
                    let v_pascal = format_ident!("{}", to_pascal_case(vname));
                    quote! { #enum_rust_name::#v_pascal => crate::#capnp_mod::#type_ident::#v_pascal }
                }).collect();
                quote! {
                    inner.#f_setter(match #f_name {
                        #(#match_arms,)*
                    });
                }
            } else {
                quote! {}
            }
        }
        _ => quote! {},
    }
}
