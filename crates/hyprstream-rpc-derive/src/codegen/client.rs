//! Client struct, response enum, request methods, and parse_response generation.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::schema::types::*;
use crate::util::*;

// ─────────────────────────────────────────────────────────────────────────────
// Response Enum
// ─────────────────────────────────────────────────────────────────────────────

pub fn generate_response_enum(service_name: &str, schema: &ParsedSchema) -> TokenStream {
    let pascal = to_pascal_case(service_name);
    let enum_name_str = format!("{pascal}ResponseVariant");
    generate_response_enum_from_variants(
        &enum_name_str,
        &schema.response_variants,
        &schema.structs,
        &schema.enums,
    )
}

pub fn generate_response_enum_from_variants(
    enum_name_str: &str,
    variants: &[UnionVariant],
    structs: &[StructDef],
    enums: &[EnumDef],
) -> TokenStream {
    let enum_name = format_ident!("{}", enum_name_str);
    let doc = format!("Response variants for the {} service.", enum_name_str);

    let variant_tokens: Vec<TokenStream> = variants
        .iter()
        .map(|v| generate_enum_variant(v, structs, enums))
        .collect();

    quote! {
        #[doc = #doc]
        #[derive(Debug)]
        pub enum #enum_name {
            #(#variant_tokens,)*
        }
    }
}

fn generate_enum_variant(
    v: &UnionVariant,
    structs: &[StructDef],
    enums: &[EnumDef],
) -> TokenStream {
    let variant_pascal = format_ident!("{}", to_pascal_case(&v.name));

    match v.type_name.as_str() {
        "Void" => quote! { #variant_pascal },
        "Bool" => quote! { #variant_pascal(bool) },
        "Text" => quote! { #variant_pascal(String) },
        "Data" => quote! { #variant_pascal(Vec<u8>) },
        "UInt32" | "UInt64" | "Int32" | "Int64" | "Float32" | "Float64" | "UInt8" | "UInt16"
        | "Int8" | "Int16" => {
            let rust_type = rust_type_tokens(&CapnpType::classify_primitive(&v.type_name).rust_owned_type());
            quote! { #variant_pascal(#rust_type) }
        }
        type_name if type_name.starts_with("List(") => {
            let inner = &type_name[5..type_name.len() - 1];
            match inner {
                "Text" => quote! { #variant_pascal(Vec<String>) },
                "Data" => quote! { #variant_pascal(Vec<Vec<u8>>) },
                _ => {
                    if structs.iter().any(|s| s.name == inner) {
                        let inner_data = format_ident!("{}Data", inner);
                        quote! { #variant_pascal(Vec<#inner_data>) }
                    } else {
                        quote! { #variant_pascal(Vec<String>) }
                    }
                }
            }
        }
        struct_name => {
            if let Some(s) = structs.iter().find(|s| s.name == struct_name) {
                let fields: Vec<TokenStream> = s
                    .fields
                    .iter()
                    .map(|field| {
                        let rust_name = format_ident!("{}", to_snake_case(&field.name));
                        let rust_type = rust_type_tokens(&CapnpType::classify(&field.type_name, structs, enums).rust_owned_type());
                        quote! { #rust_name: #rust_type }
                    })
                    .collect();
                quote! { #variant_pascal { #(#fields,)* } }
            } else {
                quote! { #variant_pascal(Vec<u8>) }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Client Struct + Impl
// ─────────────────────────────────────────────────────────────────────────────

pub fn generate_client(service_name: &str, schema: &ParsedSchema) -> TokenStream {
    let pascal = to_pascal_case(service_name);
    let client_name = format_ident!("{}Client", pascal);
    let response_type = format_ident!("{}ResponseVariant", pascal);
    let capnp_mod = format_ident!("{}_capnp", service_name);
    let req_type = format_ident!("{}", to_snake_case(&format!("{pascal}Request")));
    let doc = format!("Auto-generated client for the {pascal} service.");

    let scoped_variant_names: Vec<&str> = schema
        .scoped_clients
        .iter()
        .map(|sc| sc.factory_name.as_str())
        .collect();

    // Request methods (skip scoped variants)
    let request_methods: Vec<TokenStream> = schema
        .request_variants
        .iter()
        .filter(|v| !scoped_variant_names.contains(&v.name.as_str()))
        .map(|v| {
            generate_request_method(
                &capnp_mod,
                &req_type,
                &response_type,
                v,
                &schema.structs,
                &schema.enums,
                None,
            )
        })
        .collect();

    // parse_response method
    let parse_response = generate_parse_response_fn(
        &response_type,
        &capnp_mod,
        &format_ident!("{}", to_snake_case(&format!("{pascal}Response"))),
        &schema.response_variants,
        &schema.structs,
        &schema.enums,
    );

    // Factory methods for scoped clients
    let factory_methods: Vec<TokenStream> = schema
        .scoped_clients
        .iter()
        .map(|sc| generate_scoped_factory_method(sc))
        .collect();

    // ServiceClient impl
    let service_name_lit = service_name;

    quote! {
        #[doc = #doc]
        #[derive(Clone)]
        pub struct #client_name {
            client: Arc<ZmqClientBase>,
        }

        impl ServiceClient for #client_name {
            const SERVICE_NAME: &'static str = #service_name_lit;

            fn from_zmq(client: ZmqClientBase) -> Self {
                Self { client: Arc::new(client) }
            }
        }

        impl #client_name {
            /// Get the next request ID.
            pub fn next_id(&self) -> u64 {
                self.client.next_id()
            }

            /// Send a raw request and return the raw response bytes.
            pub async fn call(&self, payload: Vec<u8>) -> anyhow::Result<Vec<u8>> {
                self.client.call(payload, CallOptions::default()).await
            }

            /// Send a raw request with custom options and return the raw response bytes.
            pub async fn call_with_options(&self, payload: Vec<u8>, opts: CallOptions) -> anyhow::Result<Vec<u8>> {
                self.client.call(payload, opts).await
            }

            /// Get the endpoint this client is connected to.
            pub fn endpoint(&self) -> &str {
                self.client.endpoint()
            }

            /// Get the signing key used by this client.
            pub fn signing_key(&self) -> &ed25519_dalek::SigningKey {
                self.client.signing_key()
            }

            /// Get the request identity used by this client.
            pub fn identity(&self) -> &hyprstream_rpc::envelope::RequestIdentity {
                self.client.identity()
            }

            #(#request_methods)*

            #parse_response

            #(#factory_methods)*
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Request Method Generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a single request method. Used for both top-level and scoped clients.
pub fn generate_request_method(
    capnp_mod: &syn::Ident,
    req_type: &syn::Ident,
    response_type: &syn::Ident,
    variant: &UnionVariant,
    structs: &[StructDef],
    enums: &[EnumDef],
    scope: Option<&ScopedMethodContext>,
) -> TokenStream {
    let method_name = format_ident!("{}", to_snake_case(&variant.name));
    let doc = format!("{} ({} variant)", to_snake_case(&variant.name), variant.type_name);

    // For scoped methods, we wrap in the outer init
    let (outer_req_setup, inner_accessor, parse_call) = if let Some(sc) = scope {
        let factory_snake = format_ident!("{}", to_snake_case(&sc.factory_name));
        let factory_init = format_ident!("init_{}", to_snake_case(&sc.factory_name));
        let scope_setters: Vec<TokenStream> = sc.scope_fields.iter().map(|f| {
            let setter_name = format_ident!("set_{}", to_snake_case(&f.name));
            let field_name = format_ident!("{}", to_snake_case(&f.name));
            match f.type_name.as_str() {
                "Text" => quote! { inner.#setter_name(&self.#field_name); },
                _ => quote! { inner.#setter_name(self.#field_name); },
            }
        }).collect();

        (
            quote! {
                let mut inner = req.#factory_init();
                #(#scope_setters)*
            },
            Some(factory_snake),
            quote! { Self::parse_scoped_response(&response) },
        )
    } else {
        (TokenStream::new(), None, quote! { Self::parse_response(&response) })
    };

    match variant.type_name.as_str() {
        "Void" => {
            let set_method = format_ident!("set_{}", to_snake_case(&variant.name));
            let setter = if inner_accessor.is_some() {
                quote! { inner.#set_method(()); }
            } else {
                quote! { req.#set_method(()); }
            };
            quote! {
                #[doc = #doc]
                pub async fn #method_name(&self) -> anyhow::Result<#response_type> {
                    let id = self.next_id();
                    let payload = hyprstream_rpc::serialize_message(|msg| {
                        let mut req = msg.init_root::<crate::#capnp_mod::#req_type::Builder>();
                        req.set_id(id);
                        #outer_req_setup
                        #setter
                    })?;
                    let response = self.call(payload).await?;
                    #parse_call
                }
            }
        }
        "Text" => {
            let set_method = format_ident!("set_{}", to_snake_case(&variant.name));
            let setter = if inner_accessor.is_some() {
                quote! { inner.#set_method(value); }
            } else {
                quote! { req.#set_method(value); }
            };
            quote! {
                #[doc = #doc]
                pub async fn #method_name(&self, value: &str) -> anyhow::Result<#response_type> {
                    let id = self.next_id();
                    let payload = hyprstream_rpc::serialize_message(|msg| {
                        let mut req = msg.init_root::<crate::#capnp_mod::#req_type::Builder>();
                        req.set_id(id);
                        #outer_req_setup
                        #setter
                    })?;
                    let response = self.call(payload).await?;
                    #parse_call
                }
            }
        }
        "Data" => {
            let set_method = format_ident!("set_{}", to_snake_case(&variant.name));
            let setter = if inner_accessor.is_some() {
                quote! { inner.#set_method(value); }
            } else {
                quote! { req.#set_method(value); }
            };
            quote! {
                #[doc = #doc]
                pub async fn #method_name(&self, value: &[u8]) -> anyhow::Result<#response_type> {
                    let id = self.next_id();
                    let payload = hyprstream_rpc::serialize_message(|msg| {
                        let mut req = msg.init_root::<crate::#capnp_mod::#req_type::Builder>();
                        req.set_id(id);
                        #outer_req_setup
                        #setter
                    })?;
                    let response = self.call(payload).await?;
                    #parse_call
                }
            }
        }
        "Bool" => {
            let set_method = format_ident!("set_{}", to_snake_case(&variant.name));
            let setter = if inner_accessor.is_some() {
                quote! { inner.#set_method(value); }
            } else {
                quote! { req.#set_method(value); }
            };
            quote! {
                #[doc = #doc]
                pub async fn #method_name(&self, value: bool) -> anyhow::Result<#response_type> {
                    let id = self.next_id();
                    let payload = hyprstream_rpc::serialize_message(|msg| {
                        let mut req = msg.init_root::<crate::#capnp_mod::#req_type::Builder>();
                        req.set_id(id);
                        #outer_req_setup
                        #setter
                    })?;
                    let response = self.call(payload).await?;
                    #parse_call
                }
            }
        }
        struct_name => {
            if let Some(s) = structs.iter().find(|s| s.name == struct_name) {
                let is_void_wrapper = s.fields.len() == 1 && s.fields[0].type_name == "Void";

                if is_void_wrapper {
                    let init_method = format_ident!("init_{}", to_snake_case(&variant.name));
                    let setter = if inner_accessor.is_some() {
                        quote! { inner.#init_method(); }
                    } else {
                        quote! { req.#init_method(); }
                    };
                    quote! {
                        #[doc = #doc]
                        pub async fn #method_name(&self) -> anyhow::Result<#response_type> {
                            let id = self.next_id();
                            let payload = hyprstream_rpc::serialize_message(|msg| {
                                let mut req = msg.init_root::<crate::#capnp_mod::#req_type::Builder>();
                                req.set_id(id);
                                #outer_req_setup
                                #setter
                            })?;
                            let response = self.call(payload).await?;
                            #parse_call
                        }
                    }
                } else {
                    let params = generate_method_params(&s.fields, enums, structs);
                    let init_method = format_ident!("init_{}", to_snake_case(&variant.name));
                    let setters = generate_struct_setters(&s.fields, enums, structs, capnp_mod);

                    let inner_init = if inner_accessor.is_some() {
                        quote! { let mut inner = inner.#init_method(); }
                    } else {
                        quote! { let mut inner = req.#init_method(); }
                    };

                    quote! {
                        #[doc = #doc]
                        #[allow(unused_mut)]
                        pub async fn #method_name(&self #(, #params)*) -> anyhow::Result<#response_type> {
                            let id = self.next_id();
                            let payload = hyprstream_rpc::serialize_message(|msg| {
                                let mut req = msg.init_root::<crate::#capnp_mod::#req_type::Builder>();
                                req.set_id(id);
                                #outer_req_setup
                                #inner_init
                                #(#setters)*
                            })?;
                            let response = self.call(payload).await?;
                            #parse_call
                        }
                    }
                }
            } else {
                let _comment = format!("TODO: {} — struct {} not found in schema", to_snake_case(&variant.name), struct_name);
                quote! {
                    // #comment
                }
            }
        }
    }
}

/// Context for scoped method generation.
pub struct ScopedMethodContext {
    pub factory_name: String,
    pub scope_fields: Vec<FieldDef>,
}

/// Generate method parameter tokens for a struct's fields.
fn generate_method_params(
    fields: &[FieldDef],
    enums: &[EnumDef],
    structs: &[StructDef],
) -> Vec<TokenStream> {
    fields
        .iter()
        .map(|field| {
            let rust_name = format_ident!("{}", to_snake_case(&field.name));
            let type_str = CapnpType::classify(&field.type_name, structs, enums).rust_param_type();
            let rust_type = rust_type_tokens(&type_str);
            quote! { #rust_name: #rust_type }
        })
        .collect()
}

/// Generate setter calls for fields when building a request struct.
fn generate_struct_setters(
    fields: &[FieldDef],
    enums: &[EnumDef],
    structs: &[StructDef],
    capnp_mod: &syn::Ident,
) -> Vec<TokenStream> {
    fields
        .iter()
        .map(|field| {
            let rust_name = format_ident!("{}", to_snake_case(&field.name));
            let setter_name = format_ident!("set_{}", to_snake_case(&field.name));
            generate_field_setter(&rust_name, &setter_name, &field.type_name, enums, structs, capnp_mod)
        })
        .collect()
}

fn generate_field_setter(
    rust_name: &syn::Ident,
    setter_name: &syn::Ident,
    type_name: &str,
    enums: &[EnumDef],
    structs: &[StructDef],
    capnp_mod: &syn::Ident,
) -> TokenStream {
    match type_name {
        "Text" | "Data" => quote! { inner.#setter_name(#rust_name); },
        "Bool" | "UInt32" | "UInt64" | "Int32" | "Int64" | "Float32" | "Float64" => {
            quote! { inner.#setter_name(#rust_name); }
        }
        t if t.starts_with("List(") => {
            let inner_type = &t[5..t.len() - 1];
            generate_list_setter(rust_name, setter_name, inner_type, enums, structs, capnp_mod)
        }
        t => {
            if let Some(e) = enums.iter().find(|e| e.name == t) {
                let type_ident = format_ident!("{}", t);
                let match_arms: Vec<TokenStream> = e.variants.iter().map(|(vname, _)| {
                    let snake = to_snake_case(vname);
                    let pascal = format_ident!("{}", to_pascal_case(vname));
                    quote! { #snake => crate::#capnp_mod::#type_ident::#pascal }
                }).collect();
                let default_arm = if let Some((first, _)) = e.variants.first() {
                    let first_pascal = format_ident!("{}", to_pascal_case(first));
                    quote! { _ => crate::#capnp_mod::#type_ident::#first_pascal }
                } else {
                    TokenStream::new()
                };
                quote! {
                    inner.#setter_name(match #rust_name {
                        #(#match_arms,)*
                        #default_arm,
                    });
                }
            } else {
                let _comment = format!("unknown type: set_{} for {}", to_snake_case(&setter_name.to_string()), t);
                quote! { /* #comment */ }
            }
        }
    }
}

fn generate_list_setter(
    rust_name: &syn::Ident,
    setter_name: &syn::Ident,
    inner_type: &str,
    enums: &[EnumDef],
    structs: &[StructDef],
    capnp_mod: &syn::Ident,
) -> TokenStream {
    // For List fields, capnp uses init_* to allocate the list, not set_*
    let init_name = format_ident!(
        "init_{}",
        setter_name
            .to_string()
            .strip_prefix("set_")
            .unwrap_or(&setter_name.to_string())
    );

    match inner_type {
        "Text" => {
            quote! {
                {
                    let mut list = inner.reborrow().#init_name(#rust_name.len() as u32);
                    for (i, item) in #rust_name.iter().enumerate() {
                        list.set(i as u32, item.as_str());
                    }
                }
            }
        }
        "Data" => {
            quote! {
                {
                    let mut list = inner.reborrow().#init_name(#rust_name.len() as u32);
                    for (i, item) in #rust_name.iter().enumerate() {
                        list.set(i as u32, item.as_slice());
                    }
                }
            }
        }
        struct_name => {
            if let Some(s) = structs.iter().find(|s| s.name == struct_name) {
                let field_sets: Vec<TokenStream> = s.fields.iter().map(|f| {
                    let f_name = format_ident!("{}", to_snake_case(&f.name));
                    let f_setter = format_ident!("set_{}", to_snake_case(&f.name));
                    generate_list_struct_field_set(&f_name, &f_setter, &f.type_name, enums, structs, capnp_mod)
                }).collect();

                quote! {
                    {
                        let mut list = inner.reborrow().#init_name(#rust_name.len() as u32);
                        for (i, item) in #rust_name.iter().enumerate() {
                            let mut entry = list.reborrow().get(i as u32);
                            #(#field_sets)*
                        }
                    }
                }
            } else {
                let _comment = format!("List({struct_name}) — struct not found in schema");
                quote! { }
            }
        }
    }
}

fn generate_list_struct_field_set(
    f_name: &syn::Ident,
    f_setter: &syn::Ident,
    type_name: &str,
    enums: &[EnumDef],
    _structs: &[StructDef],
    capnp_mod: &syn::Ident,
) -> TokenStream {
    match type_name {
        "Text" | "Data" => quote! { entry.#f_setter(&item.#f_name); },
        "Bool" | "UInt32" | "UInt64" | "Int32" | "Int64" | "Float32" | "Float64" => {
            quote! { entry.#f_setter(item.#f_name); }
        }
        t if t.starts_with("List(") => {
            let inner_type = &t[5..t.len() - 1];
            if inner_type == "Text" {
                let init_name = format_ident!("init_{}", f_setter.to_string().strip_prefix("set_").unwrap_or(&f_setter.to_string()));
                quote! {
                    {
                        let mut list = entry.reborrow().#init_name(item.#f_name.len() as u32);
                        for (j, val) in item.#f_name.iter().enumerate() {
                            list.set(j as u32, val.as_str());
                        }
                    }
                }
            } else {
                let _comment = format!("TODO: nested List({inner_type}) in struct");
                quote! { /* #comment */ }
            }
        }
        t => {
            if let Some(e) = enums.iter().find(|e| e.name == t) {
                let enum_rust_name = format_ident!("{}Enum", t);
                let type_ident = format_ident!("{}", t);
                let match_arms: Vec<TokenStream> = e.variants.iter().map(|(vname, _)| {
                    let v_pascal = format_ident!("{}", to_pascal_case(vname));
                    quote! { #enum_rust_name::#v_pascal => crate::#capnp_mod::#type_ident::#v_pascal }
                }).collect();
                quote! {
                    entry.#f_setter(match item.#f_name {
                        #(#match_arms,)*
                    });
                }
            } else {
                let _comment = format!("unknown type: {} for {}", t, f_setter);
                quote! { /* #comment */ }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_response
// ─────────────────────────────────────────────────────────────────────────────

pub fn generate_parse_response_fn(
    response_type: &syn::Ident,
    capnp_mod: &syn::Ident,
    resp_type: &syn::Ident,
    variants: &[UnionVariant],
    structs: &[StructDef],
    enums: &[EnumDef],
) -> TokenStream {
    let which_ident = format_ident!("Which");
    let match_arms: Vec<TokenStream> = variants
        .iter()
        .map(|v| generate_parse_match_arm(response_type, capnp_mod, v, structs, enums, &which_ident))
        .collect();

    quote! {
        /// Parse a response from raw bytes.
        pub fn parse_response(bytes: &[u8]) -> anyhow::Result<#response_type> {
            let reader = capnp::serialize::read_message(
                &mut std::io::Cursor::new(bytes),
                capnp::message::ReaderOptions::new(),
            )?;
            let resp = reader.get_root::<crate::#capnp_mod::#resp_type::Reader>()?;
            use crate::#capnp_mod::#resp_type::Which;
            match resp.which()? {
                #(#match_arms)*
            }
        }
    }
}

/// Generate a single parse match arm.
///
/// The `which_ident` parameter allows this to be shared between top-level
/// (`Which`) and scoped (`InnerWhich`) parse contexts.
pub fn generate_parse_match_arm(
    response_type: &syn::Ident,
    capnp_mod: &syn::Ident,
    v: &UnionVariant,
    structs: &[StructDef],
    enums: &[EnumDef],
    which_ident: &syn::Ident,
) -> TokenStream {
    let variant_pascal = format_ident!("{}", to_pascal_case(&v.name));
    let ct = CapnpType::classify(&v.type_name, structs, enums);

    match ct {
        CapnpType::Void => quote! {
            #which_ident::#variant_pascal(()) => Ok(#response_type::#variant_pascal),
        },
        CapnpType::Bool
        | CapnpType::UInt8 | CapnpType::UInt16 | CapnpType::UInt32 | CapnpType::UInt64
        | CapnpType::Int8 | CapnpType::Int16 | CapnpType::Int32 | CapnpType::Int64
        | CapnpType::Float32 | CapnpType::Float64 => quote! {
            #which_ident::#variant_pascal(v) => Ok(#response_type::#variant_pascal(v)),
        },
        CapnpType::Text => quote! {
            #which_ident::#variant_pascal(v) => Ok(#response_type::#variant_pascal(v?.to_str()?.to_string())),
        },
        CapnpType::Data => quote! {
            #which_ident::#variant_pascal(v) => Ok(#response_type::#variant_pascal(v?.to_vec())),
        },
        CapnpType::ListText | CapnpType::ListData | CapnpType::ListStruct(_) => {
            generate_list_parse_arm(&variant_pascal, response_type, &v.type_name, structs, enums, capnp_mod, which_ident)
        }
        CapnpType::Struct(ref name) => {
            if let Some(s) = structs.iter().find(|s| s.name == *name) {
                let field_reads = generate_field_reads_tokens(&s.fields, enums, structs, capnp_mod, &format_ident!("v"));
                quote! {
                    #which_ident::#variant_pascal(v) => {
                        let v = v?;
                        Ok(#response_type::#variant_pascal {
                            #(#field_reads)*
                        })
                    }
                }
            } else {
                quote! {
                    #which_ident::#variant_pascal(_v) => {
                        Ok(#response_type::#variant_pascal(Vec::new()))
                    }
                }
            }
        }
        _ => quote! {
            #which_ident::#variant_pascal(_v) => {
                Ok(#response_type::#variant_pascal(Vec::new()))
            }
        },
    }
}

fn generate_list_parse_arm(
    variant_pascal: &syn::Ident,
    response_type: &syn::Ident,
    type_name: &str,
    structs: &[StructDef],
    enums: &[EnumDef],
    capnp_mod: &syn::Ident,
    which_ident: &syn::Ident,
) -> TokenStream {
    let ct = CapnpType::classify(type_name, structs, enums);
    match ct {
        CapnpType::ListText => quote! {
            #which_ident::#variant_pascal(v) => {
                let list = v?;
                let mut result = Vec::with_capacity(list.len() as usize);
                for i in 0..list.len() {
                    result.push(list.get(i)?.to_str()?.to_string());
                }
                Ok(#response_type::#variant_pascal(result))
            }
        },
        CapnpType::ListStruct(ref inner) => {
            if let Some(s) = structs.iter().find(|s| s.name == *inner) {
                let data_name = format_ident!("{}Data", inner);
                let item_ident = format_ident!("item");
                let field_reads = generate_field_reads_tokens(&s.fields, enums, structs, capnp_mod, &item_ident);
                quote! {
                    #which_ident::#variant_pascal(v) => {
                        let list = v?;
                        let mut result = Vec::with_capacity(list.len() as usize);
                        for i in 0..list.len() {
                            let item = list.get(i);
                            result.push(#data_name {
                                #(#field_reads)*
                            });
                        }
                        Ok(#response_type::#variant_pascal(result))
                    }
                }
            } else {
                quote! {
                    #which_ident::#variant_pascal(_v) => {
                        Ok(#response_type::#variant_pascal(Vec::new()))
                    }
                }
            }
        }
        _ => quote! {
            #which_ident::#variant_pascal(_v) => {
                Ok(#response_type::#variant_pascal(Vec::new()))
            }
        },
    }
}

/// Generate field read expressions for struct deserialization.
pub fn generate_field_reads_tokens(
    fields: &[FieldDef],
    enums: &[EnumDef],
    structs: &[StructDef],
    capnp_mod: &syn::Ident,
    var_name: &syn::Ident,
) -> Vec<TokenStream> {
    fields
        .iter()
        .map(|field| {
            let rust_name = format_ident!("{}", to_snake_case(&field.name));
            let getter_name = format_ident!("get_{}", to_snake_case(&field.name));
            generate_single_field_read(&rust_name, &getter_name, &field.type_name, enums, structs, capnp_mod, var_name)
        })
        .collect()
}

fn generate_single_field_read(
    rust_name: &syn::Ident,
    getter_name: &syn::Ident,
    type_name: &str,
    enums: &[EnumDef],
    structs: &[StructDef],
    capnp_mod: &syn::Ident,
    var_name: &syn::Ident,
) -> TokenStream {
    match type_name {
        "Text" => quote! { #rust_name: #var_name.#getter_name()?.to_str()?.to_string(), },
        "Data" => quote! { #rust_name: #var_name.#getter_name()?.to_vec(), },
        "Bool" => quote! { #rust_name: #var_name.#getter_name(), },
        "UInt32" | "UInt64" | "Int32" | "Int64" | "Float32" | "Float64" | "UInt8" | "UInt16" | "Int8" | "Int16" => {
            quote! { #rust_name: #var_name.#getter_name(), }
        }
        t if t.starts_with("List(") => {
            let inner = &t[5..t.len() - 1];
            generate_list_field_read(rust_name, getter_name, inner, enums, structs, capnp_mod, var_name)
        }
        t => {
            if let Some(e) = enums.iter().find(|e| e.name == t) {
                let enum_rust_name = format_ident!("{}Enum", t);
                let type_ident = format_ident!("{}", t);
                let match_arms: Vec<TokenStream> = e.variants.iter().map(|(vname, _)| {
                    let v_pascal = format_ident!("{}", to_pascal_case(vname));
                    quote! { crate::#capnp_mod::#type_ident::#v_pascal => #enum_rust_name::#v_pascal }
                }).collect();
                quote! {
                    #rust_name: match #var_name.#getter_name()? {
                        #(#match_arms,)*
                    },
                }
            } else if let Some(s) = structs.iter().find(|s| s.name == t) {
                let data_name = format_ident!("{}Data", t);
                let inner_var = format_ident!("{}_{}", var_name, rust_name);
                let inner_reads = generate_field_reads_tokens(&s.fields, enums, structs, capnp_mod, &inner_var);
                quote! {
                    #rust_name: {
                        let #inner_var = #var_name.#getter_name()?;
                        #data_name {
                            #(#inner_reads)*
                        }
                    },
                }
            } else {
                quote! { #rust_name: Default::default(), }
            }
        }
    }
}

fn generate_list_field_read(
    rust_name: &syn::Ident,
    getter_name: &syn::Ident,
    inner: &str,
    enums: &[EnumDef],
    structs: &[StructDef],
    capnp_mod: &syn::Ident,
    var_name: &syn::Ident,
) -> TokenStream {
    match inner {
        "Text" => quote! {
            #rust_name: {
                let list = #var_name.#getter_name()?;
                let mut result = Vec::with_capacity(list.len() as usize);
                for i in 0..list.len() {
                    result.push(list.get(i)?.to_str()?.to_string());
                }
                result
            },
        },
        "Data" => quote! {
            #rust_name: {
                let list = #var_name.#getter_name()?;
                let mut result = Vec::with_capacity(list.len() as usize);
                for i in 0..list.len() {
                    result.push(list.get(i)?.to_vec());
                }
                result
            },
        },
        struct_name => {
            if let Some(s) = structs.iter().find(|s| s.name == struct_name) {
                let data_name = format_ident!("{}Data", struct_name);
                let item_ident = format_ident!("item");
                let inner_reads = generate_field_reads_tokens(&s.fields, enums, structs, capnp_mod, &item_ident);
                quote! {
                    #rust_name: {
                        let list = #var_name.#getter_name()?;
                        let mut result = Vec::with_capacity(list.len() as usize);
                        for i in 0..list.len() {
                            let item = list.get(i);
                            result.push(#data_name {
                                #(#inner_reads)*
                            });
                        }
                        result
                    },
                }
            } else {
                quote! { #rust_name: Vec::new(), }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scoped Factory Method
// ─────────────────────────────────────────────────────────────────────────────

fn generate_scoped_factory_method(sc: &ScopedClient) -> TokenStream {
    let method_name = format_ident!("{}", to_snake_case(&sc.factory_name));
    let client_name_ident = format_ident!("{}", sc.client_name);
    let doc = format!("Create a scoped {} client.", sc.factory_name);

    let params: Vec<TokenStream> = sc.scope_fields.iter().map(|f| {
        let name = format_ident!("{}", to_snake_case(&f.name));
        let ty = rust_type_tokens(&CapnpType::classify_primitive(&f.type_name).rust_param_type());
        quote! { #name: #ty }
    }).collect();

    let field_inits: Vec<TokenStream> = sc.scope_fields.iter().map(|f| {
        let name = format_ident!("{}", to_snake_case(&f.name));
        match f.type_name.as_str() {
            "Text" => quote! { #name: #name.to_owned() },
            _ => quote! { #name },
        }
    }).collect();

    quote! {
        #[doc = #doc]
        pub fn #method_name(&self #(, #params)*) -> #client_name_ident {
            #client_name_ident {
                client: Arc::clone(&self.client),
                #(#field_inits,)*
            }
        }
    }
}
