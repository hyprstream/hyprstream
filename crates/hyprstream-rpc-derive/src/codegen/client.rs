//! Client struct, response enum, request methods, and parse_response generation.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::schema::types::*;
use crate::util::*;

// ─────────────────────────────────────────────────────────────────────────────
// Domain Type Resolution
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve a domain type path from a `$domainType` annotation value.
///
/// The annotation value is a crate-relative Rust path (e.g., `"runtime::VersionResponse"`).
/// When `types_crate` is `Some`, the path is resolved through the external crate;
/// otherwise it is resolved through `crate::`.
fn resolve_domain_type(domain_path: &str, types_crate: Option<&syn::Path>) -> TokenStream {
    let path: TokenStream = domain_path.parse().unwrap_or_else(|_| quote! { Vec<u8> });
    match types_crate {
        Some(tc) => quote! { #tc::#path },
        None => quote! { crate::#path },
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Response Enum
// ─────────────────────────────────────────────────────────────────────────────

pub fn generate_response_enum(service_name: &str, schema: &ParsedSchema, types_crate: Option<&syn::Path>) -> TokenStream {
    let pascal = to_pascal_case(service_name);
    let enum_name_str = format!("{pascal}ResponseVariant");
    generate_response_enum_from_variants(
        &enum_name_str,
        &schema.response_variants,
        &schema.structs,
        &schema.enums,
        types_crate,
    )
}

pub fn generate_response_enum_from_variants(
    enum_name_str: &str,
    variants: &[UnionVariant],
    structs: &[StructDef],
    enums: &[EnumDef],
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let enum_name = format_ident!("{}", enum_name_str);
    let doc = format!("Response variants for the {} service.", enum_name_str);

    let variant_tokens: Vec<TokenStream> = variants
        .iter()
        .map(|v| generate_enum_variant(v, structs, enums, types_crate))
        .collect();

    quote! {
        #[doc = #doc]
        #[derive(Debug, serde::Serialize)]
        pub enum #enum_name {
            #(#variant_tokens,)*
        }
    }
}

fn generate_enum_variant(
    v: &UnionVariant,
    structs: &[StructDef],
    enums: &[EnumDef],
    types_crate: Option<&syn::Path>,
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
                    let inner_ct = CapnpType::classify(inner, structs, enums);
                    if inner_ct.is_numeric() {
                        let rust_inner = rust_type_tokens(&inner_ct.rust_owned_type());
                        quote! { #variant_pascal(Vec<#rust_inner>) }
                    } else if structs.iter().any(|s| s.name == inner) {
                        let inner_data = format_ident!("{}", inner);
                        quote! { #variant_pascal(Vec<#inner_data>) }
                    } else {
                        quote! { #variant_pascal(Vec<String>) }
                    }
                }
            }
        }
        struct_name => {
            if let Some(s) = structs.iter().find(|s| s.name == struct_name) {
                if let Some(ref dt) = s.domain_type {
                    let domain_path = resolve_domain_type(dt, types_crate);
                    quote! { #variant_pascal(#domain_path) }
                } else {
                    let data_type = format_ident!("{}", struct_name);
                    quote! { #variant_pascal(#data_type) }
                }
            } else {
                quote! { #variant_pascal(Vec<u8>) }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Client Struct + Impl
// ─────────────────────────────────────────────────────────────────────────────

pub fn generate_client(service_name: &str, schema: &ParsedSchema, types_crate: Option<&syn::Path>) -> TokenStream {
    let pascal = to_pascal_case(service_name);
    let client_name = format_ident!("{}Client", pascal);
    let response_type = format_ident!("{}ResponseVariant", pascal);
    let capnp_mod_ident = format_ident!("{}_capnp", service_name);
    let capnp_mod: TokenStream = match types_crate {
        Some(tc) => quote! { #tc::#capnp_mod_ident },
        None => quote! { crate::#capnp_mod_ident },
    };
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
                Some(&schema.response_variants),
                false,
                types_crate,
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
        types_crate,
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
            pub fn signing_key(&self) -> &hyprstream_rpc::crypto::SigningKey {
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
///
/// When `response_variants` is provided, the method returns a typed result
/// (e.g., `Result<TrainStepResponseData>`) instead of the full response enum.
/// The pairing convention is:
/// - Top-level (`is_scoped=false`): request `foo` → response `fooResult`
/// - Scoped (`is_scoped=true`): request `foo` → response `foo`
pub fn generate_request_method(
    capnp_mod: &TokenStream,
    req_type: &syn::Ident,
    response_type: &syn::Ident,
    variant: &UnionVariant,
    structs: &[StructDef],
    enums: &[EnumDef],
    scope: Option<&ScopedMethodContext>,
    response_variants: Option<&[UnionVariant]>,
    is_scoped: bool,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let method_name = format_ident!("{}", to_snake_case(&variant.name));
    let doc = format!("{} ({} variant)", to_snake_case(&variant.name), variant.type_name);

    // For scoped methods, we wrap in the outer init
    let (outer_req_setup, inner_accessor, parse_call) = if let Some(sc) = scope {
        if let Some(ref parent_ctx) = sc.parent {
            // 3-level nested: parent init -> parent setters -> nested init -> nested setters
            let parent_init = format_ident!("init_{}", to_snake_case(&parent_ctx.factory_name));
            let parent_setters: Vec<TokenStream> = parent_ctx.scope_fields.iter().map(|f| {
                let setter_name = format_ident!("set_{}", to_snake_case(&f.name));
                let field_name = format_ident!("{}", to_snake_case(&f.name));
                match f.type_name.as_str() {
                    "Text" => quote! { mid.#setter_name(&self.#field_name); },
                    _ => quote! { mid.#setter_name(self.#field_name); },
                }
            }).collect();

            let nested_init = format_ident!("init_{}", to_snake_case(&sc.factory_name));
            let nested_setters: Vec<TokenStream> = sc.scope_fields.iter().map(|f| {
                let setter_name = format_ident!("set_{}", to_snake_case(&f.name));
                let field_name = format_ident!("{}", to_snake_case(&f.name));
                match f.type_name.as_str() {
                    "Text" => quote! { inner.#setter_name(&self.#field_name); },
                    _ => quote! { inner.#setter_name(self.#field_name); },
                }
            }).collect();

            let factory_snake = format_ident!("{}", to_snake_case(&sc.factory_name));

            (
                quote! {
                    let mut mid = req.#parent_init();
                    #(#parent_setters)*
                    let mut inner = mid.#nested_init();
                    #(#nested_setters)*
                },
                Some(factory_snake),
                quote! { Self::parse_scoped_response(&response) },
            )
        } else {
            // 2-level scoped (existing behavior)
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
        }
    } else {
        (TokenStream::new(), None, quote! { Self::parse_response(&response) })
    };

    // Determine typed return info if response_variants are available
    let typed_info = response_variants.and_then(|resp_vars| {
        find_typed_return_info(&variant.name, resp_vars, structs, enums, is_scoped, response_type, &parse_call, types_crate)
    });

    // Detect if this method returns StreamInfo (requires streaming call with ephemeral pubkey)
    let is_streaming = response_variants
        .and_then(|resp_vars| {
            let expected_name = if is_scoped {
                variant.name.clone()
            } else {
                format!("{}Result", variant.name)
            };
            resp_vars.iter().find(|v| v.name == expected_name)
        })
        .map(|v| v.type_name == "StreamInfo")
        .unwrap_or(false);

    // Generate return type and response handling
    let (return_type, response_handling) = if let Some(ref info) = typed_info {
        let ret = &info.return_type;
        let match_body = &info.match_body;
        (quote! { #ret }, quote! { #match_body })
    } else {
        (quote! { #response_type }, quote! { #parse_call })
    };

    // For streaming methods, add ephemeral_pubkey parameter and use call_with_options
    let (extra_param, call_expr) = if is_streaming {
        (
            quote! { , ephemeral_pubkey: [u8; 32] },
            quote! {
                let opts = CallOptions::default().ephemeral_pubkey(ephemeral_pubkey);
                let response = self.call_with_options(payload, opts).await?;
            },
        )
    } else {
        (
            TokenStream::new(),
            quote! { let response = self.call(payload).await?; },
        )
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
                pub async fn #method_name(&self #extra_param) -> anyhow::Result<#return_type> {
                    let __request_id = self.next_id();
                    let payload = hyprstream_rpc::serialize_message(|msg| {
                        let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                        req.set_id(__request_id);
                        #outer_req_setup
                        #setter
                    })?;
                    #call_expr
                    #response_handling
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
                pub async fn #method_name(&self, value: &str #extra_param) -> anyhow::Result<#return_type> {
                    let __request_id = self.next_id();
                    let payload = hyprstream_rpc::serialize_message(|msg| {
                        let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                        req.set_id(__request_id);
                        #outer_req_setup
                        #setter
                    })?;
                    #call_expr
                    #response_handling
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
                pub async fn #method_name(&self, value: &[u8] #extra_param) -> anyhow::Result<#return_type> {
                    let __request_id = self.next_id();
                    let payload = hyprstream_rpc::serialize_message(|msg| {
                        let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                        req.set_id(__request_id);
                        #outer_req_setup
                        #setter
                    })?;
                    #call_expr
                    #response_handling
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
                pub async fn #method_name(&self, value: bool #extra_param) -> anyhow::Result<#return_type> {
                    let __request_id = self.next_id();
                    let payload = hyprstream_rpc::serialize_message(|msg| {
                        let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                        req.set_id(__request_id);
                        #outer_req_setup
                        #setter
                    })?;
                    #call_expr
                    #response_handling
                }
            }
        }
        struct_name => {
            if let Some(s) = structs.iter().find(|s| s.name == struct_name) {
                let is_void_wrapper = s.fields.is_empty()
                    || (s.fields.len() == 1 && s.fields[0].type_name == "Void");

                if is_void_wrapper {
                    let init_method = format_ident!("init_{}", to_snake_case(&variant.name));
                    let setter = if inner_accessor.is_some() {
                        quote! { inner.#init_method(); }
                    } else {
                        quote! { req.#init_method(); }
                    };
                    quote! {
                        #[doc = #doc]
                        pub async fn #method_name(&self #extra_param) -> anyhow::Result<#return_type> {
                            let __request_id = self.next_id();
                            let payload = hyprstream_rpc::serialize_message(|msg| {
                                let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                                req.set_id(__request_id);
                                #outer_req_setup
                                #setter
                            })?;
                            #call_expr
                            #response_handling
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
                        pub async fn #method_name(&self #(, #params)* #extra_param) -> anyhow::Result<#return_type> {
                            let __request_id = self.next_id();
                            let payload = hyprstream_rpc::serialize_message(|msg| {
                                let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                                req.set_id(__request_id);
                                #outer_req_setup
                                #inner_init
                                #(#setters)*
                            })?;
                            #call_expr
                            #response_handling
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

// ─────────────────────────────────────────────────────────────────────────────
// Typed Return Info
// ─────────────────────────────────────────────────────────────────────────────

/// Information for generating a typed return from a request method.
struct TypedReturnInfo {
    /// The return type tokens (e.g., `()`, `bool`, `TrainStepResponseData`)
    return_type: TokenStream,
    /// The match body that extracts the typed value from the parse result
    match_body: TokenStream,
}

/// Find the paired response variant and generate typed return info.
///
/// Returns `None` if the pairing is ambiguous or the response variant isn't found,
/// in which case the method falls back to returning the full response enum.
fn find_typed_return_info(
    request_name: &str,
    response_variants: &[UnionVariant],
    structs: &[StructDef],
    enums: &[EnumDef],
    is_scoped: bool,
    response_type: &syn::Ident,
    parse_call: &TokenStream,
    types_crate: Option<&syn::Path>,
) -> Option<TypedReturnInfo> {
    // Find the expected response variant name
    let expected_name = if is_scoped {
        // Scoped: request `foo` → response `foo`
        request_name.to_string()
    } else {
        // Top-level: request `foo` → response `fooResult`
        format!("{}Result", request_name)
    };

    let resp_variant = response_variants.iter().find(|v| v.name == expected_name)?;
    let variant_pascal = format_ident!("{}", to_pascal_case(&resp_variant.name));
    let ct = CapnpType::classify(&resp_variant.type_name, structs, enums);

    // Check if there's an Error variant in the response (for the error arm)
    let has_error = response_variants.iter().any(|v| v.name == "error");

    let error_arm = if has_error {
        quote! {
            #response_type::Error(ref e) => Err(anyhow::anyhow!("{}", e.message)),
        }
    } else {
        TokenStream::new()
    };

    let wildcard_arm = quote! {
        _ => Err(anyhow::anyhow!("Unexpected response variant")),
    };

    match ct {
        CapnpType::Void => Some(TypedReturnInfo {
            return_type: quote! { () },
            match_body: quote! {
                match #parse_call? {
                    #response_type::#variant_pascal => Ok(()),
                    #error_arm
                    #wildcard_arm
                }
            },
        }),
        CapnpType::Bool => Some(TypedReturnInfo {
            return_type: quote! { bool },
            match_body: quote! {
                match #parse_call? {
                    #response_type::#variant_pascal(v) => Ok(v),
                    #error_arm
                    #wildcard_arm
                }
            },
        }),
        CapnpType::Text => Some(TypedReturnInfo {
            return_type: quote! { String },
            match_body: quote! {
                match #parse_call? {
                    #response_type::#variant_pascal(v) => Ok(v),
                    #error_arm
                    #wildcard_arm
                }
            },
        }),
        CapnpType::Data => Some(TypedReturnInfo {
            return_type: quote! { Vec<u8> },
            match_body: quote! {
                match #parse_call? {
                    #response_type::#variant_pascal(v) => Ok(v),
                    #error_arm
                    #wildcard_arm
                }
            },
        }),
        CapnpType::UInt8 | CapnpType::UInt16 | CapnpType::UInt32 | CapnpType::UInt64
        | CapnpType::Int8 | CapnpType::Int16 | CapnpType::Int32 | CapnpType::Int64
        | CapnpType::Float32 | CapnpType::Float64 => {
            let rust_type = rust_type_tokens(&ct.rust_owned_type());
            Some(TypedReturnInfo {
                return_type: quote! { #rust_type },
                match_body: quote! {
                    match #parse_call? {
                        #response_type::#variant_pascal(v) => Ok(v),
                        #error_arm
                        #wildcard_arm
                    }
                },
            })
        }
        CapnpType::ListText => Some(TypedReturnInfo {
            return_type: quote! { Vec<String> },
            match_body: quote! {
                match #parse_call? {
                    #response_type::#variant_pascal(v) => Ok(v),
                    #error_arm
                    #wildcard_arm
                }
            },
        }),
        CapnpType::ListData => Some(TypedReturnInfo {
            return_type: quote! { Vec<Vec<u8>> },
            match_body: quote! {
                match #parse_call? {
                    #response_type::#variant_pascal(v) => Ok(v),
                    #error_arm
                    #wildcard_arm
                }
            },
        }),
        CapnpType::ListPrimitive(ref inner) => {
            let rust_inner = rust_type_tokens(&inner.rust_owned_type());
            Some(TypedReturnInfo {
                return_type: quote! { Vec<#rust_inner> },
                match_body: quote! {
                    match #parse_call? {
                        #response_type::#variant_pascal(v) => Ok(v),
                        #error_arm
                        #wildcard_arm
                    }
                },
            })
        }
        CapnpType::ListStruct(ref inner) => {
            let data_name = format_ident!("{}", inner);
            Some(TypedReturnInfo {
                return_type: quote! { Vec<#data_name> },
                match_body: quote! {
                    match #parse_call? {
                        #response_type::#variant_pascal(v) => Ok(v),
                        #error_arm
                        #wildcard_arm
                    }
                },
            })
        }
        CapnpType::Struct(ref name) => {
            if let Some(s) = structs.iter().find(|s| s.name == *name) {
                let return_type = if let Some(ref dt) = s.domain_type {
                    resolve_domain_type(dt, types_crate)
                } else {
                    let data_name = format_ident!("{}", name);
                    quote! { #data_name }
                };
                Some(TypedReturnInfo {
                    return_type: return_type.clone(),
                    match_body: quote! {
                        match #parse_call? {
                            #response_type::#variant_pascal(v) => Ok(v),
                            #error_arm
                            #wildcard_arm
                        }
                    },
                })
            } else {
                None // Fallback to full enum
            }
        }
        _ => None, // Fallback to full enum for unknown types
    }
}

/// Context for scoped method generation.
///
/// For nested (3-level) scoping, `parent` is `Some(...)` and contains the
/// outer scope context. The request builder will chain: parent init -> parent setters ->
/// nested init -> nested setters -> operation init.
pub struct ScopedMethodContext {
    pub factory_name: String,
    pub scope_fields: Vec<FieldDef>,
    /// Parent scope context for 3-level nesting (e.g., Repository is parent of Fs).
    pub parent: Option<Box<ScopedMethodContext>>,
}

/// Generate method parameter tokens for a struct's fields.
///
/// Union-only struct fields (e.g., `EventTrigger`) are skipped since they have
/// no regular fields to parameterize — the setter just calls `init_*()`.
fn generate_method_params(
    fields: &[FieldDef],
    enums: &[EnumDef],
    structs: &[StructDef],
) -> Vec<TokenStream> {
    fields
        .iter()
        .filter(|field| {
            // Skip union-only struct fields (no settable fields to parameterize)
            if let CapnpType::Struct(ref name) = CapnpType::classify(&field.type_name, structs, enums) {
                if let Some(s) = structs.iter().find(|s| s.name == *name) {
                    if s.has_union && s.fields.is_empty() {
                        return false;
                    }
                }
            }
            true
        })
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
    capnp_mod: &TokenStream,
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
    capnp_mod: &TokenStream,
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
                    quote! { #snake => #capnp_mod::#type_ident::#pascal }
                }).collect();
                let default_arm = if let Some((first, _)) = e.variants.first() {
                    let first_pascal = format_ident!("{}", to_pascal_case(first));
                    quote! { _ => #capnp_mod::#type_ident::#first_pascal }
                } else {
                    TokenStream::new()
                };
                quote! {
                    inner.#setter_name(match #rust_name {
                        #(#match_arms,)*
                        #default_arm,
                    });
                }
            } else if let Some(s) = structs.iter().find(|s| s.name == t) {
                let field_snake = to_snake_case(
                    setter_name.to_string().strip_prefix("set_").unwrap_or(&setter_name.to_string())
                );
                if s.has_union && s.fields.is_empty() {
                    // Union-only struct (e.g., EventTrigger) — init without setting fields
                    let init_name = format_ident!("init_{}", &field_snake);
                    quote! { inner.reborrow().#init_name(); }
                } else {
                    let init_name = format_ident!("init_{}", &field_snake);
                    quote! {
                        hyprstream_rpc::capnp::ToCapnp::write_to(&#rust_name, &mut inner.reborrow().#init_name());
                    }
                }
            } else {
                let _comment = format!("unknown type: set_{} for {}", to_snake_case(&setter_name.to_string()), t);
                quote! { /* #_comment */ }
            }
        }
    }
}

fn generate_list_setter(
    rust_name: &syn::Ident,
    setter_name: &syn::Ident,
    inner_type: &str,
    _enums: &[EnumDef],
    structs: &[StructDef],
    _capnp_mod: &TokenStream,
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
        prim if matches!(prim, "Bool" | "UInt8" | "UInt16" | "UInt32" | "UInt64"
            | "Int8" | "Int16" | "Int32" | "Int64" | "Float32" | "Float64") => {
            quote! {
                {
                    let mut list = inner.reborrow().#init_name(#rust_name.len() as u32);
                    for (i, item) in #rust_name.iter().enumerate() {
                        list.set(i as u32, *item);
                    }
                }
            }
        }
        struct_name => {
            if structs.iter().any(|s| s.name == struct_name) {
                quote! {
                    {
                        let mut list = inner.reborrow().#init_name(#rust_name.len() as u32);
                        for (i, item) in #rust_name.iter().enumerate() {
                            hyprstream_rpc::capnp::ToCapnp::write_to(item, &mut list.reborrow().get(i as u32));
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

// ─────────────────────────────────────────────────────────────────────────────
// Nested Struct Setter Helpers
// ─────────────────────────────────────────────────────────────────────────────
// parse_response
// ─────────────────────────────────────────────────────────────────────────────

pub fn generate_parse_response_fn(
    response_type: &syn::Ident,
    capnp_mod: &TokenStream,
    resp_type: &syn::Ident,
    variants: &[UnionVariant],
    structs: &[StructDef],
    enums: &[EnumDef],
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let which_ident = format_ident!("Which");
    let match_arms: Vec<TokenStream> = variants
        .iter()
        .map(|v| generate_parse_match_arm(response_type, capnp_mod, v, structs, enums, &which_ident, types_crate))
        .collect();

    quote! {
        /// Parse a response from raw bytes.
        pub fn parse_response(bytes: &[u8]) -> anyhow::Result<#response_type> {
            let reader = capnp::serialize::read_message(
                &mut std::io::Cursor::new(bytes),
                capnp::message::ReaderOptions::new(),
            )?;
            let resp = reader.get_root::<#capnp_mod::#resp_type::Reader>()?;
            use #capnp_mod::#resp_type::Which;
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
    capnp_mod: &TokenStream,
    v: &UnionVariant,
    structs: &[StructDef],
    enums: &[EnumDef],
    which_ident: &syn::Ident,
    types_crate: Option<&syn::Path>,
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
        CapnpType::ListText | CapnpType::ListData | CapnpType::ListPrimitive(_) | CapnpType::ListStruct(_) => {
            generate_list_parse_arm(&variant_pascal, response_type, &v.type_name, structs, enums, capnp_mod, which_ident)
        }
        CapnpType::Struct(ref name) => {
            if let Some(s) = structs.iter().find(|s| s.name == *name) {
                let data_type = if let Some(ref dt) = s.domain_type {
                    resolve_domain_type(dt, types_crate)
                } else {
                    let ident = format_ident!("{}", name);
                    quote! { #ident }
                };
                quote! {
                    #which_ident::#variant_pascal(v) => {
                        let v = v?;
                        Ok(#response_type::#variant_pascal(
                            <#data_type as hyprstream_rpc::FromCapnp>::read_from(v)?
                        ))
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
    _capnp_mod: &TokenStream,
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
        CapnpType::ListPrimitive(_) => quote! {
            #which_ident::#variant_pascal(v) => {
                let list = v?;
                let result: Vec<_> = list.iter().collect();
                Ok(#response_type::#variant_pascal(result))
            }
        },
        CapnpType::ListStruct(ref inner) => {
            if structs.iter().any(|s| s.name == *inner) {
                quote! {
                    #which_ident::#variant_pascal(v) => {
                        let list = v?;
                        let mut result = Vec::with_capacity(list.len() as usize);
                        for i in 0..list.len() {
                            result.push(hyprstream_rpc::capnp::FromCapnp::read_from(list.get(i))?);
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
