//! Client struct, response enum, request methods, and parse_response generation.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::resolve::ResolvedSchema;
use crate::schema::types::*;
use crate::util::*;

// ─────────────────────────────────────────────────────────────────────────────
// Domain Type Resolution
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve a domain type path from a `$domainType` annotation value.
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

pub fn generate_response_enum(service_name: &str, resolved: &ResolvedSchema, types_crate: Option<&syn::Path>) -> TokenStream {
    let pascal = to_pascal_case(service_name);
    let enum_name_str = format!("{pascal}ResponseVariant");
    generate_response_enum_from_variants(
        &enum_name_str,
        &resolved.raw.response_variants,
        resolved,
        types_crate,
    )
}

pub fn generate_response_enum_from_variants(
    enum_name_str: &str,
    variants: &[UnionVariant],
    resolved: &ResolvedSchema,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let enum_name = format_ident!("{}", enum_name_str);
    let doc = format!("Response variants for the {} service.", enum_name_str);

    let variant_tokens: Vec<TokenStream> = variants
        .iter()
        .map(|v| generate_enum_variant(v, resolved, types_crate))
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
    resolved: &ResolvedSchema,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let variant_pascal = resolved.name(&v.name).pascal_ident.clone();

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
                    let rt = resolved.resolve_type(inner);
                    if rt.is_numeric {
                        let rust_inner = rust_type_tokens(&rt.rust_owned);
                        quote! { #variant_pascal(Vec<#rust_inner>) }
                    } else if resolved.is_struct(inner) {
                        let inner_data = format_ident!("{}", inner);
                        quote! { #variant_pascal(Vec<#inner_data>) }
                    } else {
                        quote! { #variant_pascal(Vec<String>) }
                    }
                }
            }
        }
        struct_name => {
            if let Some(s) = resolved.find_struct(struct_name) {
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

pub fn generate_client(service_name: &str, resolved: &ResolvedSchema, types_crate: Option<&syn::Path>) -> TokenStream {
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

    let scoped_variant_names: Vec<&str> = resolved.raw
        .scoped_clients
        .iter()
        .map(|sc| sc.factory_name.as_str())
        .collect();

    // Request methods (skip scoped variants)
    let request_methods: Vec<TokenStream> = resolved.raw
        .request_variants
        .iter()
        .filter(|v| !scoped_variant_names.contains(&v.name.as_str()))
        .map(|v| {
            generate_request_method(
                &capnp_mod,
                &req_type,
                &response_type,
                v,
                resolved,
                None,
                Some(&resolved.raw.response_variants),
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
        &resolved.raw.response_variants,
        resolved,
        types_crate,
    );

    // Factory methods for scoped clients
    let factory_methods: Vec<TokenStream> = resolved.raw
        .scoped_clients
        .iter()
        .map(generate_scoped_factory_method)
        .collect();

    // ServiceClient impl
    let service_name_lit = service_name;

    quote! {
        #[doc = #doc]
        #[derive(Clone)]
        pub struct #client_name {
            client: Arc<ZmqClientBase>,
            claims: Option<std::sync::Arc<hyprstream_rpc::auth::Claims>>,
        }

        impl ServiceClient for #client_name {
            const SERVICE_NAME: &'static str = #service_name_lit;

            fn from_zmq(client: ZmqClientBase) -> Self {
                Self { client: Arc::new(client), claims: None }
            }
        }

        impl #client_name {
            /// Get the next request ID.
            pub fn next_id(&self) -> u64 {
                self.client.next_id()
            }

            /// Attach claims for e2e verification. All subsequent calls include these claims.
            pub fn with_claims(mut self, claims: hyprstream_rpc::auth::Claims) -> Self {
                self.claims = Some(std::sync::Arc::new(claims));
                self
            }

            /// Send a raw request and return the raw response bytes.
            pub async fn call(&self, payload: Vec<u8>) -> anyhow::Result<Vec<u8>> {
                let opts = match &self.claims {
                    Some(c) => CallOptions::default().claims((**c).clone()),
                    None => CallOptions::default(),
                };
                self.client.call(payload, opts).await
            }

            /// Send a raw request with custom options and return the raw response bytes.
            pub async fn call_with_options(&self, payload: Vec<u8>, mut opts: CallOptions) -> anyhow::Result<Vec<u8>> {
                if opts.claims.is_none() {
                    if let Some(ref c) = self.claims {
                        opts = opts.claims((**c).clone());
                    }
                }
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
#[allow(clippy::too_many_arguments)]
pub fn generate_request_method(
    capnp_mod: &TokenStream,
    req_type: &syn::Ident,
    response_type: &syn::Ident,
    variant: &UnionVariant,
    resolved: &ResolvedSchema,
    scope: Option<&ScopedMethodContext>,
    response_variants: Option<&[UnionVariant]>,
    is_scoped: bool,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let method_name = format_ident!("{}", to_snake_case(&variant.name));
    let doc = format!("{} ({} variant)", to_snake_case(&variant.name), variant.type_name);

    // For scoped methods, walk the ScopedMethodContext chain and shadow `req`
    let (outer_req_setup, parse_call) = if let Some(sc) = scope {
        let mut chain = Vec::new();
        let mut cur = Some(sc);
        while let Some(c) = cur {
            chain.push(c);
            cur = c.parent.as_deref();
        }
        chain.reverse();

        let mut setup = TokenStream::new();
        let mut prev_var = format_ident!("req");

        for (i, level) in chain.iter().enumerate() {
            let init_fn = format_ident!("init_{}", to_snake_case(&level.factory_name));
            let tmp = format_ident!("__s{}", i);

            setup.extend(quote! { let mut #tmp = #prev_var.#init_fn(); });

            for f in &level.scope_fields {
                let setter = format_ident!("set_{}", to_snake_case(&f.name));
                let field = format_ident!("{}", to_snake_case(&f.name));
                setup.extend(match f.type_name.as_str() {
                    "Text" => quote! { #tmp.#setter(&self.#field); },
                    _ => quote! { #tmp.#setter(self.#field); },
                });
            }
            prev_var = tmp;
        }

        setup.extend(quote! { let mut req = #prev_var; });

        (setup, quote! { Self::parse_scoped_response(&response) })
    } else {
        (TokenStream::new(), quote! { Self::parse_response(&response) })
    };

    // Determine typed return info if response_variants are available
    let typed_info = response_variants.and_then(|resp_vars| {
        find_typed_return_info(&variant.name, resp_vars, resolved, is_scoped, response_type, &parse_call, types_crate)
    });

    // Detect if this method returns StreamInfo
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

    let (return_type, response_handling) = if let Some(ref info) = typed_info {
        let ret = &info.return_type;
        let match_body = &info.match_body;
        (quote! { #ret }, quote! { #match_body })
    } else {
        (quote! { #response_type }, quote! { #parse_call })
    };

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
            let setter = quote! { req.#set_method(()); };
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
            let setter = quote! { req.#set_method(value); };
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
            let setter = quote! { req.#set_method(value); };
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
            let setter = quote! { req.#set_method(value); };
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
            if let Some(s) = resolved.find_struct(struct_name) {
                let is_void_wrapper = s.fields.is_empty()
                    || (s.fields.len() == 1 && s.fields[0].type_name == "Void");

                if is_void_wrapper {
                    let init_method = format_ident!("init_{}", to_snake_case(&variant.name));
                    let setter = quote! { req.#init_method(); };
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
                    let params = generate_method_params(&s.fields, resolved);
                    let init_method = format_ident!("init_{}", to_snake_case(&variant.name));
                    let builder_var = format_ident!("req");
                    let setters = generate_struct_setters(&s.fields, resolved, capnp_mod, &builder_var);

                    let inner_init = quote! { let mut req = req.#init_method(); };

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

struct TypedReturnInfo {
    return_type: TokenStream,
    match_body: TokenStream,
}

fn find_typed_return_info(
    request_name: &str,
    response_variants: &[UnionVariant],
    resolved: &ResolvedSchema,
    is_scoped: bool,
    response_type: &syn::Ident,
    parse_call: &TokenStream,
    types_crate: Option<&syn::Path>,
) -> Option<TypedReturnInfo> {
    let expected_name = if is_scoped {
        request_name.to_owned()
    } else {
        format!("{}Result", request_name)
    };

    let resp_variant = response_variants.iter().find(|v| v.name == expected_name)?;
    let variant_pascal = resolved.name(&resp_variant.name).pascal_ident.clone();
    let ct = resolved.resolve_type(&resp_variant.type_name).capnp_type.clone();

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
            if let Some(s) = resolved.find_struct(name) {
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
                None
            }
        }
        _ => None,
    }
}

/// Context for scoped method generation.
pub struct ScopedMethodContext {
    pub factory_name: String,
    pub scope_fields: Vec<FieldDef>,
    pub parent: Option<Box<ScopedMethodContext>>,
}

/// Generate method parameter tokens for a struct's fields.
fn generate_method_params(
    fields: &[FieldDef],
    resolved: &ResolvedSchema,
) -> Vec<TokenStream> {
    fields
        .iter()
        .filter(|field| {
            if let CapnpType::Struct(ref name) = resolved.resolve_type(&field.type_name).capnp_type {
                if let Some(s) = resolved.find_struct(name) {
                    if s.has_union && s.fields.is_empty() {
                        return false;
                    }
                }
            }
            true
        })
        .map(|field| {
            let rust_name = format_ident!("{}", to_snake_case(&field.name));
            let type_str = &resolved.resolve_type(&field.type_name).rust_param;
            let rust_type = rust_type_tokens(type_str);
            quote! { #rust_name: #rust_type }
        })
        .collect()
}

/// Generate setter calls for fields when building a request struct.
fn generate_struct_setters(
    fields: &[FieldDef],
    resolved: &ResolvedSchema,
    capnp_mod: &TokenStream,
    builder_var: &syn::Ident,
) -> Vec<TokenStream> {
    fields
        .iter()
        .map(|field| {
            let rust_name = format_ident!("{}", to_snake_case(&field.name));
            let setter_name = format_ident!("set_{}", to_snake_case(&field.name));
            generate_field_setter(&rust_name, &setter_name, &field.type_name, resolved, capnp_mod, builder_var)
        })
        .collect()
}

fn generate_field_setter(
    rust_name: &syn::Ident,
    setter_name: &syn::Ident,
    type_name: &str,
    resolved: &ResolvedSchema,
    capnp_mod: &TokenStream,
    builder_var: &syn::Ident,
) -> TokenStream {
    match type_name {
        "Text" | "Data" => quote! { #builder_var.#setter_name(#rust_name); },
        "Bool" | "UInt8" | "UInt16" | "UInt32" | "UInt64" | "Int8" | "Int16" | "Int32" | "Int64" | "Float32" | "Float64" => {
            quote! { #builder_var.#setter_name(#rust_name); }
        }
        t if t.starts_with("List(") => {
            let inner_type = &t[5..t.len() - 1];
            generate_list_setter(rust_name, setter_name, inner_type, resolved, capnp_mod, builder_var)
        }
        t => {
            if let Some(e) = resolved.find_enum(t) {
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
                    #builder_var.#setter_name(match #rust_name {
                        #(#match_arms,)*
                        #default_arm,
                    });
                }
            } else if let Some(s) = resolved.find_struct(t) {
                let field_snake = to_snake_case(
                    setter_name.to_string().strip_prefix("set_").unwrap_or(&setter_name.to_string())
                );
                if s.has_union && s.fields.is_empty() {
                    let init_name = format_ident!("init_{}", &field_snake);
                    quote! { #builder_var.reborrow().#init_name(); }
                } else {
                    let init_name = format_ident!("init_{}", &field_snake);
                    quote! {
                        hyprstream_rpc::capnp::ToCapnp::write_to(&#rust_name, &mut #builder_var.reborrow().#init_name());
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
    resolved: &ResolvedSchema,
    _capnp_mod: &TokenStream,
    builder_var: &syn::Ident,
) -> TokenStream {
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
                    let mut list = #builder_var.reborrow().#init_name(#rust_name.len() as u32);
                    for (i, item) in #rust_name.iter().enumerate() {
                        list.set(i as u32, item.as_str());
                    }
                }
            }
        }
        "Data" => {
            quote! {
                {
                    let mut list = #builder_var.reborrow().#init_name(#rust_name.len() as u32);
                    for (i, item) in #rust_name.iter().enumerate() {
                        list.set(i as u32, item.as_slice());
                    }
                }
            }
        }
        "Bool" | "UInt8" | "UInt16" | "UInt32" | "UInt64"
            | "Int8" | "Int16" | "Int32" | "Int64" | "Float32" | "Float64" => {
            quote! {
                {
                    let mut list = #builder_var.reborrow().#init_name(#rust_name.len() as u32);
                    for (i, item) in #rust_name.iter().enumerate() {
                        list.set(i as u32, *item);
                    }
                }
            }
        }
        struct_name => {
            if resolved.is_struct(struct_name) {
                quote! {
                    {
                        let mut list = #builder_var.reborrow().#init_name(#rust_name.len() as u32);
                        for (i, item) in #rust_name.iter().enumerate() {
                            hyprstream_rpc::capnp::ToCapnp::write_to(item, &mut list.reborrow().get(i as u32));
                        }
                    }
                }
            } else {
                quote! { }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_response
// ─────────────────────────────────────────────────────────────────────────────

pub fn generate_parse_response_fn(
    response_type: &syn::Ident,
    capnp_mod: &TokenStream,
    resp_type: &syn::Ident,
    variants: &[UnionVariant],
    resolved: &ResolvedSchema,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let which_ident = format_ident!("Which");
    let match_arms: Vec<TokenStream> = variants
        .iter()
        .map(|v| generate_parse_match_arm(response_type, capnp_mod, v, resolved, &which_ident, types_crate))
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

pub fn generate_parse_match_arm(
    response_type: &syn::Ident,
    capnp_mod: &TokenStream,
    v: &UnionVariant,
    resolved: &ResolvedSchema,
    which_ident: &syn::Ident,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let variant_pascal = resolved.name(&v.name).pascal_ident.clone();
    let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();

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
            generate_list_parse_arm(&variant_pascal, response_type, &v.type_name, resolved, capnp_mod, which_ident)
        }
        CapnpType::Struct(ref name) => {
            if let Some(s) = resolved.find_struct(name) {
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
    resolved: &ResolvedSchema,
    _capnp_mod: &TokenStream,
    which_ident: &syn::Ident,
) -> TokenStream {
    let ct = resolved.resolve_type(type_name).capnp_type.clone();
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
            if resolved.is_struct(inner) {
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
                claims: self.claims.clone(),
                #(#field_inits,)*
            }
        }
    }
}
