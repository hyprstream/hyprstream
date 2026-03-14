//! Client struct, response enum, request methods, parse_response, and trait generation.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::resolve::ResolvedSchema;
use crate::schema::types::*;
use crate::util::*;

/// Returns `true` if `type_name` is the canonical streaming `StreamInfo` from
/// `streaming.capnp`, identified by `origin_file == Some("streaming")`.
///
/// This guards against false positives when a service schema defines its own
/// `StreamInfo` struct with unrelated semantics (e.g., `tui.capnp`).
fn is_rpc_stream_info(type_name: &str, resolved: &ResolvedSchema) -> bool {
    type_name == "StreamInfo"
        && resolved
            .find_struct("StreamInfo")
            .map(|s| s.origin_file.as_deref() == Some("streaming"))
            .unwrap_or(false)
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait Generation (Phase 2a)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate the top-level service trait and all scope traits.
pub fn generate_service_traits(service_name: &str, resolved: &ResolvedSchema, types_crate: Option<&syn::Path>) -> TokenStream {
    let mut tokens = TokenStream::new();

    // Generate scope traits first (bottom-up: nested → parent → top-level)
    for sc in &resolved.raw.scoped_clients {
        tokens.extend(generate_scope_trait_recursive(sc, resolved, types_crate));
    }

    // Generate top-level service trait
    tokens.extend(generate_top_level_trait(service_name, resolved, types_crate));

    tokens
}

/// Generate a scope trait (e.g., RuntimeRpc, ContainerRpc) and recurse into nested scopes.
fn generate_scope_trait_recursive(sc: &ScopedClient, resolved: &ResolvedSchema, types_crate: Option<&syn::Path>) -> TokenStream {
    let mut tokens = TokenStream::new();

    // Generate nested scope traits first
    for nested in &sc.nested_clients {
        tokens.extend(generate_scope_trait_recursive(nested, resolved, types_crate));
    }

    let trait_name = scope_trait_name(&sc.client_name);
    let trait_ident = format_ident!("{}", trait_name);
    let doc = format!("Generated RPC trait for {} scope.", sc.factory_name);

    // Nested scope associated types + factory methods
    let nested_assoc_types: Vec<TokenStream> = sc.nested_clients.iter().map(|nested| {
        let assoc_name = format_ident!("{}", to_pascal_case(&nested.factory_name));
        let nested_trait = format_ident!("{}", scope_trait_name(&nested.client_name));
        quote! { type #assoc_name: #nested_trait; }
    }).collect();

    let nested_factory_methods: Vec<TokenStream> = sc.nested_clients.iter().map(|nested| {
        let method = format_ident!("{}", to_snake_case(&nested.factory_name));
        let assoc_name = format_ident!("{}", to_pascal_case(&nested.factory_name));
        let params = scope_factory_params(&nested.scope_fields, resolved);
        quote! { fn #method(&self #(, #params)*) -> Self::#assoc_name; }
    }).collect();

    // Trait methods from inner_request_variants
    let trait_methods: Vec<TokenStream> = sc.inner_request_variants.iter().filter_map(|v| {
        generate_trait_method(v, &sc.inner_response_variants, resolved, types_crate, true)
    }).collect();

    tokens.extend(quote! {
        #[doc = #doc]
        #[async_trait::async_trait]
        pub trait #trait_ident: Send + Sync {
            #(#nested_assoc_types)*
            #(#nested_factory_methods)*
            #(#trait_methods)*
        }
    });

    tokens
}

/// Generate the top-level service trait (e.g., WorkerRpc, ModelRpc).
fn generate_top_level_trait(service_name: &str, resolved: &ResolvedSchema, types_crate: Option<&syn::Path>) -> TokenStream {
    let pascal = to_pascal_case(service_name);
    let trait_name = format_ident!("{}Rpc", pascal);
    let doc = format!("Generated RPC trait for the {pascal} service.");

    let scoped_variant_names: Vec<&str> = resolved.raw
        .scoped_clients
        .iter()
        .map(|sc| sc.factory_name.as_str())
        .collect();

    // Associated types for scoped clients
    let assoc_types: Vec<TokenStream> = resolved.raw.scoped_clients.iter().map(|sc| {
        let assoc_name = format_ident!("{}", to_pascal_case(&sc.factory_name));
        let scope_trait = format_ident!("{}", scope_trait_name(&sc.client_name));
        quote! { type #assoc_name: #scope_trait; }
    }).collect();

    // Factory methods for scoped clients
    let factory_methods: Vec<TokenStream> = resolved.raw.scoped_clients.iter().map(|sc| {
        let method = format_ident!("{}", to_snake_case(&sc.factory_name));
        let assoc_name = format_ident!("{}", to_pascal_case(&sc.factory_name));
        let params = scope_factory_params(&sc.scope_fields, resolved);
        quote! { fn #method(&self #(, #params)*) -> Self::#assoc_name; }
    }).collect();

    // Non-scoped methods
    let trait_methods: Vec<TokenStream> = resolved.raw.request_variants.iter()
        .filter(|v| !scoped_variant_names.contains(&v.name.as_str()))
        .filter_map(|v| {
            generate_trait_method(v, &resolved.raw.response_variants, resolved, types_crate, false)
        })
        .collect();

    quote! {
        #[doc = #doc]
        #[async_trait::async_trait]
        pub trait #trait_name: Send + Sync {
            #(#assoc_types)*
            #(#factory_methods)*
            #(#trait_methods)*
        }
    }
}

/// Generate a single trait method signature from a request variant.
fn generate_trait_method(
    variant: &UnionVariant,
    response_variants: &[UnionVariant],
    resolved: &ResolvedSchema,
    types_crate: Option<&syn::Path>,
    is_scoped: bool,
) -> Option<TokenStream> {
    let method_name = format_ident!("{}", to_snake_case(&variant.name));

    // Determine return type
    let return_type = determine_return_type(&variant.name, response_variants, resolved, is_scoped, types_crate)?;

    // Detect streaming — trait methods return StreamHandle directly (no ephemeral_pubkey param)
    let is_streaming = response_variants
        .iter()
        .find(|v| {
            let expected = if is_scoped { variant.name.clone() } else { format!("{}Result", variant.name) };
            v.name == expected
        })
        .map(|v| is_rpc_stream_info(&v.type_name, resolved))
        .unwrap_or(false);

    // Streaming methods in the trait don't take ephemeral_pubkey — they return StreamHandle
    let return_type = if is_streaming {
        quote! { hyprstream_rpc::streaming::StreamHandle }
    } else {
        return_type
    };

    // Determine params from the request variant type
    let params = match variant.type_name.as_str() {
        "Void" => Vec::new(),
        "Text" => vec![quote! { value: &str }],
        "Data" => vec![quote! { value: &[u8] }],
        "Bool" => vec![quote! { value: bool }],
        t if CapnpType::classify_primitive(t).is_numeric() => {
            let ty = rust_type_tokens(&CapnpType::classify_primitive(t).rust_owned_type());
            vec![quote! { value: #ty }]
        }
        struct_name => {
            if let Some(s) = resolved.find_struct(struct_name) {
                let nuf: Vec<_> = s.non_union_fields().collect();
                if nuf.is_empty() || (nuf.len() == 1 && nuf[0].type_name == "Void") {
                    Vec::new()
                } else {
                    let data_name = format_ident!("{}", struct_name);
                    vec![quote! { request: &#data_name }]
                }
            } else {
                return None;
            }
        }
    };

    Some(quote! {
        async fn #method_name(&self #(, #params)*) -> anyhow::Result<#return_type>;
    })
}

/// Determine the return type for a trait method.
fn determine_return_type(
    request_name: &str,
    response_variants: &[UnionVariant],
    resolved: &ResolvedSchema,
    is_scoped: bool,
    types_crate: Option<&syn::Path>,
) -> Option<TokenStream> {
    let expected_name = if is_scoped {
        request_name.to_owned()
    } else {
        format!("{}Result", request_name)
    };

    let resp_variant = response_variants.iter().find(|v| v.name == expected_name)?;
    let ct = resolved.resolve_type(&resp_variant.type_name).capnp_type.clone();

    Some(match ct {
        CapnpType::Void => quote! { () },
        CapnpType::Bool => quote! { bool },
        CapnpType::Text => quote! { String },
        CapnpType::Data => quote! { Vec<u8> },
        CapnpType::UInt8 | CapnpType::UInt16 | CapnpType::UInt32 | CapnpType::UInt64
        | CapnpType::Int8 | CapnpType::Int16 | CapnpType::Int32 | CapnpType::Int64
        | CapnpType::Float32 | CapnpType::Float64 => {
            let rust_type = rust_type_tokens(&ct.rust_owned_type());
            quote! { #rust_type }
        }
        CapnpType::ListText => quote! { Vec<String> },
        CapnpType::ListData => quote! { Vec<Vec<u8>> },
        CapnpType::ListPrimitive(ref inner) => {
            let rust_inner = rust_type_tokens(&inner.rust_owned_type());
            quote! { Vec<#rust_inner> }
        }
        CapnpType::ListStruct(ref inner) => {
            let data_name = format_ident!("{}", inner);
            quote! { Vec<#data_name> }
        }
        CapnpType::Struct(ref name) => {
            if let Some(s) = resolved.find_struct(name) {
                if let Some(ref dt) = s.domain_type {
                    resolve_domain_type(dt, types_crate)
                } else {
                    let data_name = format_ident!("{}", name);
                    quote! { #data_name }
                }
            } else {
                return None;
            }
        }
        _ => return None,
    })
}

/// Derive a scope trait name from a scoped client name.
/// e.g., "RuntimeClient" → "RuntimeRpc", "ContainerClient" → "ContainerRpc"
fn scope_trait_name(client_name: &str) -> String {
    if let Some(base) = client_name.strip_suffix("Client") {
        format!("{}Rpc", base)
    } else {
        format!("{}Rpc", client_name)
    }
}

/// Generate scope factory method parameters.
fn scope_factory_params(scope_fields: &[FieldDef], _resolved: &ResolvedSchema) -> Vec<TokenStream> {
    scope_fields.iter().map(|f| {
        let name = format_ident!("{}", to_snake_case(&f.name));
        let ty = rust_type_tokens(&CapnpType::classify_primitive(&f.type_name).rust_param_type());
        quote! { #name: #ty }
    }).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait Implementation Generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate trait impl for the top-level client + all scoped clients.
pub fn generate_trait_impls(service_name: &str, resolved: &ResolvedSchema, types_crate: Option<&syn::Path>) -> TokenStream {
    let mut tokens = TokenStream::new();

    let pascal = to_pascal_case(service_name);
    let client_ident = format_ident!("{}Client", pascal);
    let trait_ident = format_ident!("{}Rpc", pascal);

    let scoped_variant_names: Vec<&str> = resolved.raw
        .scoped_clients
        .iter()
        .map(|sc| sc.factory_name.as_str())
        .collect();

    // Associated type bindings
    let assoc_bindings: Vec<TokenStream> = resolved.raw.scoped_clients.iter().map(|sc| {
        let assoc_name = format_ident!("{}", to_pascal_case(&sc.factory_name));
        let scoped_client = format_ident!("{}", sc.client_name);
        quote! { type #assoc_name = #scoped_client; }
    }).collect();

    // Factory method impls
    let factory_impls: Vec<TokenStream> = resolved.raw.scoped_clients.iter().map(|sc| {
        let method = format_ident!("{}", to_snake_case(&sc.factory_name));
        let assoc_name = format_ident!("{}", to_pascal_case(&sc.factory_name));
        let params = scope_factory_params(&sc.scope_fields, resolved);
        let param_names: Vec<syn::Ident> = sc.scope_fields.iter().map(|f| {
            format_ident!("{}", to_snake_case(&f.name))
        }).collect();
        quote! {
            fn #method(&self #(, #params)*) -> Self::#assoc_name {
                #client_ident::#method(self #(, #param_names),*)
            }
        }
    }).collect();

    // Non-scoped method delegation
    let method_impls: Vec<TokenStream> = resolved.raw.request_variants.iter()
        .filter(|v| !scoped_variant_names.contains(&v.name.as_str()))
        .filter_map(|v| {
            generate_trait_method_impl(v, &resolved.raw.response_variants, resolved, types_crate, false)
        })
        .collect();

    tokens.extend(quote! {
        #[async_trait::async_trait]
        impl #trait_ident for #client_ident {
            #(#assoc_bindings)*
            #(#factory_impls)*
            #(#method_impls)*
        }
    });

    // Scoped client trait impls
    for sc in &resolved.raw.scoped_clients {
        tokens.extend(generate_scope_trait_impl_recursive(sc, resolved, types_crate));
    }

    tokens
}

fn generate_scope_trait_impl_recursive(sc: &ScopedClient, resolved: &ResolvedSchema, types_crate: Option<&syn::Path>) -> TokenStream {
    let mut tokens = TokenStream::new();

    let client_ident = format_ident!("{}", sc.client_name);
    let trait_ident = format_ident!("{}", scope_trait_name(&sc.client_name));

    // Nested associated type bindings + factory impls
    let nested_assoc: Vec<TokenStream> = sc.nested_clients.iter().map(|nested| {
        let assoc_name = format_ident!("{}", to_pascal_case(&nested.factory_name));
        let nested_client = format_ident!("{}", nested.client_name);
        quote! { type #assoc_name = #nested_client; }
    }).collect();

    let nested_factory_impls: Vec<TokenStream> = sc.nested_clients.iter().map(|nested| {
        let method = format_ident!("{}", to_snake_case(&nested.factory_name));
        let assoc_name = format_ident!("{}", to_pascal_case(&nested.factory_name));
        let params = scope_factory_params(&nested.scope_fields, resolved);
        let param_names: Vec<syn::Ident> = nested.scope_fields.iter().map(|f| {
            format_ident!("{}", to_snake_case(&f.name))
        }).collect();
        quote! {
            fn #method(&self #(, #params)*) -> Self::#assoc_name {
                #client_ident::#method(self #(, #param_names),*)
            }
        }
    }).collect();

    let method_impls: Vec<TokenStream> = sc.inner_request_variants.iter()
        .filter_map(|v| {
            generate_trait_method_impl(v, &sc.inner_response_variants, resolved, types_crate, true)
        })
        .collect();

    tokens.extend(quote! {
        #[async_trait::async_trait]
        impl #trait_ident for #client_ident {
            #(#nested_assoc)*
            #(#nested_factory_impls)*
            #(#method_impls)*
        }
    });

    // Recurse into nested scoped clients
    for nested in &sc.nested_clients {
        tokens.extend(generate_scope_trait_impl_recursive(nested, resolved, types_crate));
    }

    tokens
}

/// Generate a trait method impl that delegates to the inherent method on the client.
fn generate_trait_method_impl(
    variant: &UnionVariant,
    response_variants: &[UnionVariant],
    resolved: &ResolvedSchema,
    types_crate: Option<&syn::Path>,
    is_scoped: bool,
) -> Option<TokenStream> {
    let method_name = format_ident!("{}", to_snake_case(&variant.name));

    let return_type = determine_return_type(&variant.name, response_variants, resolved, is_scoped, types_crate)?;

    let is_streaming = response_variants
        .iter()
        .find(|v| {
            let expected = if is_scoped { variant.name.clone() } else { format!("{}Result", variant.name) };
            v.name == expected
        })
        .map(|v| is_rpc_stream_info(&v.type_name, resolved))
        .unwrap_or(false);

    // Streaming trait methods return StreamHandle (no ephemeral_pubkey param)
    let return_type = if is_streaming {
        quote! { hyprstream_rpc::streaming::StreamHandle }
    } else {
        return_type
    };

    // Build params and args
    let (params, args) = match variant.type_name.as_str() {
        "Void" => (Vec::new(), Vec::new()),
        "Text" => (vec![quote! { value: &str }], vec![quote! { value }]),
        "Data" => (vec![quote! { value: &[u8] }], vec![quote! { value }]),
        "Bool" => (vec![quote! { value: bool }], vec![quote! { value }]),
        t if CapnpType::classify_primitive(t).is_numeric() => {
            let ty = rust_type_tokens(&CapnpType::classify_primitive(t).rust_owned_type());
            (vec![quote! { value: #ty }], vec![quote! { value }])
        }
        struct_name => {
            if let Some(s) = resolved.find_struct(struct_name) {
                let nuf: Vec<_> = s.non_union_fields().collect();
                if nuf.is_empty() || (nuf.len() == 1 && nuf[0].type_name == "Void") {
                    (Vec::new(), Vec::new())
                } else {
                    let data_name = format_ident!("{}", struct_name);
                    (vec![quote! { request: &#data_name }], vec![quote! { request }])
                }
            } else {
                return None;
            }
        }
    };

    if is_streaming {
        // Streaming trait impl: generate ephemeral keypair, call raw method, construct StreamHandle
        Some(quote! {
            async fn #method_name(&self #(, #params)*) -> anyhow::Result<hyprstream_rpc::streaming::StreamHandle> {
                let (client_secret, client_pubkey) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
                let client_pubkey_bytes: [u8; 32] = client_pubkey.to_bytes();
                let info = Self::#method_name(self #(, #args)*, client_pubkey_bytes).await?;
                if info.server_pubkey == [0u8; 32] {
                    anyhow::bail!("Server did not provide DH public key for streaming");
                }
                hyprstream_rpc::streaming::StreamHandle::new(
                    &hyprstream_rpc::zmq_context::global_context(),
                    info.stream_id,
                    &info.endpoint,
                    &info.server_pubkey,
                    &client_secret,
                    &client_pubkey_bytes,
                )
            }
        })
    } else {
        Some(quote! {
            async fn #method_name(&self #(, #params)*) -> anyhow::Result<#return_type> {
                Self::#method_name(self #(, #args)*).await
            }
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor Generation (Phase 2b)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate `new` and `with_endpoint` constructors on the client struct.
///
/// Both constructors are `#[cfg(not(target_arch = "wasm32"))]` because `registry`
/// and `zmq_context` are non-wasm only.
pub fn generate_constructors(service_name: &str) -> TokenStream {
    let pascal = to_pascal_case(service_name);
    let client_name = format_ident!("{}Client", pascal);
    let service_name_lit = service_name;

    quote! {
        #[cfg(not(target_arch = "wasm32"))]
        impl #client_name {
            /// The service name used for endpoint resolution.
            pub const SERVICE_NAME: &'static str = #service_name_lit;

            /// Create a new client by looking up the service endpoint from the global registry.
            ///
            /// If the registry is not yet initialized (D9), falls back to the default
            /// inproc endpoint. Use `with_endpoint()` or `from_resolver()` for explicit control.
            pub fn new(
                signing_key: hyprstream_rpc::crypto::SigningKey,
                identity: hyprstream_rpc::envelope::RequestIdentity,
            ) -> Self {
                let endpoint = hyprstream_rpc::registry::try_global()
                    .map(|r| r.endpoint(#service_name_lit, hyprstream_rpc::registry::SocketKind::Rep))
                    .unwrap_or_else(|| hyprstream_rpc::transport::TransportConfig::inproc(
                        concat!("hyprstream/", #service_name_lit)
                    ))
                    .to_zmq_string();
                Self::with_endpoint(&endpoint, signing_key, identity)
            }

            /// Create a new client connected to a specific endpoint.
            pub fn with_endpoint(
                endpoint: &str,
                signing_key: hyprstream_rpc::crypto::SigningKey,
                identity: hyprstream_rpc::envelope::RequestIdentity,
            ) -> Self {
                Self::from_zmq(
                    hyprstream_rpc::zmq_context::create_service_client_base(endpoint, signing_key, identity)
                )
            }

            /// Create a new client by resolving the service endpoint via a `Resolver`.
            pub async fn from_resolver(
                resolver: &dyn hyprstream_rpc::Resolver,
                signing_key: hyprstream_rpc::crypto::SigningKey,
                identity: hyprstream_rpc::envelope::RequestIdentity,
            ) -> anyhow::Result<Self> {
                let endpoint = resolver
                    .resolve(Self::SERVICE_NAME, hyprstream_rpc::registry::SocketKind::Rep)
                    .await?
                    .to_zmq_string();
                Ok(Self::with_endpoint(&endpoint, signing_key, identity))
            }
        }
    }
}

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
        .map(|v| is_rpc_stream_info(&v.type_name, resolved))
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
        t if CapnpType::classify_primitive(t).is_numeric() => {
            let set_method = format_ident!("set_{}", to_snake_case(&variant.name));
            let setter = quote! { req.#set_method(value); };
            let ty = rust_type_tokens(&CapnpType::classify_primitive(t).rust_owned_type());
            quote! {
                #[doc = #doc]
                pub async fn #method_name(&self, value: #ty #extra_param) -> anyhow::Result<#return_type> {
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
                let nuf: Vec<_> = s.non_union_fields().collect();
                let is_void_wrapper = nuf.is_empty()
                    || (nuf.len() == 1 && nuf[0].type_name == "Void");

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
                    let data_name = format_ident!("{}", struct_name);
                    let init_method = format_ident!("init_{}", to_snake_case(&variant.name));

                    quote! {
                        #[doc = #doc]
                        pub async fn #method_name(&self, __request: &#data_name #extra_param) -> anyhow::Result<#return_type> {
                            let __request_id = self.next_id();
                            let payload = hyprstream_rpc::serialize_message(|msg| {
                                let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                                req.set_id(__request_id);
                                #outer_req_setup
                                let mut __inner = req.#init_method();
                                hyprstream_rpc::capnp::ToCapnp::write_to(__request, &mut __inner);
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
