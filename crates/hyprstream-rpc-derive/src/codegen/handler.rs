//! Handler trait, dispatch function, and response serializer generation.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::resolve::ResolvedSchema;
use crate::schema::types::*;
use crate::util::*;

/// Parse "$mcpScope" value into the action name for auth dispatch.
///
/// With the type-safe ScopeAction enum, scope is now just the action name
/// (e.g., "write", "query", "manage"). Returns None if scope is empty.
fn parse_scope_for_auth(scope: &str) -> Option<&str> {
    if scope.is_empty() {
        None
    } else {
        Some(scope)
    }
}

/// Check if a request variant is a streaming redirect (response type == StreamInfo).
///
/// Streaming variants return `(StreamInfo, Continuation)` from handler
/// methods instead of the `ResponseVariant` enum. The dispatch function serializes
/// the StreamInfo into the appropriate response variant and wraps the
/// continuation in `Some()` for post-REP execution.
///
/// For top-level dispatch, response variants are named `{requestName}Result`.
/// For scoped dispatch, response variants match the request variant name directly.
fn is_streaming_variant(req_variant: &UnionVariant, response_variants: &[UnionVariant]) -> bool {
    let response_name = format!("{}Result", req_variant.name);
    response_variants.iter().any(|rv| rv.name == response_name && rv.type_name == "StreamInfo")
}

/// Check if a request variant is streaming within a scoped context.
///
/// In scoped contexts, the response variant name matches the request variant name directly
/// (not `{name}Result`).
fn is_scoped_streaming_variant(req_variant: &UnionVariant, response_variants: &[UnionVariant]) -> bool {
    response_variants.iter().any(|rv| rv.name == req_variant.name && rv.type_name == "StreamInfo")
}

/// Get the response variant name for a streaming request variant.
fn streaming_response_variant_name(req_variant: &UnionVariant) -> String {
    format!("{}Result", req_variant.name)
}

/// Generate handler trait + dispatch function + response serializer.
///
/// When scoped clients are present, also generates per-scope handler traits,
/// dispatch functions, and response serializers. The main handler trait becomes
/// a super-trait requiring all scope handler traits.
pub fn generate_handler(service_name: &str, resolved: &ResolvedSchema, scope_handlers: bool) -> TokenStream {
    let pascal = to_pascal_case(service_name);
    let capnp_mod = format_ident!("{}_capnp", service_name);

    // Only generate scope sub-traits when opt-in via scope_handlers flag.
    // Always pass full scoped_clients for variant recognition (scoped variants
    // get raw-bytes handlers when scope_handlers is false).
    let scope_traits = if scope_handlers {
        generate_scope_handler_traits(
            &resolved.raw.scoped_clients,
            resolved,
        )
    } else {
        TokenStream::new()
    };

    let scope_serializers = if scope_handlers {
        generate_scope_response_serializers(
            &pascal,
            &capnp_mod,
            &resolved.raw.scoped_clients,
            resolved,
        )
    } else {
        TokenStream::new()
    };

    let scope_dispatchers = if scope_handlers {
        generate_scope_dispatch_fns(
            &pascal,
            &capnp_mod,
            &resolved.raw.scoped_clients,
            resolved,
        )
    } else {
        TokenStream::new()
    };

    let handler_trait = generate_handler_trait(
        &pascal,
        &resolved.raw.request_variants,
        &resolved.raw.response_variants,
        &resolved.raw.scoped_clients,
        resolved,
        scope_handlers,
    );

    let dispatch_fn = generate_dispatch_fn(
        &pascal,
        &capnp_mod,
        &resolved.raw.request_variants,
        &resolved.raw.response_variants,
        &resolved.raw.scoped_clients,
        resolved,
        scope_handlers,
    );

    let serializer = generate_response_serializer(
        &pascal,
        &capnp_mod,
        &resolved.raw.response_variants,
        resolved,
    );

    quote! {
        #scope_traits
        #scope_serializers
        #scope_dispatchers
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
    response_variants: &[UnionVariant],
    scoped_clients: &[ScopedClient],
    resolved: &ResolvedSchema,
    scope_handlers: bool,
) -> TokenStream {
    let trait_name = format_ident!("{}Handler", pascal);
    let response_type = format_ident!("{}ResponseVariant", pascal);
    let doc = format!("Generated handler trait for the {pascal} service.");
    let scoped_names: Vec<&str> = scoped_clients
        .iter()
        .map(|sc| sc.factory_name.as_str())
        .collect();

    if scope_handlers {
        // Default authorize() method — services override to enforce policy.
        let authorize_method = quote! {
            /// Authorize an operation on a resource. Override to enforce policy.
            ///
            /// Called by generated dispatch functions before handler methods when
            /// the schema method has a `$mcpScope` annotation.
            /// Default: deny all. Services MUST override to enable access.
            async fn authorize(&self, ctx: &hyprstream_rpc::service::EnvelopeContext, resource: &str, operation: &str) -> anyhow::Result<()> {
                let _ = (ctx, resource, operation);
                anyhow::bail!("Authorization not implemented for this service")
            }

            /// Check if a specific resource+operation is authorized (non-failing).
            ///
            /// Useful for filtering list results per-resource.
            /// Default delegates to `authorize().is_ok()`.
            async fn is_resource_authorized(&self, ctx: &hyprstream_rpc::service::EnvelopeContext, resource: &str, operation: &str) -> bool {
                self.authorize(ctx, resource, operation).await.is_ok()
            }
        };

        // New pattern: scope sub-traits handle inner dispatch.
        // Check if ALL variants are scoped → pure super-trait with no methods.
        let all_scoped = !scoped_clients.is_empty() && request_variants.iter().all(|v| {
            scoped_names.contains(&v.name.as_str()) || is_union_only_struct_variant(v, resolved)
        });

        if all_scoped {
            let scope_bounds: Vec<TokenStream> = scoped_clients.iter().map(|sc| {
                let scope_trait = format_ident!("{}Handler", to_pascal_case(&sc.factory_name));
                quote! { #scope_trait }
            }).collect();

            return quote! {
                #[doc = #doc]
                #[async_trait::async_trait(?Send)]
                pub trait #trait_name: #(#scope_bounds)+* {
                    #authorize_method
                }
            };
        }

        // Mixed: non-scoped methods + scope sub-trait bounds
        let methods: Vec<TokenStream> = request_variants
            .iter()
            .filter(|v| !scoped_names.contains(&v.name.as_str()))
            .map(|v| {
                let method_name = format_ident!("handle_{}", resolved.name(&v.name).snake);

                if is_union_only_struct_variant(v, resolved) {
                    let doc_str = format!("Handle `{}` request (raw bytes for inner dispatch).", v.name);
                    return quote! {
                        #[doc = #doc_str]
                        async fn #method_name(&self, ctx: &hyprstream_rpc::service::EnvelopeContext, request_id: u64, payload: &[u8]) -> anyhow::Result<Vec<u8>>;
                    };
                }

                let params = handler_method_params(v, resolved);

                // Streaming variants (response == StreamInfo) return a tuple with mandatory continuation.
                // The continuation is spawned by RequestLoop AFTER the REP is sent.
                if is_streaming_variant(v, response_variants) {
                    let doc_str = format!("Handle `{}` streaming request. Returns (StreamInfo, mandatory continuation).", v.name);
                    return quote! {
                        #[doc = #doc_str]
                        async fn #method_name(&self, ctx: &hyprstream_rpc::service::EnvelopeContext, request_id: u64 #(, #params)*) -> anyhow::Result<(StreamInfo, hyprstream_rpc::service::Continuation)>;
                    };
                }

                let doc_str = format!("Handle `{}` request.", v.name);

                quote! {
                    #[doc = #doc_str]
                    async fn #method_name(&self, ctx: &hyprstream_rpc::service::EnvelopeContext, request_id: u64 #(, #params)*) -> anyhow::Result<#response_type>;
                }
            })
            .collect();

        let scope_bounds: Vec<TokenStream> = scoped_clients.iter().map(|sc| {
            let scope_trait = format_ident!("{}Handler", to_pascal_case(&sc.factory_name));
            quote! { #scope_trait }
        }).collect();

        quote! {
            #[doc = #doc]
            #[async_trait::async_trait(?Send)]
            pub trait #trait_name: #(#scope_bounds)+* {
                #authorize_method
                #(#methods)*
            }
        }
    } else {
        // Old pattern: scoped variants get raw-bytes handler methods.
        let methods: Vec<TokenStream> = request_variants
            .iter()
            .map(|v| {
                let method_name = format_ident!("handle_{}", resolved.name(&v.name).snake);

                // Scoped or union-only struct → raw bytes passthrough
                if scoped_names.contains(&v.name.as_str()) || is_union_only_struct_variant(v, resolved) {
                    let doc_str = format!("Handle `{}` request (raw bytes for inner dispatch).", v.name);
                    return quote! {
                        #[doc = #doc_str]
                        async fn #method_name(&self, ctx: &hyprstream_rpc::service::EnvelopeContext, request_id: u64, payload: &[u8]) -> anyhow::Result<Vec<u8>>;
                    };
                }

                let params = handler_method_params(v, resolved);

                // Streaming variants (response == StreamInfo) return a tuple with mandatory continuation.
                if is_streaming_variant(v, response_variants) {
                    let doc_str = format!("Handle `{}` streaming request. Returns (StreamInfo, mandatory continuation).", v.name);
                    return quote! {
                        #[doc = #doc_str]
                        async fn #method_name(&self, ctx: &hyprstream_rpc::service::EnvelopeContext, request_id: u64 #(, #params)*) -> anyhow::Result<(StreamInfo, hyprstream_rpc::service::Continuation)>;
                    };
                }

                let doc_str = format!("Handle `{}` request.", v.name);

                quote! {
                    #[doc = #doc_str]
                    async fn #method_name(&self, ctx: &hyprstream_rpc::service::EnvelopeContext, request_id: u64 #(, #params)*) -> anyhow::Result<#response_type>;
                }
            })
            .collect();

        quote! {
            #[doc = #doc]
            #[async_trait::async_trait(?Send)]
            pub trait #trait_name {
                #(#methods)*
            }
        }
    }
}

fn handler_method_params(
    v: &UnionVariant,
    resolved: &ResolvedSchema,
) -> Vec<TokenStream> {
    let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();
    match ct {
        CapnpType::Void => vec![],
        CapnpType::Text => vec![quote! { value: &str }],
        CapnpType::Data => vec![quote! { data: &[u8] }],
        _ if ct.is_numeric() => {
            let ty = rust_type_tokens(&ct.rust_owned_type());
            vec![quote! { value: #ty }]
        }
        CapnpType::Struct(ref name) => {
            if let Some(sdef) = resolved.find_struct(name) {
                // Union-only struct (e.g., scope structs like RuntimeRequest):
                // handler needs raw payload to dispatch the inner union
                if sdef.has_union && sdef.fields.is_empty() {
                    return vec![quote! { payload: &[u8] }];
                }
                // Option 2: Accept whole struct instead of individual fields
                let struct_name = format_ident!("{}", name);
                vec![quote! { data: &#struct_name }]
            } else {
                vec![quote! { data: &[u8] }]
            }
        }
        _ => vec![quote! { data: &[u8] }],
    }
}

/// Check if a union variant's type is a union-only struct (has a union, no regular fields).
/// These act like scoped clients: raw bytes in, raw bytes out.
fn is_union_only_struct_variant(v: &UnionVariant, resolved: &ResolvedSchema) -> bool {
    if let CapnpType::Struct(ref name) = resolved.resolve_type(&v.type_name).capnp_type {
        if let Some(sdef) = resolved.find_struct(name) {
            return sdef.has_union && sdef.fields.is_empty();
        }
    }
    false
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch Function
// ─────────────────────────────────────────────────────────────────────────────

fn generate_dispatch_fn(
    pascal: &str,
    capnp_mod: &syn::Ident,
    request_variants: &[UnionVariant],
    response_variants: &[UnionVariant],
    scoped_clients: &[ScopedClient],
    resolved: &ResolvedSchema,
    scope_handlers: bool,
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
            let variant_pascal = resolved.name(&v.name).pascal_ident.clone();
            let handler_method = format_ident!("handle_{}", resolved.name(&v.name).snake);

            // Scoped clients handling depends on scope_handlers flag
            if scoped_names.contains(&v.name.as_str()) {
                if scope_handlers {
                    // New pattern: delegate to generated scope dispatch function
                    let scope_dispatch_fn = format_ident!("dispatch_{}", resolved.name(&v.name).snake);
                    return quote! {
                        Which::#variant_pascal(_) => {
                            drop(req);
                            drop(reader);
                            return #scope_dispatch_fn(handler, ctx, request_id, payload).await;
                        }
                    };
                } else {
                    // Old pattern: pass raw bytes to handler method
                    return quote! {
                        Which::#variant_pascal(_) => {
                            return Ok((handler.#handler_method(ctx, request_id, payload).await?, None));
                        }
                    };
                }
            }

            // Union-only struct variants (non-scoped): pass raw bytes, return directly
            if is_union_only_struct_variant(v, resolved) {
                let call = if scope_handlers {
                    quote! { #trait_name::#handler_method(handler, ctx, request_id, payload).await }
                } else {
                    quote! { handler.#handler_method(ctx, request_id, payload).await }
                };
                return quote! {
                    Which::#variant_pascal(_) => {
                        return Ok((#call?, None));
                    }
                };
            }

            let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();

            // Generate auth check if scope_handlers is enabled and variant has a scope annotation.
            // Resource is constructed as "{service}" for Void, "{service}:{value}" for Text.
            let service_name_lower = to_snake_case(pascal);
            let auth_check = if scope_handlers {
                parse_scope_for_auth(&v.scope)
                    .map(|action| (action.to_owned(), service_name_lower.clone()))
            } else {
                None
            };

            // When scope_handlers is enabled, use UFCS to disambiguate non-scoped
            // methods from scope sub-trait methods with the same name (e.g.,
            // ModelHandler::handle_load vs PeftHandler::handle_load).
            macro_rules! call_handler {
                ($($args:tt)*) => {
                    if scope_handlers {
                        quote! { #trait_name::#handler_method(handler, ctx, request_id, $($args)*).await }
                    } else {
                        quote! { handler.#handler_method(ctx, request_id, $($args)*).await }
                    }
                };
                () => {
                    if scope_handlers {
                        quote! { #trait_name::#handler_method(handler, ctx, request_id).await }
                    } else {
                        quote! { handler.#handler_method(ctx, request_id).await }
                    }
                };
            }

            // ── Streaming variants (response == StreamInfo) ──
            // Handler returns (StreamInfo, Continuation).
            // Dispatch wraps StreamInfo into the response variant, serializes,
            // and wraps continuation in Some() for post-REP execution.
            if is_streaming_variant(v, response_variants) {
                let resp_variant_name = streaming_response_variant_name(v);
                let resp_variant_pascal = format_ident!("{}", to_pascal_case(&resp_variant_name));
                let response_type = format_ident!("{}ResponseVariant", pascal);

                // Generate auth error return for streaming variants
                let streaming_auth_stmt = if let Some((ref action, ref service)) = auth_check {
                    let err_response_type = format_ident!("{}ResponseVariant", pascal);
                    quote! {
                        if let Err(e) = #trait_name::authorize(handler, ctx, #service, #action).await {
                            let err_variant = #err_response_type::Error(ErrorInfo {
                                message: e.to_string(),
                                code: "UNAUTHORIZED".to_string(),
                                details: String::new(),
                            });
                            return Ok((serialize_response(request_id, &err_variant)?, None));
                        }
                    }
                } else { TokenStream::new() };

                return match ct {
                    CapnpType::Void => {
                        let call = call_handler!();
                        quote! {
                            Which::#variant_pascal(()) => {
                                #streaming_auth_stmt
                                let (stream_info, continuation) = #call?;
                                let variant = #response_type::#resp_variant_pascal(stream_info);
                                return Ok((serialize_response(request_id, &variant)?, Some(continuation)));
                            }
                        }
                    },
                    CapnpType::Struct(ref name) => {
                        if let Some(sdef) = resolved.find_struct(name) {
                            if sdef.has_union && sdef.fields.is_empty() {
                                let call = call_handler!(payload);
                                quote! {
                                    Which::#variant_pascal(_) => {
                                        return Ok((#call?, None));
                                    }
                                }
                            } else {
                                let call = if scope_handlers {
                                    quote! { #trait_name::#handler_method(handler, ctx, request_id, &data).await }
                                } else {
                                    quote! { handler.#handler_method(ctx, request_id, &data).await }
                                };

                                quote! {
                                    Which::#variant_pascal(v) => {
                                        let v = v?;
                                        let data = hyprstream_rpc::capnp::FromCapnp::read_from(v)?;
                                        #streaming_auth_stmt
                                        let (stream_info, continuation) = #call?;
                                        let variant = #response_type::#resp_variant_pascal(stream_info);
                                        return Ok((serialize_response(request_id, &variant)?, Some(continuation)));
                                    }
                                }
                            }
                        } else {
                            let name_str = v.name.clone();
                            quote! { Which::#variant_pascal(_) => { anyhow::bail!("Unhandled variant: {}", #name_str); } }
                        }
                    },
                    _ => {
                        // Streaming with simple types (Text, numeric) — unlikely but handle gracefully
                        let name_str = v.name.clone();
                        quote! { Which::#variant_pascal(_) => { anyhow::bail!("Unsupported streaming variant type: {}", #name_str); } }
                    }
                };
            }

            match ct {
                CapnpType::Void => {
                    let call = call_handler!();
                    if let Some((ref action, ref service)) = auth_check {
                        // Void: resource = "{service}"
                        quote! {
                            Which::#variant_pascal(()) => {
                                #trait_name::authorize(handler, ctx, #service, #action).await?;
                                #call
                            }
                        }
                    } else {
                        quote! {
                            Which::#variant_pascal(()) => #call,
                        }
                    }
                },
                CapnpType::Text => {
                    if let Some((ref action, ref service)) = auth_check {
                        // Text: resource = "{service}:{value}"
                        let call = if scope_handlers {
                            quote! { #trait_name::#handler_method(handler, ctx, request_id, v).await }
                        } else {
                            quote! { handler.#handler_method(ctx, request_id, v).await }
                        };
                        quote! {
                            Which::#variant_pascal(val) => {
                                let v = val?.to_str()?;
                                #trait_name::authorize(handler, ctx, &format!("{}:{}", #service, v), #action).await?;
                                #call
                            }
                        }
                    } else {
                        let call = if scope_handlers {
                            quote! { #trait_name::#handler_method(handler, ctx, request_id, v?.to_str()?).await }
                        } else {
                            quote! { handler.#handler_method(ctx, request_id, v?.to_str()?).await }
                        };
                        quote! {
                            Which::#variant_pascal(v) => #call,
                        }
                    }
                },
                CapnpType::Data => {
                    let call = if scope_handlers {
                        quote! { #trait_name::#handler_method(handler, ctx, request_id, v?).await }
                    } else {
                        quote! { handler.#handler_method(ctx, request_id, v?).await }
                    };
                    if let Some((ref action, ref service)) = auth_check {
                        quote! {
                            Which::#variant_pascal(v) => {
                                #trait_name::authorize(handler, ctx, #service, #action).await?;
                                #call
                            }
                        }
                    } else {
                        quote! {
                            Which::#variant_pascal(v) => #call,
                        }
                    }
                },
                _ if ct.is_numeric() => {
                    let call = if scope_handlers {
                        quote! { #trait_name::#handler_method(handler, ctx, request_id, v).await }
                    } else {
                        quote! { handler.#handler_method(ctx, request_id, v).await }
                    };
                    if let Some((ref action, ref service)) = auth_check {
                        quote! {
                            Which::#variant_pascal(v) => {
                                #trait_name::authorize(handler, ctx, #service, #action).await?;
                                #call
                            }
                        }
                    } else {
                        quote! {
                            Which::#variant_pascal(v) => #call,
                        }
                    }
                },
                CapnpType::Struct(ref name) => {
                    if let Some(sdef) = resolved.find_struct(name) {
                        // Union-only struct (e.g., scope structs): pass raw payload
                        if sdef.has_union && sdef.fields.is_empty() {
                            let call = if scope_handlers {
                                quote! { #trait_name::#handler_method(handler, ctx, request_id, payload).await }
                            } else {
                                quote! { handler.#handler_method(ctx, request_id, payload).await }
                            };
                            return quote! {
                                Which::#variant_pascal(v) => {
                                    let _v = v?;
                                    #call
                                }
                            };
                        }

                        // Option 2: Extract whole struct and pass by reference
                        let call = if scope_handlers {
                            quote! { #trait_name::#handler_method(handler, ctx, request_id, &data).await }
                        } else {
                            quote! { handler.#handler_method(ctx, request_id, &data).await }
                        };

                        let auth_stmt = if let Some((ref action, ref service)) = auth_check {
                            // Struct: resource = "{service}"
                            quote! { #trait_name::authorize(handler, ctx, #service, #action).await?; }
                        } else {
                            TokenStream::new()
                        };

                        quote! {
                            Which::#variant_pascal(v) => {
                                let v = v?;
                                let data = hyprstream_rpc::capnp::FromCapnp::read_from(v)?;
                                #auth_stmt
                                #call
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

    // Two-phase dispatch for Send safety:
    // When all variants are scoped and delegate, extract variant discriminant first,
    // drop readers, then dispatch on discriminant.
    let all_delegated = scope_handlers && request_variants.iter().all(|v| {
        scoped_names.contains(&v.name.as_str()) || is_union_only_struct_variant(v, resolved)
    });

    let dispatch_body = if all_delegated {
        // All variants delegate to scope dispatch - use two-phase pattern
        // Extract discriminant to avoid holding Which across awaits
        let discriminant_arms: Vec<TokenStream> = request_variants.iter().map(|v| {
            let variant_pascal = resolved.name(&v.name).pascal_ident.clone();
            let disc = syn::Index::from(request_variants.iter().position(|rv| rv.name == v.name).unwrap_or(0));
            quote! {
                Which::#variant_pascal(_) => #disc,
            }
        }).collect();

        let dispatch_arms: Vec<TokenStream> = request_variants.iter().enumerate().map(|(idx, v)| {
            let disc = syn::Index::from(idx);
            if scoped_names.contains(&v.name.as_str()) {
                let scope_dispatch_fn = format_ident!("dispatch_{}", resolved.name(&v.name).snake);
                quote! {
                    #disc => return #scope_dispatch_fn(handler, ctx, request_id, payload).await,
                }
            } else if is_union_only_struct_variant(v, resolved) {
                let handler_method = format_ident!("handle_{}", resolved.name(&v.name).snake);
                quote! {
                    #disc => return Ok((#trait_name::#handler_method(handler, ctx, request_id, payload).await?, None)),
                }
            } else {
                quote! {
                    #disc => return anyhow::bail!("Unexpected variant"),
                }
            }
        }).collect();

        quote! {
            // Phase 1: Extract discriminant (no await, Reader in scope)
            let disc = match req.which()? {
                #(#discriminant_arms)*
                #[allow(unreachable_patterns)]
                _ => anyhow::bail!("Unknown request variant (not in schema)"),
            };
            // Reader dropped here
            drop(req);
            drop(reader);

            // Phase 2: Dispatch on discriminant (safe to await)
            match disc {
                #(#dispatch_arms)*
                _ => unreachable!("Discriminant out of range"),
            }
        }
    } else {
        let response_type = format_ident!("{}ResponseVariant", pascal);
        quote! {
            let result = match req.which()? {
                #(#match_arms)*
                #[allow(unreachable_patterns)]
                _ => anyhow::bail!("Unknown request variant (not in schema)"),
            };
            // Convert errors (auth failures, handler errors) to Error response variant
            let result = match result {
                Ok(v) => v,
                Err(e) => #response_type::Error(ErrorInfo {
                    message: e.to_string(),
                    code: "UNAUTHORIZED".to_string(),
                    details: String::new(),
                }),
            };
            Ok((serialize_response(request_id, &result)?, None))
        }
    };

    quote! {
        #[doc = #doc]
        pub async fn #fn_name<H: #trait_name>(handler: &H, ctx: &hyprstream_rpc::service::EnvelopeContext, payload: &[u8]) -> anyhow::Result<(Vec<u8>, Option<hyprstream_rpc::service::Continuation>)> {
            use crate::#capnp_mod::#req_snake::Which;
            let reader = capnp::serialize::read_message(
                &mut std::io::Cursor::new(payload),
                capnp::message::ReaderOptions::new(),
            )?;
            let req = reader.get_root::<crate::#capnp_mod::#req_snake::Reader>()?;
            let request_id = req.get_id();
            #dispatch_body
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Response Serializer
// ─────────────────────────────────────────────────────────────────────────────

fn generate_response_serializer(
    pascal: &str,
    capnp_mod: &syn::Ident,
    response_variants: &[UnionVariant],
    resolved: &ResolvedSchema,
) -> TokenStream {
    let response_type = format_ident!("{}ResponseVariant", pascal);
    let resp_snake = format_ident!("{}", to_snake_case(&format!("{pascal}Response")));
    let doc = format!("Serialize a {pascal}ResponseVariant to Cap'n Proto bytes.");

    let match_arms: Vec<TokenStream> = response_variants
        .iter()
        .map(|v| generate_serializer_arm(pascal, v, resolved))
        .collect();

    quote! {
        #[doc = #doc]
        pub fn serialize_response(request_id: u64, variant: &#response_type) -> anyhow::Result<Vec<u8>> {
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
    v: &UnionVariant,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let response_type = format_ident!("{}ResponseVariant", pascal);
    let variant_pascal = resolved.name(&v.name).pascal_ident.clone();
    let v_snake = &resolved.name(&v.name).snake;
    let setter_name = format_ident!("set_{}", v_snake);
    let init_name = format_ident!("init_{}", v_snake);

    let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();
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
            if resolved.is_struct(name) {
                quote! {
                    #response_type::#variant_pascal(ref v) => {
                        let mut inner = resp.#init_name();
                        <_ as hyprstream_rpc::ToCapnp>::write_to(v, &mut inner);
                    }
                }
            } else {
                quote! {
                    #response_type::#variant_pascal(_) => {
                        // Unknown struct type
                    }
                }
            }
        }
        CapnpType::ListStruct(ref inner) => {
            if resolved.is_struct(inner) {
                quote! {
                    #response_type::#variant_pascal(ref v) => {
                        let mut list = resp.#init_name(v.len() as u32);
                        for (i, item) in v.iter().enumerate() {
                            let mut builder = list.reborrow().get(i as u32);
                            <_ as hyprstream_rpc::ToCapnp>::write_to(item, &mut builder);
                        }
                    }
                }
            } else {
                quote! {
                    #response_type::#variant_pascal(_) => {
                        // Unknown list struct type
                    }
                }
            }
        }
        CapnpType::ListText | CapnpType::ListData => quote! {
            #response_type::#variant_pascal(ref v) => {
                let mut list = resp.#init_name(v.len() as u32);
                for (i, item) in v.iter().enumerate() {
                    list.set(i as u32, item);
                }
            }
        },
        CapnpType::ListPrimitive(_) => quote! {
            #response_type::#variant_pascal(ref v) => {
                let mut list = resp.#init_name(v.len() as u32);
                for (i, item) in v.iter().enumerate() {
                    list.set(i as u32, *item);
                }
            }
        },
        _ => quote! {
            #response_type::#variant_pascal(..) => {
                // Unhandled type
            }
        },
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scope Handler Traits (per-scope typed dispatch)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate scope handler traits for all scoped clients (recursively).
fn generate_scope_handler_traits(
    scoped_clients: &[ScopedClient],
    resolved: &ResolvedSchema,
) -> TokenStream {
    let mut tokens = TokenStream::new();
    for sc in scoped_clients {
        walk_scope_handler_traits(&mut tokens, sc, &[], resolved);
    }
    tokens
}

/// Recursively walk the scope tree generating handler traits (deepest first).
fn walk_scope_handler_traits(
    tokens: &mut TokenStream,
    sc: &ScopedClient,
    ancestors: &[&ScopedClient],
    resolved: &ResolvedSchema,
) {
    // Generate nested traits FIRST (so super-trait bounds resolve)
    let mut next_ancestors: Vec<&ScopedClient> = ancestors.to_vec();
    next_ancestors.push(sc);
    for nc in &sc.nested_clients {
        walk_scope_handler_traits(tokens, nc, &next_ancestors, resolved);
    }

    // Then generate this level's trait
    if ancestors.is_empty() {
        tokens.extend(generate_scope_handler_trait(sc, resolved));
    } else {
        tokens.extend(generate_nested_scope_handler_trait(sc, ancestors, resolved));
    }
}

/// Generate a single scope handler trait (e.g., `RepoHandler`).
///
/// If the scope has nested clients, the trait requires those nested handler traits
/// as super-traits instead of raw-bytes passthrough methods.
fn generate_scope_handler_trait(
    sc: &ScopedClient,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let trait_name = format_ident!("{}Handler", to_pascal_case(&sc.factory_name));
    let doc = format!("Generated handler trait for the {} scope.", sc.factory_name);

    // Build scope field params (e.g., `model_ref: &str`, `repo_id: &str`)
    let scope_field_params: Vec<TokenStream> = sc.scope_fields.iter().map(|f| {
        let name = resolved.name(&f.name).snake_ident.clone();
        let ty = rust_type_tokens(&resolved.resolve_type(&f.type_name).rust_param);
        quote! { #name: #ty }
    }).collect();

    let methods: Vec<TokenStream> = sc.inner_request_variants.iter()
        .filter(|v| v.name != "error")
        .map(|v| {
            let method_name = format_ident!("handle_{}", resolved.name(&v.name).snake);
            let params = scope_handler_method_params(v, resolved);
            let return_type = scope_handler_return_type(v, &sc.inner_response_variants, resolved);
            let doc_str = if v.description.is_empty() {
                format!("Handle `{}` request.", v.name)
            } else {
                v.description.clone()
            };
            quote! {
                #[doc = #doc_str]
                async fn #method_name(&self, ctx: &hyprstream_rpc::service::EnvelopeContext, request_id: u64 #(, #scope_field_params)* #(, #params)*) -> anyhow::Result<#return_type>;
            }
        })
        .collect();

    if sc.nested_clients.is_empty() {
        // No nested clients — simple trait
        quote! {
            #[doc = #doc]
            #[async_trait::async_trait(?Send)]
            pub trait #trait_name {
                #(#methods)*
            }
        }
    } else {
        // Has nested clients — add super-trait bounds
        let nested_bounds: Vec<TokenStream> = sc.nested_clients.iter().map(|nc| {
            let nested_trait = format_ident!("{}Handler", to_pascal_case(&nc.factory_name));
            quote! { #nested_trait }
        }).collect();

        quote! {
            #[doc = #doc]
            #[async_trait::async_trait(?Send)]
            pub trait #trait_name: #(#nested_bounds)+* {
                #(#methods)*
            }
        }
    }
}

/// Generate a nested scope handler trait (e.g., `WorktreeHandler`, `CtlHandler`).
///
/// Nested scope handler methods receive ALL ancestor scope fields AND own scope fields.
/// `ancestors` contains all scopes above this one (e.g., for CtlHandler: [RepoScope, WorktreeScope]).
fn generate_nested_scope_handler_trait(
    nc: &ScopedClient,
    ancestors: &[&ScopedClient],
    resolved: &ResolvedSchema,
) -> TokenStream {
    let trait_name = format_ident!("{}Handler", to_pascal_case(&nc.factory_name));
    let doc = format!("Generated handler trait for the nested {} scope.", nc.factory_name);

    // All ancestor scope field params (e.g., `repo_id: &str, name: &str`)
    let ancestor_scope_field_params: Vec<TokenStream> = ancestors.iter()
        .flat_map(|a| &a.scope_fields)
        .map(|f| {
            let name = resolved.name(&f.name).snake_ident.clone();
            let ty = rust_type_tokens(&resolved.resolve_type(&f.type_name).rust_param);
            quote! { #name: #ty }
        }).collect();

    // Own scope field params (e.g., `fid: u32`)
    let own_scope_field_params: Vec<TokenStream> = nc.scope_fields.iter().map(|f| {
        let name = resolved.name(&f.name).snake_ident.clone();
        let ty = rust_type_tokens(&resolved.resolve_type(&f.type_name).rust_param);
        quote! { #name: #ty }
    }).collect();

    let methods: Vec<TokenStream> = nc.inner_request_variants.iter()
        .filter(|v| v.name != "error")
        .map(|v| {
            let method_name = format_ident!("handle_{}", resolved.name(&v.name).snake);
            let params = scope_handler_method_params(v, resolved);
            let return_type = scope_handler_return_type(v, &nc.inner_response_variants, resolved);
            let doc_str = if v.description.is_empty() {
                format!("Handle `{}` request.", v.name)
            } else {
                v.description.clone()
            };
            quote! {
                #[doc = #doc_str]
                async fn #method_name(&self, ctx: &hyprstream_rpc::service::EnvelopeContext, request_id: u64 #(, #ancestor_scope_field_params)* #(, #own_scope_field_params)* #(, #params)*) -> anyhow::Result<#return_type>;
            }
        })
        .collect();

    // If this nested scope itself has nested clients, add super-trait bounds
    if nc.nested_clients.is_empty() {
        quote! {
            #[doc = #doc]
            #[async_trait::async_trait(?Send)]
            pub trait #trait_name {
                #(#methods)*
            }
        }
    } else {
        let nested_bounds: Vec<TokenStream> = nc.nested_clients.iter().map(|inner_nc| {
            let nested_trait = format_ident!("{}Handler", to_pascal_case(&inner_nc.factory_name));
            quote! { #nested_trait }
        }).collect();

        quote! {
            #[doc = #doc]
            #[async_trait::async_trait(?Send)]
            pub trait #trait_name: #(#nested_bounds)+* {
                #(#methods)*
            }
        }
    }
}

/// Generate method params for scope handler traits.
///
/// Unlike `handler_method_params` which expands struct fields into individual params,
/// this passes complex structs as a single `&StructNameData` parameter for ergonomics.
/// Simple types (Text, Void, Bool, numerics) and small request structs (≤2 fields)
/// still expand into individual params.
fn scope_handler_method_params(
    v: &UnionVariant,
    resolved: &ResolvedSchema,
) -> Vec<TokenStream> {
    let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();
    match ct {
        CapnpType::Void => vec![],
        CapnpType::Text => vec![quote! { value: &str }],
        CapnpType::Data => vec![quote! { data: &[u8] }],
        _ if ct.is_numeric() => {
            let ty = rust_type_tokens(&ct.rust_owned_type());
            vec![quote! { value: #ty }]
        }
        CapnpType::Struct(ref name) => {
            if let Some(sdef) = resolved.find_struct(name) {
                if sdef.has_union && sdef.fields.is_empty() {
                    return vec![quote! { payload: &[u8] }];
                }
                // Option 2: Accept whole struct instead of individual fields
                // (Consistent with handler_method_params for maintainability)
                let struct_name = format_ident!("{}", name);
                vec![quote! { data: &#struct_name }]
            } else {
                vec![quote! { data: &[u8] }]
            }
        }
        _ => vec![quote! { data: &[u8] }],
    }
}

/// Determine the Rust return type for a scope handler method.
///
/// Matches request variant name → response variant name (same name, scoped convention)
/// and converts the response variant's type to a Rust owned type.
///
/// Streaming variants (response == StreamInfo) return `(StreamInfo, Continuation)`
/// — the continuation is mandatory, not optional.
fn scope_handler_return_type(
    request_variant: &UnionVariant,
    response_variants: &[UnionVariant],
    resolved: &ResolvedSchema,
) -> TokenStream {
    if let Some(resp_v) = response_variants.iter().find(|v| v.name == request_variant.name) {
        if resp_v.type_name == "StreamInfo" {
            // Streaming: return tuple with mandatory continuation
            return quote! { (StreamInfo, hyprstream_rpc::service::Continuation) };
        }
        let rt = resolved.resolve_type(&resp_v.type_name);
        rust_type_tokens(&rt.rust_owned)
    } else {
        quote! { () }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scope Dispatch Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Generate scope dispatch functions for all scoped clients (recursively).
fn generate_scope_dispatch_fns(
    pascal: &str,
    capnp_mod: &syn::Ident,
    scoped_clients: &[ScopedClient],
    resolved: &ResolvedSchema,
) -> TokenStream {
    let mut tokens = TokenStream::new();
    for sc in scoped_clients {
        walk_scope_dispatch_fns(&mut tokens, pascal, capnp_mod, sc, &[], resolved);
    }
    tokens
}

/// Recursively walk the scope tree generating dispatch functions (deepest first).
fn walk_scope_dispatch_fns(
    tokens: &mut TokenStream,
    pascal: &str,
    capnp_mod: &syn::Ident,
    sc: &ScopedClient,
    ancestors: &[&ScopedClient],
    resolved: &ResolvedSchema,
) {
    // Generate nested dispatch functions FIRST
    let mut next_ancestors: Vec<&ScopedClient> = ancestors.to_vec();
    next_ancestors.push(sc);
    for nc in &sc.nested_clients {
        walk_scope_dispatch_fns(tokens, pascal, capnp_mod, nc, &next_ancestors, resolved);
    }

    // Then generate this level's dispatch function
    if ancestors.is_empty() {
        tokens.extend(generate_scope_dispatch_fn(pascal, capnp_mod, sc, resolved));
    } else {
        tokens.extend(generate_nested_scope_dispatch_fn(pascal, capnp_mod, ancestors, sc, resolved));
    }
}

/// Generate a single scope dispatch function (e.g., `dispatch_runtime()`).
///
/// The trait bound uses the **parent** handler trait (e.g., `RegistryHandler`)
/// rather than the scope sub-trait (e.g., `RepoHandler`). This allows the
/// dispatch function to call `MainHandler::authorize()` for auth enforcement
/// while still accessing scope handler methods via the super-trait bound.
fn generate_scope_dispatch_fn(
    pascal: &str,
    capnp_mod: &syn::Ident,
    sc: &ScopedClient,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let fn_name = format_ident!("dispatch_{}", to_snake_case(&sc.factory_name));
    // Use the parent (main) handler trait as the bound so we can call authorize()
    let parent_trait = format_ident!("{}Handler", pascal);
    let serializer_fn = format_ident!("serialize_{}_response", to_snake_case(&sc.factory_name));
    let doc = format!(
        "Dispatch a {} scope request: extract params, call async handler (two-phase for Send safety).",
        sc.factory_name
    );

    // Build scope field idents
    let scope_field_idents: Vec<syn::Ident> = sc.scope_fields.iter().map(|f| {
        resolved.name(&f.name).snake_ident.clone()
    }).collect();

    let scope_trait = format_ident!("{}Handler", to_pascal_case(&sc.factory_name));
    let service_name_lower = to_snake_case(pascal);

    // Generate helper types and phases
    let variant_tag_enum = generate_scope_variant_tag_enum(sc, resolved);
    let params_enum = generate_scope_params_enum(sc, resolved);
    let extraction_phase = generate_scope_extraction_phase(pascal, capnp_mod, sc, resolved);
    let dispatch_phase = generate_scope_dispatch_phase(
        sc,
        resolved,
        &scope_field_idents,
        &parent_trait,
        &scope_trait,
        &service_name_lower,
        &serializer_fn,
    );

    quote! {
        #[doc = #doc]
        pub async fn #fn_name<H: #parent_trait>(
            handler: &H,
            ctx: &hyprstream_rpc::service::EnvelopeContext,
            request_id: u64,
            payload: &[u8],
        ) -> anyhow::Result<(Vec<u8>, Option<hyprstream_rpc::service::Continuation>)> {
            // Phase 1: Extract all Cap'n Proto data to owned types
            #variant_tag_enum
            #params_enum

            #extraction_phase

            // Destructure scope fields tuple
            let (#(#scope_field_idents,)*) = scope_fields;

            // Phase 2: Dispatch to async handlers (no Readers held)
            #dispatch_phase
        }
    }
}

/// Generate a nested scope dispatch function (e.g., `dispatch_worktree()`, `dispatch_ctl()`).
///
/// This handles N-level dispatch: receives raw bytes from the parent scope dispatch,
/// re-reads the outer request, navigates outer → ancestor[0] → ... → nested scope,
/// extracts all ancestor + own scope fields, then dispatches the nested inner union.
///
/// `ancestors` contains all scopes above this one (e.g., for ctl: [repo, worktree]).
fn generate_nested_scope_dispatch_fn(
    pascal: &str,
    capnp_mod: &syn::Ident,
    ancestors: &[&ScopedClient],
    nc: &ScopedClient,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let fn_name = format_ident!("dispatch_{}", to_snake_case(&nc.factory_name));
    let parent_trait = format_ident!("{}Handler", pascal);
    let nested_trait = format_ident!("{}Handler", to_pascal_case(&nc.factory_name));
    let serializer_fn = format_ident!("serialize_{}_response", to_snake_case(&nc.factory_name));
    let depth = ancestors.len() + 1; // +1 for the nested scope itself
    let doc = format!(
        "Dispatch a nested {} scope request: {}-level extraction, async handler (two-phase for Send safety).",
        nc.factory_name, depth + 1 // +1 for outer request
    );

    // All ancestor scope field idents (for dispatch phase)
    let all_ancestor_scope_idents: Vec<syn::Ident> = ancestors.iter()
        .flat_map(|a| &a.scope_fields)
        .map(|f| resolved.name(&f.name).snake_ident.clone())
        .collect();

    let nested_scope_idents: Vec<syn::Ident> = nc.scope_fields.iter().map(|f| {
        resolved.name(&f.name).snake_ident.clone()
    }).collect();

    let service_name_lower = to_snake_case(pascal);

    // The dispatch function receives all ancestor scope fields from its parent dispatch function
    let all_ancestor_scope_params: Vec<TokenStream> = ancestors.iter()
        .flat_map(|a| &a.scope_fields)
        .map(|f| {
            let name = resolved.name(&f.name).snake_ident.clone();
            let ty = rust_type_tokens(&resolved.resolve_type(&f.type_name).rust_param);
            quote! { #name: #ty }
        }).collect();

    // Generate helper types and phases
    let variant_tag_enum = generate_scope_variant_tag_enum(nc, resolved);
    let params_enum = generate_scope_params_enum(nc, resolved);
    let extraction_phase = generate_nested_scope_extraction_phase(pascal, capnp_mod, ancestors, nc, resolved);
    let dispatch_phase = generate_nested_scope_dispatch_phase(
        nc,
        resolved,
        &all_ancestor_scope_idents,
        &nested_scope_idents,
        &parent_trait,
        &nested_trait,
        &service_name_lower,
        &serializer_fn,
    );

    quote! {
        #[doc = #doc]
        pub async fn #fn_name<H: #parent_trait>(
            handler: &H,
            ctx: &hyprstream_rpc::service::EnvelopeContext,
            request_id: u64,
            #(#all_ancestor_scope_params,)*
            payload: &[u8],
        ) -> anyhow::Result<(Vec<u8>, Option<hyprstream_rpc::service::Continuation>)> {
            // Phase 1: Extract all Cap'n Proto data to owned types
            #variant_tag_enum
            #params_enum

            #extraction_phase

            // Destructure nested scope fields tuple
            let (#(#nested_scope_idents,)*) = nested_scope_fields;

            // Phase 2: Dispatch to async handlers (no Readers held)
            #dispatch_phase
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scope Response Serializers
// ─────────────────────────────────────────────────────────────────────────────

/// Generate scope response serializers for all scoped clients (recursively).
fn generate_scope_response_serializers(
    pascal: &str,
    capnp_mod: &syn::Ident,
    scoped_clients: &[ScopedClient],
    resolved: &ResolvedSchema,
) -> TokenStream {
    let mut tokens = TokenStream::new();
    for sc in scoped_clients {
        walk_scope_response_serializers(&mut tokens, pascal, capnp_mod, sc, &[], resolved);
    }
    tokens
}

/// Recursively walk the scope tree generating response serializers (deepest first).
fn walk_scope_response_serializers(
    tokens: &mut TokenStream,
    pascal: &str,
    capnp_mod: &syn::Ident,
    sc: &ScopedClient,
    ancestors: &[&ScopedClient],
    resolved: &ResolvedSchema,
) {
    // Generate nested serializers FIRST
    let mut next_ancestors: Vec<&ScopedClient> = ancestors.to_vec();
    next_ancestors.push(sc);
    for nc in &sc.nested_clients {
        walk_scope_response_serializers(tokens, pascal, capnp_mod, nc, &next_ancestors, resolved);
    }

    // Then generate this level's serializer
    if ancestors.is_empty() {
        tokens.extend(generate_scope_response_serializer(pascal, capnp_mod, sc, resolved));
    } else {
        tokens.extend(generate_nested_scope_response_serializer(pascal, capnp_mod, ancestors, sc, resolved));
    }
}

/// Generate a single scope response serializer (e.g., `serialize_runtime_response()`).
fn generate_scope_response_serializer(
    pascal: &str,
    capnp_mod: &syn::Ident,
    sc: &ScopedClient,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let fn_name = format_ident!("serialize_{}_response", to_snake_case(&sc.factory_name));
    let response_type = format_ident!("{}ResponseVariant", sc.client_name);
    let resp_snake = format_ident!("{}", to_snake_case(&format!("{pascal}Response")));
    let scope_result_init = format_ident!("init_{}", to_snake_case(&format!("{}Result", sc.factory_name)));
    let doc = format!("Serialize a {} scope response variant to Cap'n Proto bytes.", sc.factory_name);

    let match_arms: Vec<TokenStream> = sc.inner_response_variants.iter()
        .map(|v| generate_scope_serializer_arm(&response_type, v, resolved))
        .collect();

    quote! {
        #[doc = #doc]
        fn #fn_name(request_id: u64, variant: &#response_type) -> anyhow::Result<Vec<u8>> {
            hyprstream_rpc::serialize_message(|msg| {
                let mut resp = msg.init_root::<crate::#capnp_mod::#resp_snake::Builder>();
                resp.set_request_id(request_id);
                let mut inner = resp.#scope_result_init();
                match variant {
                    #(#match_arms)*
                }
            })
        }
    }
}

/// Generate a single serializer match arm for a scope response variant.
fn generate_scope_serializer_arm(
    response_type: &syn::Ident,
    v: &UnionVariant,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let variant_pascal = resolved.name(&v.name).pascal_ident.clone();
    let v_snake = &resolved.name(&v.name).snake;
    let setter_name = format_ident!("set_{}", v_snake);
    let init_name = format_ident!("init_{}", v_snake);

    let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();
    match ct {
        CapnpType::Void => quote! {
            #response_type::#variant_pascal => {
                inner.#setter_name(());
            }
        },
        CapnpType::Text | CapnpType::Data => quote! {
            #response_type::#variant_pascal(v) => {
                inner.#setter_name(v);
            }
        },
        _ if ct.is_numeric() => quote! {
            #response_type::#variant_pascal(v) => {
                inner.#setter_name(*v);
            }
        },
        CapnpType::Struct(ref name) => {
            if resolved.is_struct(name) {
                quote! {
                    #response_type::#variant_pascal(ref v) => {
                        let mut s = inner.#init_name();
                        <_ as hyprstream_rpc::ToCapnp>::write_to(v, &mut s);
                    }
                }
            } else {
                quote! {
                    #response_type::#variant_pascal(_) => {}
                }
            }
        }
        CapnpType::ListStruct(ref inner_name) => {
            if resolved.is_struct(inner_name) {
                quote! {
                    #response_type::#variant_pascal(ref items) => {
                        let mut list = inner.#init_name(items.len() as u32);
                        for (i, item) in items.iter().enumerate() {
                            <_ as hyprstream_rpc::ToCapnp>::write_to(item, &mut list.reborrow().get(i as u32));
                        }
                    }
                }
            } else {
                quote! {
                    #response_type::#variant_pascal(_) => {}
                }
            }
        }
        CapnpType::ListText | CapnpType::ListData => quote! {
            #response_type::#variant_pascal(ref items) => {
                let mut list = inner.#init_name(items.len() as u32);
                for (i, item) in items.iter().enumerate() {
                    list.set(i as u32, item);
                }
            }
        },
        CapnpType::ListPrimitive(_) => quote! {
            #response_type::#variant_pascal(ref items) => {
                let mut list = inner.#init_name(items.len() as u32);
                for (i, item) in items.iter().enumerate() {
                    list.set(i as u32, *item);
                }
            }
        },
        _ => quote! {
            #response_type::#variant_pascal(..) => {}
        },
    }
}

/// Generate a nested scope response serializer (e.g., `serialize_worktree_response()`,
/// `serialize_ctl_response()`).
///
/// Wraps N+1 levels: outer response → ancestor[0]_result → ... → nc_result → variant.
/// `ancestors` contains all scopes above this one.
fn generate_nested_scope_response_serializer(
    pascal: &str,
    capnp_mod: &syn::Ident,
    ancestors: &[&ScopedClient],
    nc: &ScopedClient,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let fn_name = format_ident!("serialize_{}_response", to_snake_case(&nc.factory_name));
    let response_type = format_ident!("{}ResponseVariant", nc.client_name);
    let resp_snake = format_ident!("{}", to_snake_case(&format!("{pascal}Response")));
    let depth = ancestors.len() + 1;
    let doc = format!(
        "Serialize a nested {} scope response variant to Cap'n Proto bytes ({}-level wrap).",
        nc.factory_name, depth + 1 // +1 for outer response
    );

    let match_arms: Vec<TokenStream> = nc.inner_response_variants.iter()
        .map(|v| generate_scope_serializer_arm(&response_type, v, resolved))
        .collect();

    // Build the chain of init_ calls: resp.init_ancestor[0]_result().init_ancestor[1]_result()...init_nc_result()
    // All ancestors + nc itself form the full chain of scopes to wrap
    let full_chain: Vec<&ScopedClient> = ancestors.iter().copied().chain(std::iter::once(nc)).collect();

    // Generate the init chain as sequential let bindings
    let mut init_stmts = Vec::new();
    for (i, scope) in full_chain.iter().enumerate() {
        let init_fn = format_ident!("init_{}", to_snake_case(&format!("{}Result", scope.factory_name)));
        if i == 0 {
            // First level: init from resp
            init_stmts.push(quote! { let __lvl = resp.#init_fn(); });
        } else if i == full_chain.len() - 1 {
            // Last level: bind to `inner` (mutable)
            init_stmts.push(quote! { let mut inner = __lvl.#init_fn(); });
        } else {
            // Intermediate levels: chain through __lvl
            init_stmts.push(quote! { let __lvl = __lvl.#init_fn(); });
        }
    }

    quote! {
        #[doc = #doc]
        fn #fn_name(request_id: u64, variant: &#response_type) -> anyhow::Result<Vec<u8>> {
            hyprstream_rpc::serialize_message(|msg| {
                let mut resp = msg.init_root::<crate::#capnp_mod::#resp_snake::Builder>();
                resp.set_request_id(request_id);
                #(#init_stmts)*
                match variant {
                    #(#match_arms)*
                }
            })
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Two-Phase Scoped Dispatch Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a variant tag enum for scoped dispatch two-phase pattern.
///
/// This enum contains just the variant identifiers (no data), making it Copy
/// and safe to hold across await points. Used in Phase 2 to match and dispatch.
fn generate_scope_variant_tag_enum(
    sc: &ScopedClient,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let enum_name = format_ident!("{}VariantTag", to_pascal_case(&sc.factory_name));
    let variants: Vec<TokenStream> = sc.inner_request_variants.iter()
        .filter(|v| v.name != "error")
        .map(|v| {
            let variant_pascal = resolved.name(&v.name).pascal_ident.clone();
            quote! { #variant_pascal }
        })
        .collect();

    let nested_variants: Vec<TokenStream> = sc.nested_clients.iter().map(|nc| {
        let variant_pascal = format_ident!("{}", to_pascal_case(&nc.factory_name));
        quote! { #variant_pascal }
    }).collect();

    quote! {
        #[derive(Debug, Copy, Clone)]
        enum #enum_name {
            #(#variants,)*
            #(#nested_variants,)*
        }
    }
}

/// Generate parameter enums for scoped dispatch.
///
/// Creates an enum variant for each request variant, holding the extracted owned data.
fn generate_scope_params_enum(
    sc: &ScopedClient,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let enum_name = format_ident!("{}Params", to_pascal_case(&sc.factory_name));

    let variants: Vec<TokenStream> = sc.inner_request_variants.iter()
        .filter(|v| v.name != "error")
        .map(|v| {
            let variant_pascal = resolved.name(&v.name).pascal_ident.clone();
            let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();

            match ct {
                CapnpType::Void => quote! { #variant_pascal },
                CapnpType::Text => quote! { #variant_pascal(String) },
                CapnpType::Data => quote! { #variant_pascal(Vec<u8>) },
                CapnpType::Bool => quote! { #variant_pascal(bool) },
                _ if ct.is_numeric() => {
                    let rust_type = rust_type_tokens(&ct.rust_param_type());
                    quote! { #variant_pascal(#rust_type) }
                },
                CapnpType::Struct(ref name) => {
                    if let Some(sdef) = resolved.find_struct(name) {
                        if sdef.has_union && sdef.fields.is_empty() {
                            quote! { #variant_pascal }
                        } else {
                            let struct_ident = format_ident!("{}", name);
                            quote! { #variant_pascal(#struct_ident) }
                        }
                    } else {
                        quote! { #variant_pascal }
                    }
                },
                _ => quote! { #variant_pascal },
            }
        })
        .collect();

    let nested_variants: Vec<TokenStream> = sc.nested_clients.iter().map(|nc| {
        let variant_pascal = format_ident!("{}", to_pascal_case(&nc.factory_name));
        quote! { #variant_pascal }
    }).collect();

    quote! {
        enum #enum_name {
            #(#variants,)*
            #(#nested_variants,)*
        }
    }
}

/// Generate the extraction phase for scoped dispatch (Phase 1).
///
/// Returns TokenStream for a block that:
/// 1. Reads Cap'n Proto message
/// 2. Navigates to scope
/// 3. Extracts scope fields
/// 4. Matches on inner variants and extracts all params to owned types
/// 5. Returns (variant_tag, scope_fields, params)
fn generate_scope_extraction_phase(
    pascal: &str,
    capnp_mod: &syn::Ident,
    sc: &ScopedClient,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let outer_req_snake = format_ident!("{}", to_snake_case(&format!("{pascal}Request")));
    let scope_variant_pascal = format_ident!("{}", to_pascal_case(&sc.factory_name));
    let base_name = sc.client_name.trim_end_matches("Client");
    let inner_req_mod = format_ident!("{}_request", to_capnp_module_name(base_name));
    let bail_msg = format!("expected {} scope in request", sc.factory_name);
    let tag_enum = format_ident!("{}VariantTag", to_pascal_case(&sc.factory_name));
    let params_enum = format_ident!("{}Params", to_pascal_case(&sc.factory_name));

    // Scope field idents
    let scope_field_idents: Vec<syn::Ident> = sc.scope_fields.iter().map(|f| {
        resolved.name(&f.name).snake_ident.clone()
    }).collect();

    // Extraction arms for each variant
    let extraction_arms: Vec<TokenStream> = sc.inner_request_variants.iter()
        .filter(|v| v.name != "error")
        .map(|v| {
            let variant_pascal = resolved.name(&v.name).pascal_ident.clone();
            let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();

            match ct {
                CapnpType::Void => {
                    quote! {
                        Which::#variant_pascal(()) => {
                            (#tag_enum::#variant_pascal, #params_enum::#variant_pascal)
                        }
                    }
                },
                CapnpType::Text => {
                    quote! {
                        Which::#variant_pascal(v) => {
                            let v = v?.to_string()?;
                            (#tag_enum::#variant_pascal, #params_enum::#variant_pascal(v))
                        }
                    }
                },
                CapnpType::Data => {
                    quote! {
                        Which::#variant_pascal(v) => {
                            let v = v?.to_vec();
                            (#tag_enum::#variant_pascal, #params_enum::#variant_pascal(v))
                        }
                    }
                },
                _ if ct.is_numeric() => {
                    quote! {
                        Which::#variant_pascal(v) => {
                            (#tag_enum::#variant_pascal, #params_enum::#variant_pascal(v))
                        }
                    }
                },
                CapnpType::Struct(ref name) => {
                    if let Some(sdef) = resolved.find_struct(name) {
                        if sdef.has_union && sdef.fields.is_empty() {
                            quote! {
                                Which::#variant_pascal(_) => {
                                    (#tag_enum::#variant_pascal, #params_enum::#variant_pascal)
                                }
                            }
                        } else {
                            // Extract struct fields to owned types
                            quote! {
                                Which::#variant_pascal(v) => {
                                    let v = v?;
                                    let data = hyprstream_rpc::capnp::FromCapnp::read_from(v)?;
                                    (#tag_enum::#variant_pascal, #params_enum::#variant_pascal(data))
                                }
                            }
                        }
                    } else {
                        quote! {
                            Which::#variant_pascal(_) => {
                                anyhow::bail!("Struct not found in schema");
                            }
                        }
                    }
                },
                _ => {
                    quote! {
                        Which::#variant_pascal(_) => {
                            anyhow::bail!("Unsupported variant type");
                        }
                    }
                }
            }
        })
        .collect();

    // Nested client extraction arms
    let nested_extraction_arms: Vec<TokenStream> = sc.nested_clients.iter().map(|nc| {
        let variant_pascal = format_ident!("{}", to_pascal_case(&nc.factory_name));
        quote! {
            Which::#variant_pascal(_) => {
                (#tag_enum::#variant_pascal, #params_enum::#variant_pascal)
            }
        }
    }).collect();

    // Scope field extractions
    let scope_field_extractions: Vec<TokenStream> = sc.scope_fields.iter().map(|f| {
        let name = resolved.name(&f.name).snake_ident.clone();
        let getter = format_ident!("get_{}", resolved.name(&f.name).snake);
        let ct = resolved.resolve_type(&f.type_name).capnp_type.clone();
        match ct {
            CapnpType::Text => quote! {
                let #name = inner.#getter()?.to_string()?;
            },
            _ if ct.is_numeric() => quote! { let #name = inner.#getter(); },
            _ => quote! {
                let #name = inner.#getter()?.to_string()?;
            },
        }
    }).collect();

    quote! {
        let (variant_tag, scope_fields, params) = {
            let reader = capnp::serialize::read_message(
                &mut std::io::Cursor::new(payload),
                capnp::message::ReaderOptions::new(),
            )?;
            let req = reader.get_root::<crate::#capnp_mod::#outer_req_snake::Reader>()?;

            let inner = match req.which()? {
                crate::#capnp_mod::#outer_req_snake::Which::#scope_variant_pascal(r) => r?,
                _ => anyhow::bail!(#bail_msg),
            };

            #(#scope_field_extractions)*

            use crate::#capnp_mod::#inner_req_mod::Which;
            let (tag, params) = match inner.which()? {
                #(#extraction_arms)*
                #(#nested_extraction_arms)*
                #[allow(unreachable_patterns)]
                _ => anyhow::bail!("Unknown request variant"),
            };

            (tag, (#(#scope_field_idents,)*), params)
        };
    }
}

/// Generate the dispatch phase for scoped dispatch (Phase 2).
///
/// Returns TokenStream for matching on the variant tag and calling async handlers.
/// No Cap'n Proto Readers are held here, only owned types.
fn generate_scope_dispatch_phase(
    sc: &ScopedClient,
    resolved: &ResolvedSchema,
    scope_field_idents: &[syn::Ident],
    parent_trait: &syn::Ident,
    scope_trait: &syn::Ident,
    service_name: &str,
    serializer_fn: &syn::Ident,
) -> TokenStream {
    let tag_enum = format_ident!("{}VariantTag", to_pascal_case(&sc.factory_name));
    let params_enum = format_ident!("{}Params", to_pascal_case(&sc.factory_name));
    let response_type = format_ident!("{}ResponseVariant", sc.client_name);

    // Build scope field arg expressions: pass by-ref for Text/Data (String/Vec<u8>),
    // by-value for numeric types. This matches the handler trait param types.
    let scope_field_args: Vec<TokenStream> = sc.scope_fields.iter().zip(scope_field_idents.iter()).map(|(f, ident)| {
        if resolved.resolve_type(&f.type_name).is_by_ref {
            quote! { &#ident }
        } else {
            quote! { #ident }
        }
    }).collect();

    let dispatch_arms: Vec<TokenStream> = sc.inner_request_variants.iter()
        .filter(|v| v.name != "error")
        .map(|v| {
            let variant_pascal = resolved.name(&v.name).pascal_ident.clone();
            let handler_method = format_ident!("handle_{}", resolved.name(&v.name).snake);
            let resp_variant_pascal = resolved.name(&v.name).pascal_ident.clone();

            // Generate auth check if variant has a scope annotation.
            // Returns Error(ErrorInfo) on auth failure instead of propagating via ?.
            let auth_stmt = if let Some(action) = parse_scope_for_auth(&v.scope) {
                let action_str = action.to_owned();
                let auth_call = if let Some(first_scope_field) = scope_field_idents.first() {
                    let svc = service_name.to_owned();
                    quote! {
                        #parent_trait::authorize(handler, ctx, &format!("{}:{}", #svc, #first_scope_field), #action_str).await
                    }
                } else {
                    let svc = service_name.to_owned();
                    quote! {
                        #parent_trait::authorize(handler, ctx, #svc, #action_str).await
                    }
                };
                quote! {
                    if let Err(e) = #auth_call {
                        let err_variant = #response_type::Error(ErrorInfo {
                            message: e.to_string(),
                            code: "UNAUTHORIZED".to_string(),
                            details: String::new(),
                        });
                        return Ok((#serializer_fn(request_id, &err_variant)?, None));
                    }
                }
            } else {
                TokenStream::new()
            };

            // Check if streaming
            let is_streaming = is_scoped_streaming_variant(v, &sc.inner_response_variants);

            // Check if response is void
            let resp_variant = sc.inner_response_variants.iter().find(|rv| rv.name == v.name);
            let resp_type_is_void = resp_variant.map(|rv| rv.type_name == "Void").unwrap_or(false);

            let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();

            if is_streaming {
                // Streaming variant
                match ct {
                    CapnpType::Void => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                let #params_enum::#variant_pascal = params else { unreachable!() };
                                #auth_stmt
                                let (stream_info, continuation) = #scope_trait::#handler_method(handler, ctx, request_id, #(#scope_field_args,)*).await?;
                                let variant = #response_type::#resp_variant_pascal(stream_info);
                                return Ok((#serializer_fn(request_id, &variant)?, Some(continuation)));
                            }
                        }
                    },
                    _ => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                let #params_enum::#variant_pascal(data) = params else { unreachable!() };
                                #auth_stmt
                                let (stream_info, continuation) = #scope_trait::#handler_method(handler, ctx, request_id, #(#scope_field_args,)* &data).await?;
                                let variant = #response_type::#resp_variant_pascal(stream_info);
                                return Ok((#serializer_fn(request_id, &variant)?, Some(continuation)));
                            }
                        }
                    }
                }
            } else {
                // Non-streaming variant
                let ok_wrap = if resp_type_is_void {
                    quote! { Ok(()) => #response_type::#resp_variant_pascal }
                } else {
                    quote! { Ok(data) => #response_type::#resp_variant_pascal(data) }
                };

                let err_wrap = quote! {
                    Err(e) => #response_type::Error(ErrorInfo {
                        message: e.to_string(),
                        code: "INTERNAL".to_string(),
                        details: String::new(),
                    })
                };

                match ct {
                    CapnpType::Void => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                let #params_enum::#variant_pascal = params else { unreachable!() };
                                #auth_stmt
                                match #scope_trait::#handler_method(handler, ctx, request_id, #(#scope_field_args,)*).await {
                                    #ok_wrap,
                                    #err_wrap,
                                }
                            }
                        }
                    },
                    CapnpType::Text => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                let #params_enum::#variant_pascal(v) = params else { unreachable!() };
                                #auth_stmt
                                match #scope_trait::#handler_method(handler, ctx, request_id, #(#scope_field_args,)* &v).await {
                                    #ok_wrap,
                                    #err_wrap,
                                }
                            }
                        }
                    },
                    CapnpType::Data => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                let #params_enum::#variant_pascal(v) = params else { unreachable!() };
                                #auth_stmt
                                match #scope_trait::#handler_method(handler, ctx, request_id, #(#scope_field_args,)* &v).await {
                                    #ok_wrap,
                                    #err_wrap,
                                }
                            }
                        }
                    },
                    _ if ct.is_numeric() => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                let #params_enum::#variant_pascal(v) = params else { unreachable!() };
                                #auth_stmt
                                match #scope_trait::#handler_method(handler, ctx, request_id, #(#scope_field_args,)* v).await {
                                    #ok_wrap,
                                    #err_wrap,
                                }
                            }
                        }
                    },
                    CapnpType::Struct(_) => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                let #params_enum::#variant_pascal(data) = params else { unreachable!() };
                                #auth_stmt
                                match #scope_trait::#handler_method(handler, ctx, request_id, #(#scope_field_args,)* &data).await {
                                    #ok_wrap,
                                    #err_wrap,
                                }
                            }
                        }
                    },
                    _ => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                anyhow::bail!("Unsupported variant type in dispatch");
                            }
                        }
                    }
                }
            }
        })
        .collect();

    // Nested client dispatch arms
    let nested_dispatch_arms: Vec<TokenStream> = sc.nested_clients.iter().map(|nc| {
        let variant_pascal = format_ident!("{}", to_pascal_case(&nc.factory_name));
        let nested_dispatch_fn = format_ident!("dispatch_{}", to_snake_case(&nc.factory_name));
        quote! {
            #tag_enum::#variant_pascal => {
                return #nested_dispatch_fn(handler, ctx, request_id, #(#scope_field_args,)* payload).await;
            }
        }
    }).collect();

    quote! {
        use #tag_enum::*;
        let result = match variant_tag {
            #(#dispatch_arms)*
            #(#nested_dispatch_arms)*
        };

        Ok((#serializer_fn(request_id, &result)?, None))
    }
}

/// Generate extraction phase for nested scope dispatch (N-level).
///
/// Navigates outer → ancestor[0] → ancestor[1] → ... → nc, extracts nc's scope fields
/// and inner union variant. `ancestors` contains all scopes above nc.
fn generate_nested_scope_extraction_phase(
    pascal: &str,
    capnp_mod: &syn::Ident,
    ancestors: &[&ScopedClient],
    nc: &ScopedClient,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let outer_req_snake = format_ident!("{}", to_snake_case(&format!("{pascal}Request")));

    let tag_enum = format_ident!("{}VariantTag", to_pascal_case(&nc.factory_name));
    let params_enum = format_ident!("{}Params", to_pascal_case(&nc.factory_name));

    // Nested scope field idents
    let nested_scope_idents: Vec<syn::Ident> = nc.scope_fields.iter().map(|f| {
        resolved.name(&f.name).snake_ident.clone()
    }).collect();

    // Build the navigation chain: outer → ancestor[0] → ancestor[1] → ... → nc
    // The full chain of scopes to navigate through (ancestors + nc)
    let full_chain: Vec<&ScopedClient> = ancestors.iter().copied().chain(std::iter::once(nc)).collect();

    // Generate navigation statements to drill down to nc's inner request
    let mut nav_stmts = Vec::new();

    for (i, scope) in full_chain.iter().enumerate() {
        let variant_pascal = format_ident!("{}", to_pascal_case(&scope.factory_name));

        if i == 0 {
            // First level: navigate from outer request
            let bail_msg = format!("expected {} scope in request", scope.factory_name);
            nav_stmts.push(quote! {
                let __nav = match req.which()? {
                    crate::#capnp_mod::#outer_req_snake::Which::#variant_pascal(r) => r?,
                    _ => anyhow::bail!(#bail_msg),
                };
            });
        } else if i < full_chain.len() - 1 {
            // Intermediate levels: navigate through parent's inner union
            let prev = full_chain[i - 1];
            let prev_base = prev.client_name.trim_end_matches("Client");
            let prev_req_mod = format_ident!("{}_request", to_capnp_module_name(prev_base));
            let which_alias = format_ident!("__NavW{}", i);
            let bail_msg = format!("expected {} scope in {} request", scope.factory_name, prev.factory_name);
            nav_stmts.push(quote! {
                use crate::#capnp_mod::#prev_req_mod::Which as #which_alias;
                let __nav = match __nav.which()? {
                    #which_alias::#variant_pascal(r) => r?,
                    _ => anyhow::bail!(#bail_msg),
                };
            });
        } else {
            // Last level (nc): navigate to nc's inner request
            let prev = full_chain[i - 1];
            let prev_base = prev.client_name.trim_end_matches("Client");
            let prev_req_mod = format_ident!("{}_request", to_capnp_module_name(prev_base));
            let which_alias = format_ident!("__NavW{}", i);
            let bail_msg = format!("expected {} scope in {} request", scope.factory_name, prev.factory_name);
            nav_stmts.push(quote! {
                use crate::#capnp_mod::#prev_req_mod::Which as #which_alias;
                let nested_inner = match __nav.which()? {
                    #which_alias::#variant_pascal(r) => r?,
                    _ => anyhow::bail!(#bail_msg),
                };
            });
        }
    }

    // Nested scope field extractions from nested_inner
    let nested_scope_extractions: Vec<TokenStream> = nc.scope_fields.iter().map(|f| {
        let name = resolved.name(&f.name).snake_ident.clone();
        let getter = format_ident!("get_{}", resolved.name(&f.name).snake);
        let ct = resolved.resolve_type(&f.type_name).capnp_type.clone();
        match ct {
            CapnpType::Text => quote! {
                let #name = nested_inner.#getter()?.to_string()?;
            },
            _ if ct.is_numeric() => quote! { let #name = nested_inner.#getter(); },
            _ => quote! {
                let #name = nested_inner.#getter()?.to_string()?;
            },
        }
    }).collect();

    // nc's inner request module for variant extraction
    let nested_base_name = nc.client_name.trim_end_matches("Client");
    let nested_inner_req_mod = format_ident!("{}_request", to_capnp_module_name(nested_base_name));

    // Extraction arms for each nested variant
    let extraction_arms: Vec<TokenStream> = nc.inner_request_variants.iter()
        .filter(|v| v.name != "error")
        .map(|v| {
            let variant_pascal = resolved.name(&v.name).pascal_ident.clone();
            let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();

            match ct {
                CapnpType::Void => {
                    quote! {
                        Which::#variant_pascal(()) => {
                            (#tag_enum::#variant_pascal, #params_enum::#variant_pascal)
                        }
                    }
                },
                CapnpType::Text => {
                    quote! {
                        Which::#variant_pascal(v) => {
                            let v = v?.to_string()?;
                            (#tag_enum::#variant_pascal, #params_enum::#variant_pascal(v))
                        }
                    }
                },
                CapnpType::Data => {
                    quote! {
                        Which::#variant_pascal(v) => {
                            let v = v?.to_vec();
                            (#tag_enum::#variant_pascal, #params_enum::#variant_pascal(v))
                        }
                    }
                },
                _ if ct.is_numeric() => {
                    quote! {
                        Which::#variant_pascal(v) => {
                            (#tag_enum::#variant_pascal, #params_enum::#variant_pascal(v))
                        }
                    }
                },
                CapnpType::Struct(ref name) => {
                    if let Some(sdef) = resolved.find_struct(name) {
                        if sdef.has_union && sdef.fields.is_empty() {
                            quote! {
                                Which::#variant_pascal(_) => {
                                    (#tag_enum::#variant_pascal, #params_enum::#variant_pascal)
                                }
                            }
                        } else {
                            quote! {
                                Which::#variant_pascal(v) => {
                                    let v = v?;
                                    let data = hyprstream_rpc::capnp::FromCapnp::read_from(v)?;
                                    (#tag_enum::#variant_pascal, #params_enum::#variant_pascal(data))
                                }
                            }
                        }
                    } else {
                        quote! {
                            Which::#variant_pascal(_) => {
                                anyhow::bail!("Struct not found in schema");
                            }
                        }
                    }
                },
                _ => {
                    quote! {
                        Which::#variant_pascal(_) => {
                            anyhow::bail!("Unsupported variant type");
                        }
                    }
                }
            }
        })
        .collect();

    // Nested extraction arms for nc's own nested clients (if any)
    let nested_extraction_arms: Vec<TokenStream> = nc.nested_clients.iter().map(|inner_nc| {
        let variant_pascal = format_ident!("{}", to_pascal_case(&inner_nc.factory_name));
        quote! {
            Which::#variant_pascal(_) => {
                (#tag_enum::#variant_pascal, #params_enum::#variant_pascal)
            }
        }
    }).collect();

    quote! {
        let (variant_tag, nested_scope_fields, params) = {
            let reader = capnp::serialize::read_message(
                &mut std::io::Cursor::new(payload),
                capnp::message::ReaderOptions::new(),
            )?;
            let req = reader.get_root::<crate::#capnp_mod::#outer_req_snake::Reader>()?;

            // Navigate through ancestor scopes to reach nc
            #(#nav_stmts)*

            // Extract nested scope fields
            #(#nested_scope_extractions)*

            // Dispatch the nested inner union
            use crate::#capnp_mod::#nested_inner_req_mod::Which;
            let (tag, params) = match nested_inner.which()? {
                #(#extraction_arms)*
                #(#nested_extraction_arms)*
                #[allow(unreachable_patterns)]
                _ => anyhow::bail!("Unknown request variant"),
            };

            (tag, (#(#nested_scope_idents,)*), params)
        };
    }
}

/// Generate dispatch phase for nested scope dispatch.
///
/// `ancestor_scope_idents` contains ALL ancestor scope field idents (flattened).
fn generate_nested_scope_dispatch_phase(
    nc: &ScopedClient,
    resolved: &ResolvedSchema,
    ancestor_scope_idents: &[syn::Ident],
    nested_scope_idents: &[syn::Ident],
    parent_trait: &syn::Ident,
    nested_trait: &syn::Ident,
    service_name: &str,
    serializer_fn: &syn::Ident,
) -> TokenStream {
    let tag_enum = format_ident!("{}VariantTag", to_pascal_case(&nc.factory_name));
    let params_enum = format_ident!("{}Params", to_pascal_case(&nc.factory_name));
    let response_type = format_ident!("{}ResponseVariant", nc.client_name);

    // Ancestor scope fields are already by-ref (function params have &str type),
    // so pass them as-is. Nested scope fields are extracted as owned String,
    // so pass by-ref for Text/Data, by-value for numeric.
    let parent_scope_args: Vec<TokenStream> = ancestor_scope_idents.iter().map(|ident| {
        quote! { #ident }
    }).collect();

    let nested_scope_args: Vec<TokenStream> = nc.scope_fields.iter().zip(nested_scope_idents.iter()).map(|(f, ident)| {
        if resolved.resolve_type(&f.type_name).is_by_ref {
            quote! { &#ident }
        } else {
            quote! { #ident }
        }
    }).collect();

    let dispatch_arms: Vec<TokenStream> = nc.inner_request_variants.iter()
        .filter(|v| v.name != "error")
        .map(|v| {
            let variant_pascal = resolved.name(&v.name).pascal_ident.clone();
            let handler_method = format_ident!("handle_{}", resolved.name(&v.name).snake);
            let resp_variant_pascal = resolved.name(&v.name).pascal_ident.clone();

            // Generate auth check — returns Error(ErrorInfo) on failure instead of propagating via ?
            let auth_stmt = if let Some(action) = parse_scope_for_auth(&v.scope) {
                let action_str = action.to_owned();
                let auth_call = if let Some(first_parent_field) = ancestor_scope_idents.first() {
                    let svc = service_name.to_owned();
                    quote! {
                        #parent_trait::authorize(handler, ctx, &format!("{}:{}", #svc, #first_parent_field), #action_str).await
                    }
                } else {
                    let svc = service_name.to_owned();
                    quote! {
                        #parent_trait::authorize(handler, ctx, #svc, #action_str).await
                    }
                };
                quote! {
                    if let Err(e) = #auth_call {
                        let err_variant = #response_type::Error(ErrorInfo {
                            message: e.to_string(),
                            code: "UNAUTHORIZED".to_string(),
                            details: String::new(),
                        });
                        return Ok((#serializer_fn(request_id, &err_variant)?, None));
                    }
                }
            } else {
                TokenStream::new()
            };

            // Check if streaming
            let is_streaming = is_scoped_streaming_variant(v, &nc.inner_response_variants);

            // Check if response is void
            let resp_variant = nc.inner_response_variants.iter().find(|rv| rv.name == v.name);
            let resp_type_is_void = resp_variant.map(|rv| rv.type_name == "Void").unwrap_or(false);

            let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();

            if is_streaming {
                match ct {
                    CapnpType::Void => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                let #params_enum::#variant_pascal = params else { unreachable!() };
                                #auth_stmt
                                let (stream_info, continuation) = #nested_trait::#handler_method(
                                    handler, ctx, request_id, #(#parent_scope_args,)* #(#nested_scope_args,)*
                                ).await?;
                                let variant = #response_type::#resp_variant_pascal(stream_info);
                                return Ok((#serializer_fn(request_id, &variant)?, Some(continuation)));
                            }
                        }
                    },
                    _ => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                let #params_enum::#variant_pascal(data) = params else { unreachable!() };
                                #auth_stmt
                                let (stream_info, continuation) = #nested_trait::#handler_method(
                                    handler, ctx, request_id, #(#parent_scope_args,)* #(#nested_scope_args,)* &data
                                ).await?;
                                let variant = #response_type::#resp_variant_pascal(stream_info);
                                return Ok((#serializer_fn(request_id, &variant)?, Some(continuation)));
                            }
                        }
                    }
                }
            } else {
                let ok_wrap = if resp_type_is_void {
                    quote! { Ok(()) => #response_type::#resp_variant_pascal }
                } else {
                    quote! { Ok(data) => #response_type::#resp_variant_pascal(data) }
                };

                let err_wrap = quote! {
                    Err(e) => #response_type::Error(ErrorInfo {
                        message: e.to_string(),
                        code: "INTERNAL".to_string(),
                        details: String::new(),
                    })
                };

                match ct {
                    CapnpType::Void => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                let #params_enum::#variant_pascal = params else { unreachable!() };
                                #auth_stmt
                                match #nested_trait::#handler_method(
                                    handler, ctx, request_id, #(#parent_scope_args,)* #(#nested_scope_args,)*
                                ).await {
                                    #ok_wrap,
                                    #err_wrap,
                                }
                            }
                        }
                    },
                    CapnpType::Text => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                let #params_enum::#variant_pascal(v) = params else { unreachable!() };
                                #auth_stmt
                                match #nested_trait::#handler_method(
                                    handler, ctx, request_id, #(#parent_scope_args,)* #(#nested_scope_args,)* &v
                                ).await {
                                    #ok_wrap,
                                    #err_wrap,
                                }
                            }
                        }
                    },
                    CapnpType::Data => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                let #params_enum::#variant_pascal(v) = params else { unreachable!() };
                                #auth_stmt
                                match #nested_trait::#handler_method(
                                    handler, ctx, request_id, #(#parent_scope_args,)* #(#nested_scope_args,)* &v
                                ).await {
                                    #ok_wrap,
                                    #err_wrap,
                                }
                            }
                        }
                    },
                    _ if ct.is_numeric() => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                let #params_enum::#variant_pascal(v) = params else { unreachable!() };
                                #auth_stmt
                                match #nested_trait::#handler_method(
                                    handler, ctx, request_id, #(#parent_scope_args,)* #(#nested_scope_args,)* v
                                ).await {
                                    #ok_wrap,
                                    #err_wrap,
                                }
                            }
                        }
                    },
                    CapnpType::Struct(_) => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                let #params_enum::#variant_pascal(data) = params else { unreachable!() };
                                #auth_stmt
                                match #nested_trait::#handler_method(
                                    handler, ctx, request_id, #(#parent_scope_args,)* #(#nested_scope_args,)* &data
                                ).await {
                                    #ok_wrap,
                                    #err_wrap,
                                }
                            }
                        }
                    },
                    _ => {
                        quote! {
                            #tag_enum::#variant_pascal => {
                                anyhow::bail!("Unsupported variant type in dispatch");
                            }
                        }
                    }
                }
            }
        })
        .collect();

    // Nested client dispatch arms: delegate to nc's own nested dispatch functions
    let nested_dispatch_arms: Vec<TokenStream> = nc.nested_clients.iter().map(|inner_nc| {
        let variant_pascal = format_ident!("{}", to_pascal_case(&inner_nc.factory_name));
        let nested_dispatch_fn = format_ident!("dispatch_{}", to_snake_case(&inner_nc.factory_name));
        // Pass all ancestor scope args + nc's own scope args
        quote! {
            #tag_enum::#variant_pascal => {
                return #nested_dispatch_fn(handler, ctx, request_id, #(#parent_scope_args,)* #(#nested_scope_args,)* payload).await;
            }
        }
    }).collect();

    quote! {
        use #tag_enum::*;
        let result = match variant_tag {
            #(#dispatch_arms)*
            #(#nested_dispatch_arms)*
        };

        Ok((#serializer_fn(request_id, &result)?, None))
    }
}
