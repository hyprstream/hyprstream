//! Transport-agnostic JSON dispatch generation.
//!
//! Generates a standalone `dispatch()` async function that routes method calls
//! by name, serializing JSON → Cap'n Proto → send → Cap'n Proto → JSON.
//! Works on all targets (native + wasm32) with any send function.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::resolve::ResolvedSchema;
use crate::schema::types::*;
use crate::util::*;

/// Generate a portable dispatch function for a service.
///
/// The generated function has signature:
/// ```ignore
/// pub async fn dispatch<F, Fut>(
///     method: &str,
///     args_json: &str,
///     request_id: u64,
///     send: F,
/// ) -> Result<String, String>
/// where
///     F: FnOnce(Vec<u8>) -> Fut,
///     Fut: std::future::Future<Output = Result<Vec<u8>, String>>,
/// ```
pub fn generate_portable_dispatch(
    service_name: &str,
    resolved: &ResolvedSchema,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let pascal = to_pascal_case(service_name);
    let response_type = format_ident!("{}ResponseVariant", pascal);
    let capnp_mod_ident = format_ident!("{}_capnp", service_name);
    let capnp_mod: TokenStream = match types_crate {
        Some(tc) => quote! { #tc::#capnp_mod_ident },
        None => quote! { crate::#capnp_mod_ident },
    };
    let req_type = format_ident!("{}", to_snake_case(&format!("{pascal}Request")));

    let scoped_names: Vec<&str> = resolved
        .raw
        .scoped_clients
        .iter()
        .map(|sc| sc.factory_name.as_str())
        .collect();

    // Identify which methods are streaming
    let is_streaming = |v: &UnionVariant, resp_variants: &[UnionVariant]| -> bool {
        let result_name = format!("{}Result", v.name);
        resp_variants.iter().any(|r| r.name == result_name && r.type_name == "StreamInfo")
    };

    // Generate match arms for non-streaming, non-scoped methods
    let dispatch_arms: Vec<TokenStream> = resolved
        .raw
        .request_variants
        .iter()
        .filter(|v| !scoped_names.contains(&v.name.as_str()))
        .filter(|v| !v.cli_hidden)
        .filter(|v| !is_streaming(v, &resolved.raw.response_variants))
        .map(|v| generate_dispatch_arm(v, resolved, &capnp_mod, &req_type, &response_type))
        .collect();

    // Generate streaming dispatch arms (top-level)
    let streaming_arms: Vec<TokenStream> = resolved
        .raw
        .request_variants
        .iter()
        .filter(|v| !scoped_names.contains(&v.name.as_str()))
        .filter(|v| !v.cli_hidden)
        .filter(|v| is_streaming(v, &resolved.raw.response_variants))
        .map(|v| generate_streaming_dispatch_arm(v, resolved, &capnp_mod, &req_type, &response_type))
        .collect();

    // Generate scoped dispatch arms (e.g., "ttt.init", "ttt.train")
    // Only generates for direct scoped clients, not nested (3rd level).
    let scoped_dispatch_arms: Vec<TokenStream> = resolved
        .raw
        .scoped_clients
        .iter()
        .filter(|sc| !sc.inner_request_variants.is_empty())
        .flat_map(|sc| {
            let scope_snake = to_snake_case(&sc.factory_name);
            let scope_init = format_ident!("init_{}", to_snake_case(&sc.factory_name));

            // Scope fields: extract from args JSON (before serialize_message closure)
            let scope_field_extracts: Vec<TokenStream> = sc.scope_fields.iter().enumerate().map(|(i, f)| {
                let field_snake = to_snake_case(&f.name);
                let field_camel = &f.name;
                let var_name = format_ident!("__scope_val_{}", i);
                match f.type_name.as_str() {
                    "Text" => quote! {
                        let #var_name = args.get(#field_camel)
                            .or_else(|| args.get(#field_snake))
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| format!("missing scope field '{}' in args", #field_camel))?
                            .to_owned();
                    },
                    _ => quote! {},
                }
            }).collect();
            let scope_field_setters: Vec<TokenStream> = sc.scope_fields.iter().enumerate().map(|(i, f)| {
                let setter = format_ident!("set_{}", to_snake_case(&f.name));
                let var_name = format_ident!("__scope_val_{}", i);
                match f.type_name.as_str() {
                    "Text" => quote! { __scope.#setter(&#var_name); },
                    _ => quote! {},
                }
            }).collect();

            // Inner scoped response type — uses client_name from scoped client detection
            let scoped_response_type = format_ident!("{}ResponseVariant", sc.client_name);

            sc.inner_request_variants.iter()
                .filter(|v| !v.cli_hidden)
                .filter(|v| !is_streaming(v, &sc.inner_response_variants))
                .map(|v| {
                    let method_snake = to_snake_case(&v.name);
                    let full_name = format!("{}.{}", scope_snake, method_snake);
                    let inner_init = format_ident!("init_{}", to_snake_case(&v.name));
                    let inner_set = format_ident!("set_{}", to_snake_case(&v.name));
                    let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();

                    let inner_serialize = match ct {
                        CapnpType::Void => quote! { __scope.#inner_set(()); },
                        CapnpType::Text => quote! {
                            let __val = args.get(#method_snake).or_else(|| args.get("value"))
                                .and_then(|v| v.as_str()).unwrap_or_default();
                            __scope.#inner_set(__val);
                        },
                        _ => {
                            if let Some(sdef) = resolved.find_struct(&v.type_name) {
                                let nuf: Vec<_> = sdef.non_union_fields().collect();
                                let is_void_wrapper = nuf.is_empty()
                                    || (nuf.len() == 1 && nuf[0].type_name == "Void");
                                if is_void_wrapper {
                                    quote! { __scope.#inner_init(); }
                                } else {
                                    // Deserialization happens OUTSIDE the closure
                                    // The closure just uses the pre-deserialized value
                                    quote! {
                                        let mut __inner = __scope.#inner_init();
                                        hyprstream_rpc::capnp::ToCapnp::write_to(&__scoped_req, &mut __inner);
                                    }
                                }
                            } else {
                                quote! { __scope.#inner_init(); }
                            }
                        }
                    };

                    // Pre-deserialize struct args outside the closure
                    let pre_deserialize = match ct.clone() {
                        CapnpType::Struct(_) | CapnpType::Unknown(_) => {
                            if let Some(sdef) = resolved.find_struct(&v.type_name) {
                                let nuf: Vec<_> = sdef.non_union_fields().collect();
                                let is_void_wrapper = nuf.is_empty()
                                    || (nuf.len() == 1 && nuf[0].type_name == "Void");
                                if is_void_wrapper {
                                    quote! {}
                                } else {
                                    let data_name = format_ident!("{}", v.type_name);
                                    quote! {
                                        let __scoped_req: #data_name = serde_json::from_value(args.clone())
                                            .map_err(|e| format!("deserialize {}: {e}", #method_snake))?;
                                    }
                                }
                            } else { quote! {} }
                        }
                        _ => quote! {},
                    };

                    quote! {
                        #full_name => {
                            #(#scope_field_extracts)*
                            #pre_deserialize
                            let payload = hyprstream_rpc::serialize_message(|msg| {
                                let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                                req.set_id(request_id);
                                let mut __scope = req.#scope_init();
                                #(#scope_field_setters)*
                                #inner_serialize
                            }).map_err(|e| format!("serialize: {e}"))?;
                            let response_bytes = send(payload).await?;
                            let variant = #scoped_response_type::parse_scoped_response(&response_bytes)
                                .map_err(|e| format!("parse response: {e}"))?;
                            let json = serde_json::to_string(&variant)
                                .map_err(|e| format!("serialize response: {e}"))?;
                            Ok(DispatchResult::Response(json))
                        }
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Scoped streaming dispatch arms (e.g., "ttt.train_stream", "infer.generate_stream")
    let scoped_streaming_arms: Vec<TokenStream> = resolved
        .raw
        .scoped_clients
        .iter()
        .filter(|sc| !sc.inner_request_variants.is_empty())
        .flat_map(|sc| {
            let scope_snake = to_snake_case(&sc.factory_name);
            let scope_init = format_ident!("init_{}", to_snake_case(&sc.factory_name));

            let scope_field_extracts: Vec<TokenStream> = sc.scope_fields.iter().enumerate().map(|(i, f)| {
                let field_snake = to_snake_case(&f.name);
                let field_camel = &f.name;
                let var_name = format_ident!("__scope_val_{}", i);
                match f.type_name.as_str() {
                    "Text" => quote! {
                        let #var_name = args.get(#field_camel)
                            .or_else(|| args.get(#field_snake))
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| format!("missing scope field '{}' in args", #field_camel))?
                            .to_owned();
                    },
                    _ => quote! {},
                }
            }).collect();
            let scope_field_setters: Vec<TokenStream> = sc.scope_fields.iter().enumerate().map(|(i, f)| {
                let setter = format_ident!("set_{}", to_snake_case(&f.name));
                let var_name = format_ident!("__scope_val_{}", i);
                match f.type_name.as_str() {
                    "Text" => quote! { __scope.#setter(&#var_name); },
                    _ => quote! {},
                }
            }).collect();

            let scoped_response_type = format_ident!("{}ResponseVariant", sc.client_name);

            sc.inner_request_variants.iter()
                .filter(|v| !v.cli_hidden)
                .filter(|v| is_streaming(v, &sc.inner_response_variants))
                .map(|v| {
                    let method_snake = to_snake_case(&v.name);
                    let full_name = format!("{}.{}", scope_snake, method_snake);
                    let inner_init = format_ident!("init_{}", to_snake_case(&v.name));
                    let inner_set = format_ident!("set_{}", to_snake_case(&v.name));
                    let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();

                    let inner_serialize = match ct.clone() {
                        CapnpType::Void => quote! { __scope.#inner_set(()); },
                        _ => {
                            if let Some(sdef) = resolved.find_struct(&v.type_name) {
                                let nuf: Vec<_> = sdef.non_union_fields().collect();
                                let is_void_wrapper = nuf.is_empty()
                                    || (nuf.len() == 1 && nuf[0].type_name == "Void");
                                if is_void_wrapper {
                                    quote! { __scope.#inner_init(); }
                                } else {
                                    quote! {
                                        let mut __inner = __scope.#inner_init();
                                        hyprstream_rpc::capnp::ToCapnp::write_to(&__scoped_req, &mut __inner);
                                    }
                                }
                            } else {
                                quote! { __scope.#inner_init(); }
                            }
                        }
                    };

                    let pre_deserialize = match ct {
                        CapnpType::Struct(_) | CapnpType::Unknown(_) => {
                            if let Some(sdef) = resolved.find_struct(&v.type_name) {
                                let nuf: Vec<_> = sdef.non_union_fields().collect();
                                let is_void_wrapper = nuf.is_empty()
                                    || (nuf.len() == 1 && nuf[0].type_name == "Void");
                                if is_void_wrapper { quote! {} } else {
                                    let data_name = format_ident!("{}", v.type_name);
                                    quote! {
                                        let __scoped_req: #data_name = serde_json::from_value(args.clone())
                                            .map_err(|e| format!("deserialize {}: {e}", #method_snake))?;
                                    }
                                }
                            } else { quote! {} }
                        }
                        _ => quote! {},
                    };

                    quote! {
                        #full_name => {
                            #(#scope_field_extracts)*
                            #pre_deserialize
                            let payload = hyprstream_rpc::serialize_message(|msg| {
                                let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                                req.set_id(request_id);
                                let mut __scope = req.#scope_init();
                                #(#scope_field_setters)*
                                #inner_serialize
                            }).map_err(|e| format!("serialize: {e}"))?;
                            let response_bytes = send(payload).await?;
                            // Parse the scoped streaming response to extract StreamInfo
                            let variant = #scoped_response_type::parse_scoped_response(&response_bytes)
                                .map_err(|e| format!("parse streaming response: {e}"))?;
                            let json = serde_json::to_string(&variant)
                                .map_err(|e| format!("serialize stream info: {e}"))?;
                            Ok(DispatchResult::Stream(json))
                        }
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let method_list: Vec<String> = resolved
        .raw
        .request_variants
        .iter()
        .filter(|v| !scoped_names.contains(&v.name.as_str()) && !v.cli_hidden)
        .map(|v| to_snake_case(&v.name))
        .collect();
    let scoped_method_list: Vec<String> = resolved
        .raw
        .scoped_clients
        .iter()
        .flat_map(|sc| {
            let scope_snake = to_snake_case(&sc.factory_name);
            sc.inner_request_variants.iter()
                .filter(|v| !v.cli_hidden)
                .map(move |v| format!("{}.{}", scope_snake, to_snake_case(&v.name)))
        })
        .collect();
    let mut all_methods = method_list;
    all_methods.extend(scoped_method_list);
    let methods_str = all_methods.join(", ");

    quote! {
        /// Result of a dispatch call — either a JSON response or a streaming setup.
        pub enum DispatchResult {
            /// Normal response — serialized JSON string.
            Response(String),
            /// Streaming response — JSON string of the parsed StreamInfo.
            /// The caller should parse this to get endpoint, serverPubkey, streamId,
            /// then set up the SUB subscription.
            Stream(String),
        }

        /// Transport-agnostic JSON dispatch for this service.
        ///
        /// Routes method calls by name, handling serialization on both sides.
        /// Scoped methods use dot notation: `"ttt.init"`, `"ttt.train"`.
        /// Streaming methods return `DispatchResult::Stream` with raw StreamInfo bytes.
        pub async fn dispatch<F, Fut>(
            method: &str,
            args_json: &str,
            request_id: u64,
            send: F,
        ) -> ::core::result::Result<DispatchResult, String>
        where
            F: FnOnce(Vec<u8>) -> Fut,
            Fut: ::core::future::Future<Output = ::core::result::Result<Vec<u8>, String>>,
        {
            let args: serde_json::Value = serde_json::from_str(args_json)
                .map_err(|e| format!("invalid JSON: {e}"))?;

            match method {
                #(#dispatch_arms)*
                #(#scoped_dispatch_arms)*
                #(#streaming_arms)*
                #(#scoped_streaming_arms)*
                _ => Err(format!("unknown method '{}', available: {}", method, #methods_str)),
            }
        }
    }
}

/// Generate a single dispatch arm for a method.
fn generate_dispatch_arm(
    v: &UnionVariant,
    resolved: &ResolvedSchema,
    capnp_mod: &TokenStream,
    req_type: &proc_macro2::Ident,
    response_type: &proc_macro2::Ident,
) -> TokenStream {
    let method_name_str = to_snake_case(&v.name);
    let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();
    let set_method = format_ident!("set_{}", to_snake_case(&v.name));
    let init_method = format_ident!("init_{}", to_snake_case(&v.name));

    // Build the serialization code based on argument type
    let serialize_body = match ct {
        CapnpType::Void => {
            quote! {
                let payload = hyprstream_rpc::serialize_message(|msg| {
                    let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                    req.set_id(request_id);
                    req.#set_method(());
                }).map_err(|e| format!("serialize: {e}"))?;
            }
        }
        CapnpType::Text => {
            quote! {
                let value = args.as_str()
                    .or_else(|| args[#method_name_str].as_str())
                    .or_else(|| args["value"].as_str())
                    .unwrap_or_default();
                let payload = hyprstream_rpc::serialize_message(|msg| {
                    let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                    req.set_id(request_id);
                    req.#set_method(value);
                }).map_err(|e| format!("serialize: {e}"))?;
            }
        }
        CapnpType::Struct(_) | CapnpType::Unknown(_) => {
            if let Some(sdef) = resolved.find_struct(&v.type_name) {
                let nuf: Vec<_> = sdef.non_union_fields().collect();
                let is_void_wrapper =
                    nuf.is_empty() || (nuf.len() == 1 && nuf[0].type_name == "Void");

                if is_void_wrapper {
                    quote! {
                        let payload = hyprstream_rpc::serialize_message(|msg| {
                            let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                            req.set_id(request_id);
                            req.#init_method();
                        }).map_err(|e| format!("serialize: {e}"))?;
                    }
                } else {
                    let data_name = format_ident!("{}", v.type_name);
                    quote! {
                        let __req: #data_name = serde_json::from_value(args.clone())
                            .map_err(|e| format!("deserialize {}: {e}", #method_name_str))?;
                        let payload = hyprstream_rpc::serialize_message(|msg| {
                            let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                            req.set_id(request_id);
                            let mut __inner = req.#init_method();
                            hyprstream_rpc::capnp::ToCapnp::write_to(&__req, &mut __inner);
                        }).map_err(|e| format!("serialize: {e}"))?;
                    }
                }
            } else {
                let err = format!("struct type not found: {}", v.type_name);
                return quote! { #method_name_str => Err(#err.to_owned()), };
            }
        }
        _ => {
            // Other primitives (Bool, UInt32, etc.)
            quote! {
                let payload = hyprstream_rpc::serialize_message(|msg| {
                    let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                    req.set_id(request_id);
                    req.#set_method(());
                }).map_err(|e| format!("serialize: {e}"))?;
            }
        }
    };

    quote! {
        #method_name_str => {
            #serialize_body
            let response_bytes = send(payload).await?;
            let variant = #response_type::parse_response(&response_bytes)
                .map_err(|e| format!("parse response: {e}"))?;
            let json = serde_json::to_string(&variant)
                .map_err(|e| format!("serialize response: {e}"))?;
            Ok(DispatchResult::Response(json))
        }
    }
}

/// Generate a streaming dispatch arm — returns raw response bytes as DispatchResult::Stream.
fn generate_streaming_dispatch_arm(
    v: &UnionVariant,
    resolved: &ResolvedSchema,
    capnp_mod: &TokenStream,
    req_type: &proc_macro2::Ident,
    response_type: &proc_macro2::Ident,
) -> TokenStream {
    let method_name_str = to_snake_case(&v.name);
    let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();
    let set_method = format_ident!("set_{}", to_snake_case(&v.name));
    let init_method = format_ident!("init_{}", to_snake_case(&v.name));

    // Reuse same serialization patterns as non-streaming
    let serialize_body = match ct {
        CapnpType::Void => quote! {
            let payload = hyprstream_rpc::serialize_message(|msg| {
                let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                req.set_id(request_id);
                req.#set_method(());
            }).map_err(|e| format!("serialize: {e}"))?;
        },
        _ => {
            if let Some(sdef) = resolved.find_struct(&v.type_name) {
                let nuf: Vec<_> = sdef.non_union_fields().collect();
                let is_void_wrapper = nuf.is_empty() || (nuf.len() == 1 && nuf[0].type_name == "Void");
                if is_void_wrapper {
                    quote! {
                        let payload = hyprstream_rpc::serialize_message(|msg| {
                            let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                            req.set_id(request_id);
                            req.#init_method();
                        }).map_err(|e| format!("serialize: {e}"))?;
                    }
                } else {
                    let data_name = format_ident!("{}", v.type_name);
                    quote! {
                        let __req: #data_name = serde_json::from_value(args.clone())
                            .map_err(|e| format!("deserialize {}: {e}", #method_name_str))?;
                        let payload = hyprstream_rpc::serialize_message(|msg| {
                            let mut req = msg.init_root::<#capnp_mod::#req_type::Builder>();
                            req.set_id(request_id);
                            let mut __inner = req.#init_method();
                            hyprstream_rpc::capnp::ToCapnp::write_to(&__req, &mut __inner);
                        }).map_err(|e| format!("serialize: {e}"))?;
                    }
                }
            } else {
                let err = format!("struct type not found: {}", v.type_name);
                return quote! { #method_name_str => Err(#err.to_owned()), };
            }
        }
    };

    quote! {
        #method_name_str => {
            #serialize_body
            let response_bytes = send(payload).await?;
            // Parse the response to extract StreamInfo variant
            let variant = #response_type::parse_response(&response_bytes)
                .map_err(|e| format!("parse streaming response: {e}"))?;
            let json = serde_json::to_string(&variant)
                .map_err(|e| format!("serialize stream info: {e}"))?;
            Ok(DispatchResult::Stream(json))
        }
    }
}
