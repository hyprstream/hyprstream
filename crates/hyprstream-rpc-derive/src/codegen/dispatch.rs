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

    // Generate match arms for non-streaming, non-scoped methods
    let dispatch_arms: Vec<TokenStream> = resolved
        .raw
        .request_variants
        .iter()
        .filter(|v| !scoped_names.contains(&v.name.as_str()))
        .filter(|v| !v.cli_hidden)
        .filter(|v| {
            // Skip streaming methods (they return StreamInfo, not a normal response)
            let result_name = format!("{}Result", v.name);
            !resolved
                .raw
                .response_variants
                .iter()
                .any(|r| r.name == result_name && r.type_name == "StreamInfo")
        })
        .map(|v| generate_dispatch_arm(v, resolved, &capnp_mod, &req_type, &response_type))
        .collect();

    let method_list: Vec<String> = resolved
        .raw
        .request_variants
        .iter()
        .filter(|v| !scoped_names.contains(&v.name.as_str()) && !v.cli_hidden)
        .map(|v| to_snake_case(&v.name))
        .collect();
    let methods_str = method_list.join(", ");

    quote! {
        /// Transport-agnostic JSON dispatch for this service.
        ///
        /// Routes method calls by name, handling serialization on both sides.
        /// Works with any async send function (ZMQ, WebTransport, etc.).
        pub async fn dispatch<F, Fut>(
            method: &str,
            args_json: &str,
            request_id: u64,
            send: F,
        ) -> ::core::result::Result<String, String>
        where
            F: FnOnce(Vec<u8>) -> Fut,
            Fut: ::core::future::Future<Output = ::core::result::Result<Vec<u8>, String>>,
        {
            let args: serde_json::Value = serde_json::from_str(args_json)
                .map_err(|e| format!("invalid JSON: {e}"))?;

            match method {
                #(#dispatch_arms)*
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
            serde_json::to_string(&variant)
                .map_err(|e| format!("serialize response: {e}"))
        }
    }
}
