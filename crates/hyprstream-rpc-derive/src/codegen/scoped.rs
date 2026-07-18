//! Scoped client generation (e.g., RepositoryClient, ModelSessionClient).
//!
//! Uses a recursive approach with `ancestors` to support N-depth nesting.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::codegen::client::{
    generate_parse_match_arm, generate_portable_request_method,
    generate_response_enum_from_variants, ScopedMethodContext,
};
use crate::resolve::ResolvedSchema;
use crate::schema::types::*;
use crate::util::*;

/// Generate portable scoped client structs using `Arc<dyn RpcClient>`.
/// No ZMQ deps, no jwt_token field — compiles on all targets.
pub fn generate_portable_scoped_clients(service_name: &str, resolved: &ResolvedSchema, types_crate: Option<&syn::Path>) -> TokenStream {
    let mut tokens = TokenStream::new();
    let pascal = to_pascal_case(service_name);
    let rust_service_name = service_name.replace('-', "_");
    let capnp_mod_ident = format_ident!("{}_capnp", to_snake_case(&rust_service_name));
    let capnp_mod: TokenStream = match types_crate {
        Some(tc) => quote! { #tc::#capnp_mod_ident },
        None => quote! { crate::#capnp_mod_ident },
    };

    for sc in &resolved.raw.scoped_clients {
        walk_portable_scoped(
            &mut tokens, service_name, &pascal, &capnp_mod, sc, &[],
            resolved, types_crate,
        );
    }

    tokens
}

/// Generate only the response enums and parse functions for scoped clients.
/// No client structs, no transport deps — compiles on all targets including wasm32.
pub fn generate_scoped_response_types(service_name: &str, resolved: &ResolvedSchema, types_crate: Option<&syn::Path>) -> TokenStream {
    let mut tokens = TokenStream::new();
    let pascal = to_pascal_case(service_name);
    let capnp_mod_ident = format_ident!("{}_capnp", service_name);
    let capnp_mod: TokenStream = match types_crate {
        Some(tc) => quote! { #tc::#capnp_mod_ident },
        None => quote! { crate::#capnp_mod_ident },
    };

    for sc in &resolved.raw.scoped_clients {
        walk_scoped_response_types(&mut tokens, &pascal, &capnp_mod, sc, &[], resolved, types_crate);
    }

    tokens
}

fn walk_scoped_response_types(
    tokens: &mut TokenStream,
    pascal: &str,
    capnp_mod: &TokenStream,
    sc: &ScopedClient,
    ancestors: &[&ScopedClient],
    resolved: &ResolvedSchema,
    types_crate: Option<&syn::Path>,
) {
    tokens.extend(generate_scoped_response_only(pascal, capnp_mod, sc, ancestors, resolved, types_crate));
    let mut next_ancestors: Vec<&ScopedClient> = ancestors.to_vec();
    next_ancestors.push(sc);
    for nested in &sc.nested_clients {
        walk_scoped_response_types(tokens, pascal, capnp_mod, nested, &next_ancestors, resolved, types_crate);
    }
}

/// Generate response enum + parse_scoped_response for one scoped client level.
fn generate_scoped_response_only(
    pascal: &str,
    capnp_mod: &TokenStream,
    sc: &ScopedClient,
    ancestors: &[&ScopedClient],
    resolved: &ResolvedSchema,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let _client_name = format_ident!("{}Client", to_pascal_case(&sc.client_name));
    let response_type = format_ident!("{}ResponseVariant", to_pascal_case(&sc.client_name));

    let response_enum = generate_response_enum_from_variants(
        &response_type.to_string(),
        &sc.inner_response_variants,
        resolved,
        types_crate,
    );

    // Build parse_scoped_response
    let outer_resp_type = format_ident!("{}", to_snake_case(&format!("{pascal}Response")));
    let full_chain: Vec<&ScopedClient> = ancestors.iter().copied().chain(std::iter::once(sc)).collect();

    let mut levels = Vec::new();
    for (i, level_sc) in full_chain.iter().enumerate() {
        let resp_module = if i == 0 {
            format_ident!("{}", to_snake_case(&format!("{pascal}Response")))
        } else {
            format_ident!("{}", full_chain[i - 1].capnp_inner_response)
        };
        let variant = format_ident!("{}", to_pascal_case(&format!("{}Result", level_sc.factory_name)));
        let which_alias = format_ident!("__W{}", i);
        levels.push(UnwrapLevel { resp_module, variant, which_alias });
    }

    let inner_which = format_ident!("__WInner");
    let inner_match_arms: Vec<TokenStream> = sc.inner_response_variants.iter().map(|v| {
        generate_parse_match_arm(&response_type, capnp_mod, v, resolved, &inner_which, types_crate)
    }).collect();

    let inner_resp_mod = format_ident!("{}", sc.capnp_inner_response);
    let mut body = quote! {
        use #capnp_mod::#inner_resp_mod::Which as __WInner;
        match resp.which()? {
            #(#inner_match_arms)*
            #[allow(unreachable_patterns)]
            _ => Err(anyhow::anyhow!("unexpected inner response variant")),
        }
    };

    for level in levels.iter().rev() {
        let which_alias = &level.which_alias;
        let variant = &level.variant;
        let resp_module = &level.resp_module;
        body = quote! {
            use #capnp_mod::#resp_module::Which as #which_alias;
            match resp.which()? {
                #which_alias::#variant(resp) => {
                    let resp = resp?;
                    #body
                }
                #which_alias::Error(err) => {
                    let err = err?;
                    let msg = err.get_message()?.to_str()?.to_string();
                    Err(anyhow::anyhow!(msg))
                }
                _ => Err(anyhow::anyhow!("unexpected response variant")),
            }
        };
    }

    quote! {
        #response_enum

        impl #response_type {
            /// Parse a scoped response from raw bytes.
            pub fn parse_scoped_response(bytes: &[u8]) -> anyhow::Result<#response_type> {
                let reader = capnp::serialize::read_message(
                    &mut std::io::Cursor::new(bytes),
                    capnp::message::ReaderOptions::new(),
                )?;
                let resp = reader.get_root::<#capnp_mod::#outer_resp_type::Reader>()?;
                #body
            }
        }
    }
}

/// Build a `ScopedMethodContext` chain from ancestors + current scope.
fn build_scope_context(
    sc: &ScopedClient,
    ancestors: &[&ScopedClient],
    resolved: &ResolvedSchema,
) -> ScopedMethodContext {
    let root_name = ancestors
        .first()
        .map_or(sc.factory_name.as_str(), |root| root.factory_name.as_str());
    let root_method_discriminator = resolved
        .raw
        .request_struct
        .as_ref()
        .and_then(|request| request.union_fields().find(|field| field.name == root_name))
        .map(|field| field.discriminant_value)
        .or_else(|| {
            resolved
                .raw
                .request_variants
                .iter()
                .position(|variant| variant.name == root_name)
                .and_then(|index| u16::try_from(index).ok())
        })
        .unwrap_or(0);
    let mut parent_ctx: Option<Box<ScopedMethodContext>> = None;
    for ancestor in ancestors {
        parent_ctx = Some(Box::new(ScopedMethodContext {
            factory_name: ancestor.factory_name.clone(),
            scope_fields: ancestor.scope_fields.clone(),
            parent: parent_ctx,
            root_method_discriminator,
        }));
    }
    ScopedMethodContext {
        factory_name: sc.factory_name.clone(),
        scope_fields: sc.scope_fields.clone(),
        parent: parent_ctx,
        root_method_discriminator,
    }
}

/// Intermediate representation for response unwrap levels.
struct UnwrapLevel {
    resp_module: syn::Ident,
    variant: syn::Ident,
    which_alias: syn::Ident,
}

// ============================================================================
// Portable scoped clients (Arc<dyn RpcClient>, no ZMQ deps)
// ============================================================================

fn walk_portable_scoped(
    tokens: &mut TokenStream,
    service_name: &str,
    pascal: &str,
    capnp_mod: &TokenStream,
    sc: &ScopedClient,
    ancestors: &[&ScopedClient],
    resolved: &ResolvedSchema,
    types_crate: Option<&syn::Path>,
) {
    tokens.extend(generate_portable_scoped_client_recursive(
        service_name, pascal, capnp_mod, sc, ancestors, resolved, types_crate,
    ));
    let mut next_ancestors: Vec<&ScopedClient> = ancestors.to_vec();
    next_ancestors.push(sc);
    for nested in &sc.nested_clients {
        walk_portable_scoped(
            tokens,
            service_name,
            pascal,
            capnp_mod,
            nested,
            &next_ancestors,
            resolved,
            types_crate,
        );
    }
}

/// Generate a portable scoped client struct using `Arc<dyn RpcClient>`.
fn generate_portable_scoped_client_recursive(
    service_name: &str,
    pascal: &str,
    capnp_mod: &TokenStream,
    sc: &ScopedClient,
    ancestors: &[&ScopedClient],
    resolved: &ResolvedSchema,
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let client_name = format_ident!("{}", sc.client_name);
    let response_type = format_ident!("{}ResponseVariant", sc.client_name);
    let outer_req_type = format_ident!("{}", to_snake_case(&format!("{pascal}Request")));
    let doc = format!("Scoped client for {} operations (transport-agnostic).", sc.factory_name);

    // Struct fields: all ancestor scope fields + own scope fields
    let all_scope_field_defs: Vec<TokenStream> = ancestors.iter()
        .flat_map(|a| &a.scope_fields)
        .chain(&sc.scope_fields)
        .map(|f| {
            let name = format_ident!("{}", to_snake_case(&f.name));
            let ty = rust_type_tokens(&CapnpType::classify_primitive(&f.type_name).rust_owned_type());
            quote! { #name: #ty }
        })
        .collect();

    // Build parse_scoped_response body (same as native — pure capnp parsing)
    let outer_resp_type = format_ident!("{}", to_snake_case(&format!("{pascal}Response")));
    let full_chain: Vec<&ScopedClient> = ancestors.iter().copied().chain(std::iter::once(sc)).collect();

    let mut levels = Vec::new();
    for (i, level_sc) in full_chain.iter().enumerate() {
        let resp_module = if i == 0 {
            format_ident!("{}", to_snake_case(&format!("{pascal}Response")))
        } else {
            format_ident!("{}", full_chain[i - 1].capnp_inner_response)
        };
        let variant = format_ident!("{}", to_pascal_case(&format!("{}Result", level_sc.factory_name)));
        let which_alias = format_ident!("__W{}", i);
        levels.push(UnwrapLevel { resp_module, variant, which_alias });
    }

    let inner_which = format_ident!("__WInner");
    let inner_match_arms: Vec<TokenStream> = sc.inner_response_variants.iter().map(|v| {
        generate_parse_match_arm(&response_type, capnp_mod, v, resolved, &inner_which, types_crate)
    }).collect();

    let inner_resp_mod = format_ident!("{}", sc.capnp_inner_response);
    let mut body = quote! {
        use #capnp_mod::#inner_resp_mod::Which as __WInner;
        match resp.which()? {
            #(#inner_match_arms)*
            #[allow(unreachable_patterns)]
            _ => Err(anyhow::anyhow!("unexpected inner response variant")),
        }
    };

    for level in levels.iter().rev() {
        let which_alias = &level.which_alias;
        let variant = &level.variant;
        let resp_module = &level.resp_module;
        body = quote! {
            use #capnp_mod::#resp_module::Which as #which_alias;
            match resp.which()? {
                #which_alias::#variant(resp) => {
                    let resp = resp?;
                    #body
                }
                #which_alias::Error(err) => {
                    let err = err?;
                    let msg = err.get_message()?.to_str()?.to_string();
                    Err(anyhow::anyhow!(msg))
                }
                _ => Err(anyhow::anyhow!("unexpected response variant")),
            }
        };
    }

    // Request methods (portable mode)
    let scope_ctx = build_scope_context(sc, ancestors, resolved);
    let request_methods: Vec<TokenStream> = sc.inner_request_variants.iter().map(|v| {
        generate_portable_request_method(
            capnp_mod,
            &outer_req_type,
            &response_type,
            v,
            resolved,
            Some(&scope_ctx),
            Some(&sc.inner_response_variants),
            true,
            types_crate,
        )
    }).collect();

    // Nested factory methods (portable — no jwt_token)
    let nested_factory_methods: Vec<TokenStream> = sc.nested_clients.iter().map(|nested| {
        generate_portable_nested_factory_method(nested, ancestors, sc)
    }).collect();

    quote! {
        #[doc = #doc]
        #[derive(Clone)]
        pub struct #client_name {
            client: std::sync::Arc<dyn hyprstream_rpc::RpcClient>,
            #(#all_scope_field_defs,)*
        }

        impl #client_name {
            /// Get the next request ID.
            pub fn next_id(&self) -> u64 {
                self.client.next_id()
            }

            /// Send a raw request and return the raw response bytes.
            pub async fn call(&self, payload: Vec<u8>) -> anyhow::Result<Vec<u8>> {
                self.client.call_for_service(#service_name, payload).await
            }

            /// Send a generated request with its canonical root method id.
            async fn call_with_method(
                &self,
                method_discriminator: u16,
                payload: Vec<u8>,
            ) -> anyhow::Result<Vec<u8>> {
                self.client
                    .call_for_service_with_method(#service_name, method_discriminator, payload)
                    .await
            }

            /// Send a streaming request with ephemeral DH pubkey.
            pub async fn call_streaming(&self, payload: Vec<u8>, ephemeral_pubkey: [u8; 32]) -> anyhow::Result<Vec<u8>> {
                self.client
                    .call_streaming_for_service(#service_name, payload, ephemeral_pubkey)
                    .await
            }

            /// Send a generated streaming request with its canonical root method id.
            async fn call_streaming_with_method(
                &self,
                method_discriminator: u16,
                payload: Vec<u8>,
                ephemeral_pubkey: [u8; 32],
            ) -> anyhow::Result<Vec<u8>> {
                self.client
                    .call_streaming_for_service_with_method(
                        #service_name,
                        method_discriminator,
                        payload,
                        ephemeral_pubkey,
                    )
                    .await
            }

            /// Parse a scoped response from raw bytes.
            pub fn parse_scoped_response(bytes: &[u8]) -> anyhow::Result<#response_type> {
                let reader = capnp::serialize::read_message(
                    &mut std::io::Cursor::new(bytes),
                    capnp::message::ReaderOptions::new(),
                )?;
                let resp = reader.get_root::<#capnp_mod::#outer_resp_type::Reader>()?;
                #body
            }

            #(#request_methods)*

            #(#nested_factory_methods)*
        }
    }
}

/// Portable nested factory method (no jwt_token).
fn generate_portable_nested_factory_method(
    nested: &ScopedClient,
    ancestors: &[&ScopedClient],
    parent: &ScopedClient,
) -> TokenStream {
    let method_name = format_ident!("{}", to_snake_case(&nested.factory_name));
    let client_name_ident = format_ident!("{}", nested.client_name);
    let doc = format!("Create a scoped {} client.", nested.factory_name);

    let params: Vec<TokenStream> = nested.scope_fields.iter().map(|f| {
        let name = format_ident!("{}", to_snake_case(&f.name));
        let ty = rust_type_tokens(&CapnpType::classify_primitive(&f.type_name).rust_param_type());
        quote! { #name: #ty }
    }).collect();

    let all_parent_field_inits: Vec<TokenStream> = ancestors.iter()
        .flat_map(|a| &a.scope_fields)
        .chain(&parent.scope_fields)
        .map(|f| {
            let name = format_ident!("{}", to_snake_case(&f.name));
            quote! { #name: self.#name.clone() }
        })
        .collect();

    let own_field_inits: Vec<TokenStream> = nested.scope_fields.iter().map(|f| {
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
                client: std::sync::Arc::clone(&self.client),
                #(#all_parent_field_inits,)*
                #(#own_field_inits,)*
            }
        }
    }
}

#[cfg(test)]
mod canonical_service_authority_tests {
    use super::*;

    #[test]
    fn hyphenated_scoped_service_keeps_original_authority() {
        let schema = ParsedSchema {
            request_variants: Vec::new(),
            response_variants: Vec::new(),
            structs: Vec::new(),
            scoped_clients: vec![ScopedClient {
                factory_name: "document".to_owned(),
                client_name: "DocumentClient".to_owned(),
                scope_fields: Vec::new(),
                inner_request_variants: Vec::new(),
                inner_response_variants: Vec::new(),
                capnp_inner_response: "document_response".to_owned(),
                nested_clients: Vec::new(),
            }],
            enums: Vec::new(),
            request_struct: None,
            response_struct: None,
        };
        let resolved = ResolvedSchema::from(&schema);
        let generated = generate_portable_scoped_clients("text-generation", &resolved, None)
            .to_string();

        assert!(generated.contains("text-generation"));
        assert!(!generated.contains("text_generation\""));
        assert!(generated.contains("text_generation_capnp"));
    }
}
