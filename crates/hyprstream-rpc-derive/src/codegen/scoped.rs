//! Scoped client generation (e.g., RepositoryClient, ModelSessionClient).
//!
//! Uses a recursive approach with `ancestors` to support N-depth nesting.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::codegen::client::{
    generate_parse_match_arm, generate_request_method,
    generate_response_enum_from_variants, ScopedMethodContext,
};
use crate::resolve::ResolvedSchema;
use crate::schema::types::*;
use crate::util::*;

/// Generate all scoped client structs and impls, recursively.
pub fn generate_scoped_clients(service_name: &str, resolved: &ResolvedSchema, types_crate: Option<&syn::Path>) -> TokenStream {
    let mut tokens = TokenStream::new();
    let pascal = to_pascal_case(service_name);
    let capnp_mod_ident = format_ident!("{}_capnp", service_name);
    let capnp_mod: TokenStream = match types_crate {
        Some(tc) => quote! { #tc::#capnp_mod_ident },
        None => quote! { crate::#capnp_mod_ident },
    };

    for sc in &resolved.raw.scoped_clients {
        walk_scoped(
            &mut tokens, &pascal, &capnp_mod, sc, &[],
            resolved, types_crate,
        );
    }

    tokens
}

/// Recursively walk the scoped client tree, generating each client with its ancestors.
fn walk_scoped(
    tokens: &mut TokenStream,
    pascal: &str,
    capnp_mod: &TokenStream,
    sc: &ScopedClient,
    ancestors: &[&ScopedClient],
    resolved: &ResolvedSchema,
    types_crate: Option<&syn::Path>,
) {
    tokens.extend(generate_scoped_client_recursive(
        pascal, capnp_mod, sc, ancestors, resolved, types_crate,
    ));
    let mut next_ancestors: Vec<&ScopedClient> = ancestors.to_vec();
    next_ancestors.push(sc);
    for nested in &sc.nested_clients {
        walk_scoped(tokens, pascal, capnp_mod, nested, &next_ancestors, resolved, types_crate);
    }
}

/// Build a `ScopedMethodContext` chain from ancestors + current scope.
fn build_scope_context(sc: &ScopedClient, ancestors: &[&ScopedClient]) -> ScopedMethodContext {
    let mut parent_ctx: Option<Box<ScopedMethodContext>> = None;
    for ancestor in ancestors {
        parent_ctx = Some(Box::new(ScopedMethodContext {
            factory_name: ancestor.factory_name.clone(),
            scope_fields: ancestor.scope_fields.clone(),
            parent: parent_ctx,
        }));
    }
    ScopedMethodContext {
        factory_name: sc.factory_name.clone(),
        scope_fields: sc.scope_fields.clone(),
        parent: parent_ctx,
    }
}

/// Intermediate representation for response unwrap levels.
struct UnwrapLevel {
    resp_module: syn::Ident,
    variant: syn::Ident,
    which_alias: syn::Ident,
}

/// Generate a single scoped client struct and impl at any nesting depth.
fn generate_scoped_client_recursive(
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
    let doc = format!("Scoped client for {} operations.", sc.factory_name);

    // Response enum
    let response_enum = generate_response_enum_from_variants(
        &format!("{}ResponseVariant", sc.client_name),
        &sc.inner_response_variants,
        resolved,
        types_crate,
    );

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

    // Build parse_scoped_response with iterative unwrap levels
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

    // Innermost match (actual response variants)
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

    // Wrap each level around it (inside-out)
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

    // Request methods
    let scope_ctx = build_scope_context(sc, ancestors);

    let request_methods: Vec<TokenStream> = sc.inner_request_variants.iter().map(|v| {
        generate_request_method(
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

    // Factory methods for nested scoped clients
    let nested_factory_methods: Vec<TokenStream> = sc.nested_clients.iter().map(|nested| {
        generate_nested_factory_method(nested, ancestors, sc)
    }).collect();

    quote! {
        #response_enum

        #[doc = #doc]
        #[derive(Clone)]
        pub struct #client_name {
            client: Arc<ZmqClientBase>,
            #(#all_scope_field_defs,)*
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

/// Generate a factory method on the parent scoped client for creating a nested scoped client.
fn generate_nested_factory_method(
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

    // Clone all ancestor scope fields + parent scope fields
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
                client: Arc::clone(&self.client),
                #(#all_parent_field_inits,)*
                #(#own_field_inits,)*
            }
        }
    }
}
