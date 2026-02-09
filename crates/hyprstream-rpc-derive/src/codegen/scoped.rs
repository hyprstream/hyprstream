//! Scoped client generation (e.g., RepositoryClient, ModelSessionClient).

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::codegen::client::{
    generate_parse_match_arm, generate_request_method,
    generate_response_enum_from_variants, ScopedMethodContext,
};
use crate::schema::types::*;
use crate::util::*;

/// Generate all scoped client structs and impls, including nested scoped clients.
pub fn generate_scoped_clients(service_name: &str, schema: &ParsedSchema) -> TokenStream {
    let mut tokens = TokenStream::new();
    let pascal = to_pascal_case(service_name);
    let capnp_mod = format_ident!("{}_capnp", service_name);

    for sc in &schema.scoped_clients {
        tokens.extend(generate_scoped_client(
            &pascal, &capnp_mod, sc, &schema.structs, &schema.enums,
        ));

        // Generate nested scoped clients (3rd level)
        for nested in &sc.nested_clients {
            tokens.extend(generate_nested_scoped_client(
                &pascal, &capnp_mod, nested, sc, &schema.structs, &schema.enums,
            ));
        }
    }

    tokens
}

fn generate_scoped_client(
    pascal: &str,
    capnp_mod: &syn::Ident,
    sc: &ScopedClient,
    all_structs: &[StructDef],
    enums: &[EnumDef],
) -> TokenStream {
    let client_name = format_ident!("{}", sc.client_name);
    let response_type = format_ident!("{}ResponseVariant", sc.client_name);
    let outer_req_type = format_ident!("{}", to_snake_case(&format!("{pascal}Request")));
    let doc = format!("Scoped client for {} operations.", sc.factory_name);

    // Response enum
    let response_enum = generate_response_enum_from_variants(
        &format!("{}ResponseVariant", sc.client_name),
        &sc.inner_response_variants,
        all_structs,
        enums,
    );

    // Scope fields for the struct
    let scope_field_defs: Vec<TokenStream> = sc.scope_fields.iter().map(|f| {
        let name = format_ident!("{}", to_snake_case(&f.name));
        let ty = rust_type_tokens(&CapnpType::classify_primitive(&f.type_name).rust_owned_type());
        quote! { #name: #ty }
    }).collect();

    // parse_scoped_response
    let outer_resp_type = format_ident!("{}", to_snake_case(&format!("{pascal}Response")));
    let resp_variant_pascal = format_ident!("{}", to_pascal_case(&format!("{}Result", sc.factory_name)));
    let inner_resp_mod = format_ident!("{}", sc.capnp_inner_response);

    let inner_which = format_ident!("InnerWhich");
    let inner_match_arms: Vec<TokenStream> = sc.inner_response_variants.iter().map(|v| {
        generate_parse_match_arm(&response_type, capnp_mod, v, all_structs, enums, &inner_which)
    }).collect();

    // Request methods
    let scope_ctx = ScopedMethodContext {
        factory_name: sc.factory_name.clone(),
        scope_fields: sc.scope_fields.clone(),
        parent: None,
    };

    let request_methods: Vec<TokenStream> = sc.inner_request_variants.iter().map(|v| {
        generate_request_method(
            capnp_mod,
            &outer_req_type,
            &response_type,
            v,
            all_structs,
            enums,
            Some(&scope_ctx),
            Some(&sc.inner_response_variants),
            true,
        )
    }).collect();

    // Factory methods for nested scoped clients
    let nested_factory_methods: Vec<TokenStream> = sc.nested_clients.iter().map(|nested| {
        generate_nested_factory_method(nested, &sc.scope_fields)
    }).collect();

    quote! {
        #response_enum

        #[doc = #doc]
        #[derive(Clone)]
        pub struct #client_name {
            client: Arc<ZmqClientBase>,
            #(#scope_field_defs,)*
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
                let resp = reader.get_root::<crate::#capnp_mod::#outer_resp_type::Reader>()?;
                use crate::#capnp_mod::#outer_resp_type::Which;
                match resp.which()? {
                    Which::#resp_variant_pascal(inner) => {
                        let inner = inner?;
                        use crate::#capnp_mod::#inner_resp_mod::Which as InnerWhich;
                        match inner.which()? {
                            #(#inner_match_arms)*
                            #[allow(unreachable_patterns)]
                            _ => Err(anyhow::anyhow!("unexpected inner response variant")),
                        }
                    }
                    Which::Error(err) => {
                        let err = err?;
                        let msg = err.get_message()?.to_str()?.to_string();
                        Err(anyhow::anyhow!(msg))
                    }
                    _ => Err(anyhow::anyhow!("unexpected outer response variant")),
                }
            }

            #(#request_methods)*

            #(#nested_factory_methods)*
        }
    }
}

/// Generate a factory method on the parent scoped client for creating a nested scoped client.
fn generate_nested_factory_method(
    nested: &ScopedClient,
    parent_scope_fields: &[FieldDef],
) -> TokenStream {
    let method_name = format_ident!("{}", to_snake_case(&nested.factory_name));
    let client_name_ident = format_ident!("{}", nested.client_name);
    let doc = format!("Create a scoped {} client.", nested.factory_name);

    let params: Vec<TokenStream> = nested.scope_fields.iter().map(|f| {
        let name = format_ident!("{}", to_snake_case(&f.name));
        let ty = rust_type_tokens(&CapnpType::classify_primitive(&f.type_name).rust_param_type());
        quote! { #name: #ty }
    }).collect();

    // Clone parent scope fields + own scope fields
    let parent_field_inits: Vec<TokenStream> = parent_scope_fields.iter().map(|f| {
        let name = format_ident!("{}", to_snake_case(&f.name));
        quote! { #name: self.#name.clone() }
    }).collect();

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
                #(#parent_field_inits,)*
                #(#own_field_inits,)*
            }
        }
    }
}

/// Generate a nested scoped client (3rd level, e.g., FsClient within RepositoryClient).
///
/// The nested client carries both parent scope fields (e.g., repo_id) and its own
/// scope fields (e.g., worktree). Request serialization builds a 3-level envelope
/// and response parsing unwraps 3 levels.
fn generate_nested_scoped_client(
    pascal: &str,
    capnp_mod: &syn::Ident,
    nested: &ScopedClient,
    parent: &ScopedClient,
    all_structs: &[StructDef],
    enums: &[EnumDef],
) -> TokenStream {
    let client_name = format_ident!("{}", nested.client_name);
    let response_type = format_ident!("{}ResponseVariant", nested.client_name);
    let outer_req_type = format_ident!("{}", to_snake_case(&format!("{pascal}Request")));
    let doc = format!("Nested scoped client for {} operations.", nested.factory_name);

    // Response enum
    let response_enum = generate_response_enum_from_variants(
        &format!("{}ResponseVariant", nested.client_name),
        &nested.inner_response_variants,
        all_structs,
        enums,
    );

    // Parent scope field defs (e.g., repo_id: String)
    let parent_scope_field_defs: Vec<TokenStream> = parent.scope_fields.iter().map(|f| {
        let name = format_ident!("{}", to_snake_case(&f.name));
        let ty = rust_type_tokens(&CapnpType::classify_primitive(&f.type_name).rust_owned_type());
        quote! { #name: #ty }
    }).collect();

    // Own scope field defs (e.g., worktree: String)
    let own_scope_field_defs: Vec<TokenStream> = nested.scope_fields.iter().map(|f| {
        let name = format_ident!("{}", to_snake_case(&f.name));
        let ty = rust_type_tokens(&CapnpType::classify_primitive(&f.type_name).rust_owned_type());
        quote! { #name: #ty }
    }).collect();

    // 3-level parse_scoped_response:
    // RegistryResponse -> repoResult -> RepositoryResponse -> fsResult -> FsResponse -> variant
    let outer_resp_type = format_ident!("{}", to_snake_case(&format!("{pascal}Response")));
    let parent_resp_variant_pascal = format_ident!("{}", to_pascal_case(&format!("{}Result", parent.factory_name)));
    let parent_inner_resp_mod = format_ident!("{}", parent.capnp_inner_response);
    let nested_resp_variant_pascal = format_ident!("{}", to_pascal_case(&format!("{}Result", nested.factory_name)));
    let nested_inner_resp_mod = format_ident!("{}", nested.capnp_inner_response);

    let inner_which = format_ident!("DeepWhich");
    let inner_match_arms: Vec<TokenStream> = nested.inner_response_variants.iter().map(|v| {
        generate_parse_match_arm(&response_type, capnp_mod, v, all_structs, enums, &inner_which)
    }).collect();

    // Request methods use NestedScopedMethodContext (3-level envelope)
    let scope_ctx = ScopedMethodContext {
        factory_name: nested.factory_name.clone(),
        scope_fields: nested.scope_fields.clone(),
        parent: Some(Box::new(ScopedMethodContext {
            factory_name: parent.factory_name.clone(),
            scope_fields: parent.scope_fields.clone(),
            parent: None,
        })),
    };

    let request_methods: Vec<TokenStream> = nested.inner_request_variants.iter().map(|v| {
        generate_request_method(
            capnp_mod,
            &outer_req_type,
            &response_type,
            v,
            all_structs,
            enums,
            Some(&scope_ctx),
            Some(&nested.inner_response_variants),
            true,
        )
    }).collect();

    quote! {
        #response_enum

        #[doc = #doc]
        #[derive(Clone)]
        pub struct #client_name {
            client: Arc<ZmqClientBase>,
            #(#parent_scope_field_defs,)*
            #(#own_scope_field_defs,)*
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

            /// Parse a nested scoped response from raw bytes (3-level unwrap).
            pub fn parse_scoped_response(bytes: &[u8]) -> anyhow::Result<#response_type> {
                let reader = capnp::serialize::read_message(
                    &mut std::io::Cursor::new(bytes),
                    capnp::message::ReaderOptions::new(),
                )?;
                let resp = reader.get_root::<crate::#capnp_mod::#outer_resp_type::Reader>()?;
                use crate::#capnp_mod::#outer_resp_type::Which;
                match resp.which()? {
                    Which::#parent_resp_variant_pascal(mid) => {
                        let mid = mid?;
                        use crate::#capnp_mod::#parent_inner_resp_mod::Which as MidWhich;
                        match mid.which()? {
                            MidWhich::#nested_resp_variant_pascal(inner) => {
                                let inner = inner?;
                                use crate::#capnp_mod::#nested_inner_resp_mod::Which as DeepWhich;
                                match inner.which()? {
                                    #(#inner_match_arms)*
                                    #[allow(unreachable_patterns)]
                                    _ => Err(anyhow::anyhow!("unexpected deep response variant")),
                                }
                            }
                            MidWhich::Error(err) => {
                                let err = err?;
                                let msg = err.get_message()?.to_str()?.to_string();
                                Err(anyhow::anyhow!(msg))
                            }
                            _ => Err(anyhow::anyhow!("unexpected mid-level response variant")),
                        }
                    }
                    Which::Error(err) => {
                        let err = err?;
                        let msg = err.get_message()?.to_str()?.to_string();
                        Err(anyhow::anyhow!(msg))
                    }
                    _ => Err(anyhow::anyhow!("unexpected outer response variant")),
                }
            }

            #(#request_methods)*
        }
    }
}

