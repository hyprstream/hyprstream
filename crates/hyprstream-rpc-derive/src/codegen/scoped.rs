//! Scoped client generation (e.g., RepositoryClient, ModelSessionClient).

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::codegen::client::{
    generate_parse_match_arm, generate_request_method,
    generate_response_enum_from_variants, ScopedMethodContext,
};
use crate::schema::types::*;
use crate::util::*;

/// Generate all scoped client structs and impls.
pub fn generate_scoped_clients(service_name: &str, schema: &ParsedSchema) -> TokenStream {
    let mut tokens = TokenStream::new();
    let pascal = to_pascal_case(service_name);
    let capnp_mod = format_ident!("{}_capnp", service_name);

    for sc in &schema.scoped_clients {
        tokens.extend(generate_scoped_client(
            &pascal, &capnp_mod, sc, &schema.structs, &schema.enums,
        ));
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
        )
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

