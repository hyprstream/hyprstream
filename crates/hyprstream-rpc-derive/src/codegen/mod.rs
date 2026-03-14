//! Code generation from parsed Cap'n Proto schemas using quote!.

pub mod client;
pub mod data;
pub mod handler;
pub mod metadata;
pub mod scoped;

use crate::resolve::ResolvedSchema;
use crate::schema::types::ParsedSchema;

/// Generate all service code from a parsed schema.
///
/// When `types_crate` is `Some`, generates client-only code (no handler/dispatch)
/// and resolves capnp module paths through the external crate.
pub fn generate_service(service_name: &str, schema: &ParsedSchema, types_crate: Option<&syn::Path>, scope_handlers: bool) -> proc_macro2::TokenStream {
    let resolved = ResolvedSchema::from(schema);
    let is_data_only = resolved.raw.request_variants.is_empty();

    let data_structs = data::generate_data_structs(&resolved, service_name, types_crate);

    if is_data_only {
        // Data-only schema: only emit data structs (no client, handler, metadata)
        return quote::quote! {
            #data_structs
        };
    }

    let metadata_code = metadata::generate_metadata(service_name, &resolved, types_crate);

    let response_enum = client::generate_response_enum(service_name, &resolved, types_crate);
    let client_struct = client::generate_client(service_name, &resolved, types_crate);
    let scoped_clients = scoped::generate_scoped_clients(service_name, &resolved, types_crate);
    let service_traits = client::generate_service_traits(service_name, &resolved, types_crate);
    let trait_impls = client::generate_trait_impls(service_name, &resolved, types_crate);
    let constructors = client::generate_constructors(service_name);

    // Skip handler generation when types_crate is set (client-only mode)
    let handler = if types_crate.is_none() {
        handler::generate_handler(service_name, &resolved, scope_handlers)
    } else {
        proc_macro2::TokenStream::new()
    };

    quote::quote! {
        use std::sync::Arc;
        use hyprstream_rpc::service::ZmqClient as ZmqClientBase;
        use hyprstream_rpc::service::ServiceClient;
        use hyprstream_rpc::service::CallOptions;

        #data_structs
        #response_enum
        #client_struct
        #scoped_clients
        #service_traits
        #trait_impls
        #constructors
        #handler
        #metadata_code
    }
}
