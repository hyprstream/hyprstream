//! Code generation from parsed Cap'n Proto schemas using quote!.

pub mod client;
pub mod data;
pub mod handler;
pub mod metadata;
pub mod scoped;

use crate::schema::types::ParsedSchema;

/// Generate all service code from a parsed schema.
pub fn generate_service(service_name: &str, schema: &ParsedSchema) -> proc_macro2::TokenStream {
    let data_structs = data::generate_data_structs(schema);
    let response_enum = client::generate_response_enum(service_name, schema);
    let client_struct = client::generate_client(service_name, schema);
    let scoped_clients = scoped::generate_scoped_clients(service_name, schema);
    let handler = handler::generate_handler(service_name, schema);
    let metadata_code = metadata::generate_metadata(service_name, schema);

    quote::quote! {
        use std::sync::Arc;
        use hyprstream_rpc::service::ZmqClient as ZmqClientBase;
        use hyprstream_rpc::service::factory::ServiceClient;
        use hyprstream_rpc::service::CallOptions;

        #data_structs
        #response_enum
        #client_struct
        #scoped_clients
        #handler
        #metadata_code
    }
}
