//! Code generation from parsed Cap'n Proto schemas using quote!.

pub mod client;
pub mod data;
pub mod dispatch;
pub mod handler;
pub mod leaf;
pub mod metadata;
pub mod scoped;
pub mod vfs;

use crate::resolve::ResolvedSchema;
use crate::schema::types::ParsedSchema;

/// Generate client-only code from a parsed schema.
///
/// Emits data structs, response enum, client struct, scoped clients, client
/// traits, JSON dispatch, and metadata. Does NOT emit server-side handlers or
/// service implementation code.
/// Compiles to all targets including wasm32.
pub fn generate_client_only(service_name: &str, schema: &ParsedSchema, types_crate: Option<&syn::Path>) -> proc_macro2::TokenStream {
    let resolved = ResolvedSchema::from(schema);
    let is_data_only = resolved.raw.request_variants.is_empty();

    let data_structs = data::generate_data_structs(&resolved, service_name, types_crate);

    if is_data_only {
        return quote::quote! { #data_structs };
    }

    let pascal = crate::util::to_pascal_case(service_name);
    let response_type = quote::format_ident!("{}ResponseVariant", pascal);
    let capnp_mod_ident = quote::format_ident!("{}_capnp", service_name);
    let capnp_mod: proc_macro2::TokenStream = match types_crate {
        Some(tc) => quote::quote! { #tc::#capnp_mod_ident },
        None => quote::quote! { crate::#capnp_mod_ident },
    };
    let resp_type = quote::format_ident!("{}", crate::util::to_snake_case(&format!("{pascal}Response")));

    let response_enum = client::generate_response_enum(service_name, &resolved, types_crate);
    let parse_response = client::generate_parse_response_fn(
        &response_type,
        &capnp_mod,
        &resp_type,
        &resolved.raw.response_variants,
        &resolved,
        types_crate,
    );

    // Scoped response enums + parsers (e.g., InferClientResponseVariant with parse_scoped_response)
    let scoped_response_types = scoped::generate_scoped_response_types(service_name, &resolved, types_crate);

    // Schema metadata, JSON dispatch, and render_doc() for client-side
    // documentation and schema-driven tooling.
    let metadata_code = metadata::generate_metadata_client_with_dispatch(service_name, &resolved, types_crate);

    // Transport-agnostic dispatch function — routes method calls by name
    let portable_dispatch = dispatch::generate_portable_dispatch(service_name, &resolved, types_crate);

    // Client struct (cfg-gated: concrete on native, dyn on wasm)
    let client_struct = client::generate_client(service_name, &resolved, types_crate);

    // Scoped client structs (same cfg-gated pattern)
    let scoped_clients = scoped::generate_portable_scoped_clients(service_name, &resolved, types_crate);

    // Public client-side traits live with the generated clients.  Keeping these
    // traits in the contract crate lets consumers depend on a single
    // permissively licensed surface while service crates generate only their
    // handlers and wire dispatch.
    let service_traits = client::generate_service_traits(service_name, &resolved, types_crate);
    let trait_impls = client::generate_trait_impls(service_name, &resolved, types_crate);
    let constructors = client::generate_portable_constructors(service_name);

    // Generated 9P/VFS node table (#539 T2) — must compile to wasm32 too.
    let vfs_mount = vfs::generate_mount(service_name, &resolved);

    quote::quote! {
        #data_structs
        #response_enum

        impl #response_type {
            #parse_response
        }

        #scoped_response_types
        #client_struct
        #scoped_clients
        #service_traits
        #trait_impls
        #constructors
        #metadata_code
        #portable_dispatch
        #vfs_mount
    }
}

/// Generate server-side dispatch code against contracts that were generated in
/// another crate.
///
/// This is the counterpart to [`generate_client_only`].  It deliberately emits
/// no data structs, response enums, client implementations, or JSON client
/// dispatchers: those remain owned by the permissively licensed contract crate.
/// The invocation module imports the contract crate's generated client module
/// (for handler parameter/response types) and Cap'n Proto module, then emits
/// only the handler traits, wire dispatch, metadata, and VFS projection needed
/// by an implementation/service crate.
pub fn generate_server(
    service_name: &str,
    schema: &ParsedSchema,
    types_crate: &syn::Path,
    scope_handlers: bool,
) -> proc_macro2::TokenStream {
    let resolved = ResolvedSchema::from(schema);
    if resolved.raw.request_variants.is_empty() {
        // A data-only schema has no server dispatch surface.  Its contracts
        // are already exported by the contract crate, so there is nothing to
        // emit here.
        return quote::quote! {};
    }

    let client_mod = quote::format_ident!("{}_client", service_name);
    let capnp_mod = quote::format_ident!("{}_capnp", service_name);
    let imports = quote::quote! {
        // Keep contract ownership in the permissively licensed crate.  These
        // imports are intentionally private: this module exposes server
        // traits/dispatch only, never a second public client implementation.
        use #types_crate::#client_mod::*;
        use #types_crate::#capnp_mod;
    };
    let handler = handler::generate_handler(service_name, &resolved, scope_handlers);
    // Scoped metadata functions are emitted alongside the server surface, so
    // the local tree must point at those functions (not at the contract crate's
    // client-only metadata module).
    let metadata = metadata::generate_metadata_client_only(service_name, &resolved, None);
    let vfs_mount = vfs::generate_mount(service_name, &resolved);

    quote::quote! {
        #imports
        #handler
        #metadata
        #vfs_mount
    }
}

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
    let scoped_clients = scoped::generate_portable_scoped_clients(service_name, &resolved, types_crate);
    let scoped_response_types = scoped::generate_scoped_response_types(service_name, &resolved, types_crate);
    let service_traits = client::generate_service_traits(service_name, &resolved, types_crate);
    let trait_impls = client::generate_trait_impls(service_name, &resolved, types_crate);
    let constructors = client::generate_constructors(service_name);

    // Skip handler generation when types_crate is set (client-only mode)
    let handler = if types_crate.is_none() {
        handler::generate_handler(service_name, &resolved, scope_handlers)
    } else {
        proc_macro2::TokenStream::new()
    };

    // Generated 9P/VFS node table (#539 T2).
    let vfs_mount = vfs::generate_mount(service_name, &resolved);

    // Handler code resolves the Cap'n Proto module in the invocation module's
    // scope.  Existing full-service invocations keep their capnp modules at
    // crate root, so import that module before emitting the handler.
    let capnp_import = if types_crate.is_none() {
        let capnp_mod = quote::format_ident!("{}_capnp", service_name);
        quote::quote! { use crate::#capnp_mod; }
    } else {
        quote::quote! {}
    };

    quote::quote! {
        #data_structs
        #response_enum
        #client_struct
        #scoped_clients
        #scoped_response_types
        #service_traits
        #trait_impls
        #constructors
        #capnp_import
        #handler
        #metadata_code
        #vfs_mount
    }
}
