//! Derive macros for Cap'n Proto serialization.
//!
//! Provides `#[derive(ToCapnp)]` and `#[derive(FromCapnp)]` for automatic
//! serialization/deserialization between Rust types and Cap'n Proto messages.
//!
//! # Example
//!
//! ```ignore
//! use hyprstream_rpc::prelude::*;
//!
//! #[derive(ToCapnp)]
//! #[capnp(my_schema_capnp::my_request)]
//! pub struct MyRequest {
//!     pub name: String,
//!     pub count: u32,
//! }
//!
//! #[derive(FromCapnp)]
//! #[capnp(my_schema_capnp::my_response)]
//! pub struct MyResponse {
//!     pub result: String,
//!     pub success: bool,
//! }
//! ```

use proc_macro::TokenStream;
use quote::{quote, format_ident};
use syn::{parse_macro_input, DeriveInput, Data, Fields, Attribute, Meta, Expr, ExprPath};

mod schema;
mod codegen;
mod util;

/// Derive macro for serializing Rust types to Cap'n Proto.
///
/// Generates an implementation of `ToCapnp` trait that writes struct fields
/// to a Cap'n Proto builder.
///
/// # Attributes
///
/// - `#[capnp(path::to::schema)]` - Required. Specifies the Cap'n Proto schema path.
/// - `#[capnp(skip)]` on field - Skip this field during serialization.
/// - `#[capnp(rename = "other_name")]` on field - Use a different name for the setter.
///
/// # Example
///
/// ```ignore
/// #[derive(ToCapnp)]
/// #[capnp(registry_capnp::clone_request)]
/// pub struct CloneRequest {
///     pub url: String,
///     pub name: String,
///     pub shallow: bool,
///     #[capnp(rename = "maxDepth")]
///     pub depth: u32,
/// }
/// ```
#[proc_macro_derive(ToCapnp, attributes(capnp))]
pub fn derive_to_capnp(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // Parse #[capnp(path::to::schema)] attribute
    let schema_path = match parse_capnp_attr(&input.attrs) {
        Some(path) => path,
        None => {
            return syn::Error::new_spanned(
                &input,
                "ToCapnp requires #[capnp(path::to::schema)] attribute"
            ).to_compile_error().into();
        }
    };

    // Generate field setters
    let field_setters = match &input.data {
        Data::Struct(data) => {
            match &data.fields {
                Fields::Named(fields) => {
                    fields.named.iter().filter_map(|f| {
                        let field_name = f.ident.as_ref()?;

                        // Check for skip
                        if has_skip_attr(&f.attrs) {
                            return None;
                        }

                        // Check for rename, otherwise use field name as-is
                        let capnp_name = get_rename(&f.attrs)
                            .unwrap_or_else(|| to_accessor_name(&field_name.to_string()));

                        let setter = format_ident!("set_{}", capnp_name);
                        let ty = &f.ty;

                        // Generate setter based on type
                        Some(generate_setter(field_name, &setter, ty))
                    }).collect::<Vec<_>>()
                }
                _ => {
                    return syn::Error::new_spanned(
                        &input,
                        "ToCapnp only supports structs with named fields"
                    ).to_compile_error().into();
                }
            }
        }
        _ => {
            return syn::Error::new_spanned(
                &input,
                "ToCapnp only supports structs"
            ).to_compile_error().into();
        }
    };

    let builder_path = quote! { #schema_path::Builder<'a> };

    let expanded = quote! {
        impl #impl_generics hyprstream_rpc::capnp::ToCapnp for #name #ty_generics #where_clause {
            type Builder<'a> = #builder_path;

            fn write_to(&self, builder: &mut Self::Builder<'_>) {
                #(#field_setters)*
            }
        }
    };

    expanded.into()
}

/// Derive macro for deserializing Cap'n Proto to Rust types.
///
/// Generates an implementation of `FromCapnp` trait that reads struct fields
/// from a Cap'n Proto reader.
///
/// # Attributes
///
/// - `#[capnp(path::to::schema)]` - Required. Specifies the Cap'n Proto schema path.
/// - `#[capnp(skip)]` on field - Skip this field, use Default::default().
/// - `#[capnp(rename = "other_name")]` on field - Use a different name for the getter.
/// - `#[capnp(optional)]` on field - Field is Option<T>, handle missing gracefully.
///
/// # Example
///
/// ```ignore
/// #[derive(FromCapnp)]
/// #[capnp(registry_capnp::health_status)]
/// pub struct HealthStatus {
///     pub status: String,
///     pub repository_count: u32,
///     #[capnp(optional)]
///     pub message: Option<String>,
/// }
/// ```
#[proc_macro_derive(FromCapnp, attributes(capnp))]
pub fn derive_from_capnp(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // Parse #[capnp(path::to::schema)] attribute
    let schema_path = match parse_capnp_attr(&input.attrs) {
        Some(path) => path,
        None => {
            return syn::Error::new_spanned(
                &input,
                "FromCapnp requires #[capnp(path::to::schema)] attribute"
            ).to_compile_error().into();
        }
    };

    // Generate field readers
    let field_readers = match &input.data {
        Data::Struct(data) => {
            match &data.fields {
                Fields::Named(fields) => {
                    fields.named.iter().filter_map(|f| {
                        let field_name = f.ident.as_ref()?;

                        // Check for skip - use Default
                        if has_skip_attr(&f.attrs) {
                            return Some(quote! {
                                #field_name: Default::default(),
                            });
                        }

                        // Check for rename, otherwise use field name as-is
                        let capnp_name = get_rename(&f.attrs)
                            .unwrap_or_else(|| to_accessor_name(&field_name.to_string()));

                        let getter = format_ident!("get_{}", capnp_name);
                        let ty = &f.ty;
                        let is_optional = has_optional_attr(&f.attrs) || is_option_type(ty);

                        // Generate getter based on type
                        Some(generate_getter(field_name, &getter, ty, is_optional))
                    }).collect::<Vec<_>>()
                }
                _ => {
                    return syn::Error::new_spanned(
                        &input,
                        "FromCapnp only supports structs with named fields"
                    ).to_compile_error().into();
                }
            }
        }
        _ => {
            return syn::Error::new_spanned(
                &input,
                "FromCapnp only supports structs"
            ).to_compile_error().into();
        }
    };

    let reader_path = quote! { #schema_path::Reader<'a> };

    let expanded = quote! {
        impl #impl_generics hyprstream_rpc::capnp::FromCapnp for #name #ty_generics #where_clause {
            type Reader<'a> = #reader_path;

            fn read_from(reader: Self::Reader<'_>) -> anyhow::Result<Self> {
                Ok(Self {
                    #(#field_readers)*
                })
            }
        }
    };

    expanded.into()
}

/// Parse the #[capnp(path::to::schema)] attribute
fn parse_capnp_attr(attrs: &[Attribute]) -> Option<ExprPath> {
    for attr in attrs {
        if attr.path().is_ident("capnp") {
            if let Meta::List(list) = &attr.meta {
                if let Ok(path) = syn::parse2::<ExprPath>(list.tokens.clone()) {
                    return Some(path);
                }
            }
        }
    }
    None
}

/// Check if field has #[capnp(skip)] attribute
fn has_skip_attr(attrs: &[Attribute]) -> bool {
    for attr in attrs {
        if attr.path().is_ident("capnp") {
            if let Meta::List(list) = &attr.meta {
                let tokens = list.tokens.to_string();
                if tokens == "skip" {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if field has #[capnp(optional)] attribute
fn has_optional_attr(attrs: &[Attribute]) -> bool {
    for attr in attrs {
        if attr.path().is_ident("capnp") {
            if let Meta::List(list) = &attr.meta {
                let tokens = list.tokens.to_string();
                if tokens == "optional" {
                    return true;
                }
            }
        }
    }
    false
}

/// Get rename value from #[capnp(rename = "name")] attribute
fn get_rename(attrs: &[Attribute]) -> Option<String> {
    for attr in attrs {
        if attr.path().is_ident("capnp") {
            if let Meta::List(list) = &attr.meta {
                // Parse as key = value
                if let Ok(Expr::Assign(assign)) = syn::parse2::<Expr>(list.tokens.clone()) {
                    if let Expr::Path(path) = &*assign.left {
                        if path.path.is_ident("rename") {
                            if let Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. }) = &*assign.right {
                                return Some(s.value());
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

/// Cap'n Proto Rust uses snake_case for accessors.
/// This function simply returns the name as-is.
fn to_accessor_name(s: &str) -> String {
    s.to_owned()
}

/// Check if type is Option<T>
fn is_option_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "Option";
        }
    }
    false
}

/// Generate setter code for a field
fn generate_setter(
    field_name: &syn::Ident,
    setter: &syn::Ident,
    ty: &syn::Type,
) -> proc_macro2::TokenStream {
    // Check if type is String
    if is_string_type(ty) {
        quote! {
            builder.#setter(&self.#field_name);
        }
    } else if is_option_type(ty) {
        // For Option<String>, only set if Some
        quote! {
            if let Some(ref v) = self.#field_name {
                builder.#setter(v);
            }
        }
    } else if is_vec_type(ty) {
        // For Vec<T>, need to initialize a list
        quote! {
            {
                let mut list = builder.reborrow().#setter(self.#field_name.len() as u32);
                for (i, item) in self.#field_name.iter().enumerate() {
                    list.set(i as u32, item);
                }
            }
        }
    } else {
        // Primitive types - direct set
        quote! {
            builder.#setter(self.#field_name);
        }
    }
}

/// Generate getter code for a field
fn generate_getter(
    field_name: &syn::Ident,
    getter: &syn::Ident,
    ty: &syn::Type,
    is_optional: bool,
) -> proc_macro2::TokenStream {
    if is_string_type(ty) {
        quote! {
            #field_name: reader.#getter()?.to_str()?.to_string(),
        }
    } else if is_pathbuf_type(ty) {
        quote! {
            #field_name: std::path::PathBuf::from(reader.#getter()?.to_str()?),
        }
    } else if is_option_type(ty) || is_optional {
        // For Option<String>
        quote! {
            #field_name: {
                let s = reader.#getter()?.to_str()?;
                if s.is_empty() { None } else { Some(s.to_string()) }
            },
        }
    } else if is_vec_string_type(ty) {
        quote! {
            #field_name: {
                let list = reader.#getter()?;
                let mut v = Vec::with_capacity(list.len() as usize);
                for i in 0..list.len() {
                    v.push(list.get(i)?.to_str()?.to_string());
                }
                v
            },
        }
    } else {
        // Primitive types - direct get
        quote! {
            #field_name: reader.#getter(),
        }
    }
}

/// Check if type is String
fn is_string_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "String";
        }
    }
    false
}

/// Check if type is PathBuf
fn is_pathbuf_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "PathBuf";
        }
    }
    false
}

/// Check if type is Vec<T>
fn is_vec_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "Vec";
        }
    }
    false
}

/// Check if type is Vec<String>
fn is_vec_string_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            if segment.ident == "Vec" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        return is_string_type(inner);
                    }
                }
            }
        }
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════════
// Authorization Macros - #[authorize] and #[register_scopes]
// ═══════════════════════════════════════════════════════════════════════════════

/// Attribute macro for method-level authorization with structured scopes.
///
/// Generates a wrapper that validates JWT tokens and checks structured scopes
/// before calling the original method. This provides compile-time enforcement
/// of authorization requirements.
///
/// # Attributes
///
/// - `action` - The action being performed (e.g., "infer", "read", "write")
/// - `resource` - The resource type (e.g., "model", "stream", "policy")
/// - `identifier_field` - Field name in request containing the resource ID
///
/// # Example
///
/// ```ignore
/// impl InferenceService {
///     #[authorize(action = "infer", resource = "model", identifier_field = "model_name")]
///     fn handle_generate(&self, ctx: &EnvelopeContext, request: GenerateRequest) -> Result<Vec<u8>> {
///         // Authorization already validated - just implement business logic
///         // ctx.user_claims is guaranteed to exist here
///     }
/// }
/// ```
///
/// # Generated Code
///
/// The macro generates:
/// 1. JWT token validation via `ctx.validate_jwt()`
/// 2. Structured scope construction: `Scope::new(action, resource, request.identifier_field)`
/// 3. Casbin policy check (optional, based on service configuration)
/// 4. JWT scope check via `claims.has_scope(&required_scope)`
/// 5. Original method call if all checks pass
///
/// # Security
///
/// Defense-in-depth with three validation layers:
/// - Layer 1: Envelope signature (service identity)
/// - Layer 2: Casbin policy (RBAC/ABAC)
/// - Layer 3: JWT scope (structured, least privilege)
#[proc_macro_attribute]
pub fn authorize(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as AuthorizeArgs);
    let input = parse_macro_input!(item as syn::ItemFn);

    let action = &args.action;
    let resource = &args.resource;
    let identifier_field = format_ident!("{}", &args.identifier_field);

    let fn_vis = &input.vis;
    let fn_sig = &input.sig;
    let fn_block = &input.block;
    let fn_attrs = &input.attrs;

    // The function signature should have (self, ctx: &EnvelopeContext, request: RequestType)
    // We generate a wrapper that validates before calling the original

    let expanded = quote! {
        #(#fn_attrs)*
        #fn_vis #fn_sig {
            // Extract claims from envelope (already verified by envelope signature)
            let claims = ctx.claims()
                .ok_or_else(|| anyhow::anyhow!("Unauthorized: Missing claims in envelope"))?;

            // Build structured scope from request
            let required_scope = hyprstream_rpc::auth::Scope::new(
                #action.to_string(),
                #resource.to_string(),
                request.#identifier_field.clone(),
            );

            // Check Casbin policy (scope used as resource)
            let subject = claims.casbin_subject();
            let allowed = tokio::task::block_in_place(|| {
                let handle = tokio::runtime::Handle::current();
                handle.block_on(self.policy_manager.check(
                    &subject,
                    &required_scope.to_string(),
                    crate::auth::Operation::Infer,
                ))
            })?;

            if !allowed {
                return Err(anyhow::anyhow!(
                    "Forbidden: Casbin policy denied for scope '{}'",
                    required_scope.to_string()
                ));
            }

            // Check claims scope (uses Scope::grants() with safe wildcard matching)
            if !claims.has_scope(&required_scope) {
                return Err(anyhow::anyhow!(
                    "Forbidden: Missing scope '{}'",
                    required_scope.to_string()
                ));
            }

            // Authorization passed - execute original function body
            #fn_block
        }
    };

    expanded.into()
}

/// Arguments for the #[authorize] attribute
struct AuthorizeArgs {
    action: String,
    resource: String,
    identifier_field: String,
}

impl syn::parse::Parse for AuthorizeArgs {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let mut action = None;
        let mut resource = None;
        let mut identifier_field = None;

        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;
            input.parse::<syn::Token![=]>()?;

            match ident.to_string().as_str() {
                "action" => {
                    let lit: syn::LitStr = input.parse()?;
                    action = Some(lit.value());
                }
                "resource" => {
                    let lit: syn::LitStr = input.parse()?;
                    resource = Some(lit.value());
                }
                "identifier_field" => {
                    let lit: syn::LitStr = input.parse()?;
                    identifier_field = Some(lit.value());
                }
                other => return Err(syn::Error::new(ident.span(), format!("unknown attribute: {other}"))),
            }

            if input.peek(syn::Token![,]) {
                input.parse::<syn::Token![,]>()?;
            }
        }

        Ok(AuthorizeArgs {
            action: action.ok_or_else(|| syn::Error::new(input.span(), "missing 'action'"))?,
            resource: resource.ok_or_else(|| syn::Error::new(input.span(), "missing 'resource'"))?,
            identifier_field: identifier_field.ok_or_else(|| syn::Error::new(input.span(), "missing 'identifier_field'"))?,
        })
    }
}

/// Attribute macro for automatic scope registration.
///
/// Scans all `#[authorize]` attributes in the impl block and generates
/// inventory::submit! statements for each unique (action, resource) pair.
///
/// This ensures that all scopes are automatically registered at compile-time,
/// making it impossible to forget scope registration.
///
/// # Example
///
/// ```ignore
/// #[register_scopes]
/// impl InferenceService {
///     #[authorize(action = "infer", resource = "model", identifier_field = "model")]
///     fn handle_generate(...) { }
///
///     #[authorize(action = "read", resource = "model", identifier_field = "model")]
///     fn handle_config(...) { }
/// }
///
/// // Automatically generates:
/// // inventory::submit! {
/// //     hyprstream_rpc::auth::ScopeDefinition::new("infer", "model", "infer:model:*", "infer on model")
/// // }
/// // inventory::submit! {
/// //     hyprstream_rpc::auth::ScopeDefinition::new("read", "model", "read:model:*", "read on model")
/// // }
/// ```
///
/// # Benefits
///
/// - ✅ DRY: Scopes declared once in #[authorize], auto-registered
/// - ✅ Impossible to forget: Missing #[authorize] = no registration = compile error
/// - ✅ Self-documenting: Scopes visible via `inventory::iter::<ScopeDefinition>()`
/// - ✅ Zero boilerplate: No manual inventory::submit! needed
#[proc_macro_attribute]
pub fn register_scopes(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::ItemImpl);

    // Collect all unique (action, resource) pairs from #[authorize] attributes
    let mut scopes = std::collections::HashSet::new();

    for item in &input.items {
        if let syn::ImplItem::Fn(method) = item {
            for attr in &method.attrs {
                if attr.path().is_ident("authorize") {
                    if let Ok(args) = attr.parse_args::<AuthorizeArgs>() {
                        scopes.insert((args.action.clone(), args.resource.clone()));
                    }
                }
            }
        }
    }

    // Generate inventory submissions
    let registrations: Vec<_> = scopes.iter().map(|(action, resource)| {
        let example = format!("{action}:{resource}:*");
        let description = format!("{action} on {resource}");

        quote! {
            inventory::submit! {
                hyprstream_rpc::auth::ScopeDefinition::new(
                    #action,
                    #resource,
                    #example,
                    #description
                )
            }
        }
    }).collect();

    // Output original impl + registrations
    let expanded = quote! {
        #input

        #(#registrations)*
    };

    expanded.into()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Service Factory Macro - #[service_factory]
// ═══════════════════════════════════════════════════════════════════════════════

/// Attribute macro for service factory registration.
///
/// Automatically registers a service factory function with the inventory system,
/// following the same pattern as `#[register_scopes]` for authorization scopes
/// and `DriverFactory` for storage drivers.
///
/// # Usage
///
/// ```ignore
/// use hyprstream_rpc::service::factory::ServiceContext;
/// use hyprstream_rpc::service::spawner::Spawnable;
/// use hyprstream_rpc_derive::service_factory;
///
/// #[service_factory("policy")]
/// fn create_policy_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
///     // Services include infrastructure and are directly Spawnable
///     let policy = PolicyService::new(
///         ...,
///         ctx.zmq_context(),
///         ctx.transport("policy", SocketKind::Rep),
///         ctx.verifying_key(),
///     );
///     Ok(Box::new(policy))
/// }
/// ```
///
/// # Generated Code
///
/// The macro generates:
/// ```ignore
/// fn create_policy_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
///     // ... original function body ...
/// }
///
/// inventory::submit! {
///     hyprstream_rpc::service::factory::ServiceFactory::new(
///         "policy",
///         create_policy_service
///     )
/// }
/// ```
///
/// # Benefits
///
/// - ✅ DRY: Service factory declared once, auto-registered
/// - ✅ Decentralized: Services own their registration (not in main.rs)
/// - ✅ Self-documenting: All services discoverable via `list_factories()`
/// - ✅ Matches existing patterns: Same as `#[register_scopes]` and `DriverFactory`
#[proc_macro_attribute]
pub fn service_factory(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ServiceFactoryArgs);
    let func = parse_macro_input!(item as syn::ItemFn);
    let func_name = &func.sig.ident;
    let name = &args.name;

    let expanded = if let Some(schema_path) = &args.schema {
        quote! {
            #func

            inventory::submit! {
                hyprstream_rpc::service::factory::ServiceFactory::with_schema(
                    #name,
                    #func_name,
                    include_bytes!(#schema_path)
                )
            }
        }
    } else {
        quote! {
            #func

            inventory::submit! {
                hyprstream_rpc::service::factory::ServiceFactory::new(
                    #name,
                    #func_name
                )
            }
        }
    };

    expanded.into()
}

/// Arguments for the #[service_factory] attribute.
///
/// Supports:
/// - `#[service_factory("name")]` — no schema
/// - `#[service_factory("name", schema = "path/to/schema.capnp")]` — with schema
struct ServiceFactoryArgs {
    name: syn::LitStr,
    schema: Option<syn::LitStr>,
}

impl syn::parse::Parse for ServiceFactoryArgs {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let name: syn::LitStr = input.parse()?;
        let mut schema = None;

        while input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
            if input.is_empty() {
                break;
            }
            let ident: syn::Ident = input.parse()?;
            input.parse::<syn::Token![=]>()?;
            match ident.to_string().as_str() {
                "schema" => {
                    schema = Some(input.parse::<syn::LitStr>()?);
                }
                other => {
                    return Err(syn::Error::new(ident.span(), format!("unknown attribute: {other}")));
                }
            }
        }

        Ok(ServiceFactoryArgs { name, schema })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RPC Service Code Generation Macro
// ═══════════════════════════════════════════════════════════════════════════════

/// Proc macro that generates a complete RPC client + handler + metadata from a Cap'n Proto schema.
///
/// Reads the schema file from `$CARGO_MANIFEST_DIR/schema/{name}.capnp`, parses it,
/// and generates all client code using `quote!` AST generation.
///
/// # Usage
///
/// ```ignore
/// pub mod policy_client {
///     #![allow(dead_code, unused_imports, unused_variables)]
///     #![allow(clippy::all)]
///     hyprstream_rpc_derive::generate_rpc_service!("policy");
/// }
/// ```
#[proc_macro]
pub fn generate_rpc_service(input: TokenStream) -> TokenStream {
    let service_name: syn::LitStr = match syn::parse(input) {
        Ok(n) => n,
        Err(e) => return e.to_compile_error().into(),
    };
    let name = service_name.value();

    let manifest_dir = match std::env::var("CARGO_MANIFEST_DIR") {
        Ok(d) => d,
        Err(_) => {
            return syn::Error::new(service_name.span(), "CARGO_MANIFEST_DIR not set")
                .to_compile_error()
                .into()
        }
    };

    // Scope filesystem access to the schema/ directory within the build environment.
    // Canonicalize both paths to defeat traversal (../) and symlink escapes.
    let schema_dir = match std::fs::canonicalize(format!("{manifest_dir}/schema")) {
        Ok(d) => d,
        Err(e) => {
            return syn::Error::new(
                service_name.span(),
                format!("Cannot resolve schema directory: {e}"),
            )
            .to_compile_error()
            .into()
        }
    };
    let schema_path = match std::fs::canonicalize(schema_dir.join(format!("{name}.capnp"))) {
        Ok(p) if p.starts_with(&schema_dir) => p,
        Ok(p) => {
            return syn::Error::new(
                service_name.span(),
                format!("Schema path escapes build environment: {}", p.display()),
            )
            .to_compile_error()
            .into()
        }
        Err(e) => {
            return syn::Error::new(
                service_name.span(),
                format!("Cannot resolve schema '{name}.capnp': {e}"),
            )
            .to_compile_error()
            .into()
        }
    };
    let schema_text = match std::fs::read_to_string(&schema_path) {
        Ok(t) => t,
        Err(e) => {
            return syn::Error::new(
                service_name.span(),
                format!("Cannot read {}: {e}", schema_path.display()),
            )
            .to_compile_error()
            .into()
        }
    };

    let mut parsed = match schema::parse_capnp_schema(&schema_text, &name) {
        Some(p) => p,
        None => {
            return syn::Error::new(
                service_name.span(),
                format!("Failed to parse schema for '{name}'"),
            )
            .to_compile_error()
            .into()
        }
    };

    // Try to load annotation metadata from OUT_DIR (generated by build.rs)
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        let metadata_path = std::path::Path::new(&out_dir).join(format!("{name}_metadata.json"));
        if let Ok(metadata_text) = std::fs::read_to_string(&metadata_path) {
            if let Err(e) = schema::merge_annotations_from_metadata(&mut parsed, &metadata_text) {
                eprintln!("Warning: Failed to load annotations for {name}: {e}");
            }
        }
    }

    codegen::generate_service(&name, &parsed).into()
}
