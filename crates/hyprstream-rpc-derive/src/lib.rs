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
                if let Ok(expr) = syn::parse2::<Expr>(list.tokens.clone()) {
                    if let Expr::Assign(assign) = expr {
                        if let Expr::Path(path) = &*assign.left {
                            if path.path.is_ident("rename") {
                                if let Expr::Lit(lit) = &*assign.right {
                                    if let syn::Lit::Str(s) = &lit.lit {
                                        return Some(s.value());
                                    }
                                }
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
    s.to_string()
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
