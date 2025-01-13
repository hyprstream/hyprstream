//! # `err-derive`
//!
//! ## Deriving error causes / sources
//!
//! Add an `#[error(source)]` attribute to the field:
//!
//! ```
//! use std::io;
//! use err_derive::Error;
//!
//! /// `MyError::source` will return a reference to the `io_error` field
//! #[derive(Debug, Error)]
//! #[error(display = "An error occurred.")]
//! struct MyError {
//!     #[error(source)]
//!     io_error: io::Error,
//! }
//! #
//! # fn main() {}
//! ```
//!
//! ## Automatic From implementations
//!
//! From<Other> will be automatically implemented when there is a single field
//! in the enum variant or struct, and that field has the `#[source]` attribute.
//!
//! In cases where multiple enum variants have a `#[source]` field of the same type
//! all but one of the variants need to be opted-out from the automatic From implementation (see
//! below).
//!
//! ```
//! use std::io;
//! use err_derive::Error;
//!
//! /// `From<io::Error>` will be implemented for `MyError`
//! #[derive(Debug, Error)]
//! #[error(display = "An error occurred.")]
//! struct MyError {
//!     #[error(from)]
//!     io_error: io::Error,
//! }
//! #
//! # fn main() {}
//! ```
//!
//! ### Opt out of From implementation
//!
//! Use the `#[no_from]` attribute on either the enum or a single variant to opt-out of the
//! automatic From implementation.
//!
//! When `#[no_from]` is set on the enum, you can opt-in individual variants by using `#[from]`
//!
//! ```rust
//! use err_derive::Error;
//! use std::{io, fmt};
//!
//! #[derive(Debug, Error)]
//! enum ClientError {
//!     #[error(display = "regular bad io error {}", _0)]
//!     Io(#[source] io::Error),
//!     #[error(display = "extra bad io error {}", _0)]
//!     // Without #[no_from], this From impl would conflict with the normal Io error
//!     ReallyBadIo(#[error(source, no_from)] io::Error)
//! }
//!
//! #[derive(Debug, Error)]
//! #[error(no_from)] // Don't impl From for any variants by default
//! enum InnerError {
//!     #[error(display = "an error")]
//!     Io(#[source] io::Error),
//!     #[error(display = "an error")]
//!     // Opt-in impl From for a single variant
//!     Formatting(#[error(source, from)] fmt::Error)
//! }
//! ```
//!
//! ### Auto-boxing From implementation
//!
//! If an enum single variant has `Box<T>` as its type, a `From`
//! implementation for `T` will be automatically be generated that wraps it in
//! `Box::new`.
//!
//! ```rust
//! use err_derive::Error;
//! use std::{io, fmt};
//!
//! #[derive(Debug, Error)]
//! enum ClientError {
//!     #[error(display = "io error in a box{}", _0)]
//!     Io(#[error(source)] Box<io::Error>),
//! }
//! ```
//!
//! ## Formatting fields
//!
//! ```rust
//! use std::path::PathBuf;
//! use err_derive::Error;
//!
//! #[derive(Debug, Error)]
//! pub enum FormatError {
//!     #[error(display = "invalid header (expected: {:?}, got: {:?})", expected, found)]
//!     InvalidHeader {
//!         expected: String,
//!         found: String,
//!     },
//!     // Note that tuple fields need to be prefixed with `_`
//!     #[error(display = "missing attribute: {:?}", _0)]
//!     MissingAttribute(String),
//!
//! }
//!
//! #[derive(Debug, Error)]
//! pub enum LoadingError {
//!     #[error(display = "could not decode file")]
//!     FormatError(#[error(source)] #[error(from)] FormatError),
//!     #[error(display = "could not find file: {:?}", path)]
//!     NotFound { path: PathBuf },
//! }
//! #
//! # fn main() {}
//! ```
//!
//! ## Printing the error
//!
//! ```
//! use std::error::Error;
//!
//! fn print_error(e: &dyn Error) {
//!     eprintln!("error: {}", e);
//!     let mut cause = e.source();
//!     while let Some(e) = cause {
//!         eprintln!("caused by: {}", e);
//!         cause = e.source();
//!     }
//! }
//! ```
//!

extern crate proc_macro;
extern crate syn;

use quote::quote;
use synstructure::decl_derive;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use syn::spanned::Spanned;

extern crate proc_macro_error;

use proc_macro_error::{abort, proc_macro_error};
use syn::Attribute;

decl_derive!([Error, attributes(error, source, cause)] => #[proc_macro_error] error_derive);

fn error_derive(s: synstructure::Structure) -> TokenStream {
    let source_body = s.each_variant(|v| {
        if let Some(source) = v.bindings().iter().find(|binding| {
            has_attr(&binding.ast().attrs, "source") || has_attr(&binding.ast().attrs, "cause")
        })
        // TODO https://github.com/rust-lang/rust/issues/54140 deprecate cause with warning
        {
            quote!(return Some(#source as & ::std::error::Error))
        } else {
            quote!(return None)
        }
    });

    let source_method = quote! {
        #[allow(unreachable_code)]
        fn source(&self) -> ::std::option::Option<&(::std::error::Error + 'static)> {
            match *self { #source_body }
            None
        }
    };

    let cause_method = quote! {
        #[allow(unreachable_code)]
        fn cause(&self) -> ::std::option::Option<& ::std::error::Error> {
            match *self { #source_body }
            None
        }
    };

    let error = if cfg!(feature = "std") {
        s.unbound_impl(
            quote!(::std::error::Error),
            quote! {
                fn description(&self) -> &str {
                    "description() is deprecated; use Display"
                }

                #cause_method
                #source_method
            },
        )
    } else {
        quote!()
    };

    let display_body = display_body(&s);
    let display = s.unbound_impl(
        quote!(::core::fmt::Display),
        quote! {
            #[allow(unreachable_code)]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match *self { #display_body }
                write!(f, "An error has occurred.")
            }
        },
    );

    let from = from_body(&s);

    quote!(#error #display #from).into()
}

fn display_body(s: &synstructure::Structure) -> TokenStream2 {
    s.each_variant(|v| {
        let span = v.ast().ident.span();
        let msg = match find_error_msg(&v.ast().attrs) {
            Some(msg) => msg,
            None => abort!(span, "Variant is missing display attribute."),
        };
        if msg.nested.is_empty() {
            abort!(span, "Expected at least one argument to error attribute");
        }

        let format_string = match msg.nested[0] {
            syn::NestedMeta::Meta(syn::Meta::NameValue(ref nv))
                if nv
                    .path
                    .get_ident()
                    .map_or(false, |ident| ident == "display") =>
            {
                nv.lit.clone()
            }
            _ => abort!(
                msg.nested.span(),
                "Error attribute must begin `display = \"\"` to control the Display message."
            ),
        };
        let args = msg.nested.iter().skip(1).map(|arg| match *arg {
            syn::NestedMeta::Lit(syn::Lit::Int(ref i)) => {
                let bi = &v.bindings()[i
                    .base10_parse::<usize>()
                    .unwrap_or_else(|_| abort!(i.span(), "integer literal overflows usize"))];
                quote!(#bi)
            }
            syn::NestedMeta::Meta(syn::Meta::Path(ref path)) => {
                let id = match path.get_ident() {
                    Some(id) => id,
                    // Allows std::u8::MAX (for example)
                    None => return quote!(#arg),
                };
                let id_s = id.to_string();
                if id_s.starts_with('_') {
                    if let Ok(idx) = id_s[1..].parse::<usize>() {
                        let bi = match v.bindings().get(idx) {
                            Some(bi) => bi,
                            None => {
                                abort!(
                                    id.span(),
                                    "display attempted to access field `{}` in `{}::{}` which \
                                     does not exist (there {} {} field{})",
                                    idx,
                                    s.ast().ident,
                                    v.ast().ident,
                                    if v.bindings().len() != 1 { "are" } else { "is" },
                                    v.bindings().len(),
                                    if v.bindings().len() != 1 { "s" } else { "" }
                                );
                            }
                        };
                        return quote!(#bi);
                    }
                }
                for bi in v.bindings() {
                    if bi.ast().ident.as_ref() == Some(id) {
                        return quote!(#bi);
                    }
                }
                // Arg is not a field - might be in global scope
                return quote!(#id);
            }
            // Allows u8::max_value() (for example)
            syn::NestedMeta::Meta(syn::Meta::List(ref list)) => return quote!(#list),
            _ => abort!(msg.nested.span(), "Invalid argument to error attribute!"),
        });

        quote! {
            return write!(f, #format_string #(, #args)*)
        }
    })
}

fn find_error_msg(attrs: &[syn::Attribute]) -> Option<syn::MetaList> {
    let mut error_msg = None;
    for attr in attrs {
        if let Ok(meta) = attr.parse_meta() {
            if meta
                .path()
                .get_ident()
                .map_or(false, |ident| ident == "error")
            {
                let span = attr.span();
                if error_msg.is_some() {
                    abort!(span, "Cannot have two display attributes")
                } else if let syn::Meta::List(list) = meta {
                    error_msg = Some(list);
                } else {
                    abort!(span, "error attribute must take a list in parentheses")
                }
            }
        }
    }
    error_msg
}

fn has_attr(attributes: &[Attribute], attr_name: &str) -> bool {
    let mut found_attr = false;
    for attr in attributes {
        if let Ok(meta) = attr.parse_meta() {
            if meta
                .path()
                .get_ident()
                .map_or(false, |ident| ident == attr_name)
            {
                if found_attr {
                    abort!(attr.span(), "Cannot have two `{}` attributes", attr_name);
                }
                found_attr = true;
            }

            if meta
                .path()
                .get_ident()
                .map_or(false, |ident| ident == "error")
            {
                if let syn::Meta::List(ref list) = meta {
                    for pair in list.nested.iter() {
                        if let syn::NestedMeta::Meta(syn::Meta::Path(path)) = pair {
                            let is_attr_name = |ident: &syn::Ident| {
                                ident.to_string().split(", ").any(|part| part == attr_name)
                            };

                            if path.get_ident().map_or(false, is_attr_name) {
                                if found_attr {
                                    abort!(
                                        path.span(),
                                        "Cannot have two `{}` attributes",
                                        attr_name
                                    );
                                }
                                found_attr = true;
                            }
                        }
                    }
                }
            }
        }
    }
    found_attr
}

// If `ty` is a `Box<T>`, return `Some(T)`. Otherwise return `None`.
fn box_content_type(ty: &syn::Type) -> Option<&syn::Type> {
    let path = match ty {
        syn::Type::Path(p) => p,
        _ => return None,
    };

    if path.path.segments.len() != 1 {
        return None;
    }

    let seg = &path.path.segments[0];
    if seg.ident != "Box" {
        return None;
    }

    let args = match &seg.arguments {
        syn::PathArguments::AngleBracketed(args) => args,
        _ => return None,
    };

    if args.args.len() != 1 {
        return None;
    }

    let inner_ty = match &args.args[0] {
        syn::GenericArgument::Type(ty) => ty,
        _ => return None,
    };

    Some(inner_ty)
}

fn from_body(s: &synstructure::Structure) -> TokenStream2 {
    let default_from = !has_attr(&s.ast().attrs, "no_from");
    let mut from_types = Vec::new();
    let froms = s.variants().iter().flat_map(|v| {
        let span = v.ast().ident.span();
        if let Some((from, is_explicit)) = v.bindings().iter().find_map(|binding| {
            let is_explicit = has_attr(&binding.ast().attrs, "from");
            let is_source = has_attr(&binding.ast().attrs, "source");
            // TODO https://github.com/rust-lang/rust/issues/54140 deprecate cause with warning
            let is_cause = has_attr(&binding.ast().attrs, "cause");
            let exclude = has_attr(&binding.ast().attrs, "no_from");

            if is_source && is_cause {
                abort!(
                    span,
                    "#[error(cause)] is deprecated, use #[error(source)] instead"
                )
            }

            let is_source = is_source || is_cause;

            if ((default_from && is_source) || is_explicit) && !exclude {
                Some((binding, is_explicit))
            } else {
                None
            }
        }) {
            if v.bindings().len() > 1 {
                if is_explicit {
                    abort!(
                        span,
                        "Variants containing `from` can only contain a single field"
                    );
                } else {
                    return vec![];
                }
            }

            let from_ident = &from.ast().ty;

            if from_types
                .iter()
                .any(|existing_from_type| *existing_from_type == from_ident)
            {
                abort!(
                    from_ident.span(),
                    "`from` can only be applied for a type once{}",
                    if is_explicit {
                        ""
                    } else {
                        ", hint: use #[error(no_from)] to disable automatic From derive"
                    }
                );
            }

            from_types.push(from_ident);
            let construct = v.construct(|_, _| quote! {from});

            let mut from_impls = vec![s.unbound_impl(
                quote! {::core::convert::From<#from_ident>},
                quote! {
                    fn from(from: #from_ident) -> Self {
                        #construct
                    }
                },
            )];

            if let Some(box_content_ty) = box_content_type(from_ident) {
                from_impls.push(s.unbound_impl(
                    quote! {::core::convert::From<#box_content_ty>},
                    quote! {
                        fn from(from: #box_content_ty) -> Self {
                            Box::new(from).into()
                        }
                    },
                ));
            }

            from_impls
        } else {
            vec![]
        }
    });

    quote! {
        #(#froms)*
    }
}
