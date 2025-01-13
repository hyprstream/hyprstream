use proc_macro_error2::{abort, ResultExt};
use syn::spanned::Spanned;
use syn::{Lit, Meta, MetaNameValue};

pub const ARROW_FIELD: &'static str = "arrow_field";
pub const FIELD_TYPE: &'static str = "type";
pub const FIELD_NAME: &'static str = "name";
pub const UNION_MODE: &'static str = "mode";

pub fn field_type(field: &syn::Field) -> syn::Type {
    for attr in &field.attrs {
        if let Ok(meta) = attr.parse_meta() {
            if meta.path().is_ident(ARROW_FIELD) {
                if let Meta::List(list) = meta {
                    for nested in list.nested {
                        if let syn::NestedMeta::Meta(meta) = nested {
                            match meta {
                                Meta::NameValue(MetaNameValue {
                                    lit: Lit::Str(string),
                                    path,
                                    ..
                                }) => {
                                    if path.is_ident(FIELD_TYPE) {
                                        return syn::parse_str(&string.value()).unwrap_or_abort();
                                    } else if path.is_ident(FIELD_NAME) {
                                        return syn::parse_str(&string.value()).unwrap_or_abort();
                                    }
                                }
                                _ => {
                                    abort!(meta.span(), "Unexpected attribute");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    field.ty.clone()
}

pub fn union_type(input: &syn::DeriveInput) -> bool {
    for attr in &input.attrs {
        if let Ok(meta) = attr.parse_meta() {
            if meta.path().is_ident(ARROW_FIELD) {
                if let Meta::List(list) = meta {
                    for nested in list.nested {
                        if let syn::NestedMeta::Meta(meta) = nested {
                            match meta {
                                Meta::NameValue(MetaNameValue {
                                    lit: Lit::Str(string),
                                    path,
                                    ..
                                }) => {
                                    if path.is_ident(UNION_MODE) {
                                        match string.value().as_ref() {
                                            "sparse" => {
                                                return false;
                                            }
                                            "dense" => {
                                                return true;
                                            }
                                            _ => {
                                                abort!(path.span(), "Unexpected value for mode")
                                            }
                                        }
                                    }
                                }
                                _ => {
                                    abort!(meta.span(), "Unexpected attribute");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    abort!(input.span(), "Missing mode attribute for enum");
}

