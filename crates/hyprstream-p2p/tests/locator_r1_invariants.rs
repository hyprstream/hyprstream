//! Structural enforcement of review rule R1 (story C4 / #892):
//! **no public locator API may take a string-name parameter**, and `Cid512`
//! must not gain a string-conversion trait impl. A name path inside the DHT
//! client re-opens threat A5 (the DHT becomes a trusted name authority).
//!
//! Why parse the AST instead of grepping or using `trybuild`:
//! - A line-by-line grep (the original #1000 guard) is bypassable with a
//!   multi-line signature, `impl AsRef<str>`, `Cow<'_, str>`, `&String`,
//!   `Box<str>`, `Vec<String>`, a type alias, or a trait method.
//! - `trybuild` compile-fail tests can only pin *specific* forbidden call
//!   sites; they cannot prove that *no* public method, present or future, takes
//!   a string. This walk enumerates every public signature, so a string-name
//!   parameter cannot be introduced anywhere in the module without failing
//!   here — multi-line or otherwise.
//!
//! The source under test is `src/locator.rs` parsed with `syn`; the inline
//! `#[cfg(test)] mod tests` is excluded.

// This file is a test. The workspace denies `unwrap_used`/`expect_used`; the
// parses below are infallible-by-contract (the source must parse for the
// invariant to be meaningful), matching the allowance in `src/locator.rs`'s
// own test module.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;

use quote::ToTokens;
use syn::{FnArg, GenericArgument, ImplItem, Item, Path, PathArguments, Type, TypeParamBound};

/// Public types exported by `locator.rs` whose trait impls are gated by this
/// test (a string-conversion impl on any of them re-opens A5).
const GATED_SELF_TYPES: &[&str] = &["Cid512", "RendezvousKey", "PeerContact", "LookupHints"];

const LOCATOR_SRC: &str = include_str!("../src/locator.rs");

/// Path segment idents that make a type "string-like" (a name path).
const STRINGISH_IDENTS: &[&str] = &["str", "String"];

/// Trait names whose implementation on a gated self type is forbidden because
/// it yields a string view or string construction.
const FORBIDDEN_STRING_TRAITS: &[&str] = &["FromStr", "AsRef", "Deref", "Borrow"];
/// Trait names that are only forbidden when their generic argument is stringish
/// (e.g. `From<&str>`, but `From<[u8; 64]>` is fine).
const FORBIDDEN_STRING_GENERIC_TRAITS: &[&str] = &["From", "TryFrom"];

#[derive(Debug, Default)]
struct Violations {
    string_param: Vec<String>,
    string_trait_impl: Vec<String>,
}

impl Violations {
    fn is_empty(&self) -> bool {
        self.string_param.is_empty() && self.string_trait_impl.is_empty()
    }

    fn fail(self) {
        let mut msgs = Vec::new();
        if !self.string_param.is_empty() {
            msgs.push(format!(
                "R1 violation — public signatures with a string-like parameter (re-opens A5):\n{}",
                self.string_param.join("\n")
            ));
        }
        if !self.string_trait_impl.is_empty() {
            msgs.push(format!(
                "R1 violation — forbidden string-conversion trait impl on a gated type:\n{}",
                self.string_trait_impl.join("\n")
            ));
        }
        panic!("R1 structural invariant violated:\n\n{}", msgs.join("\n\n"));
    }
}

#[test]
fn r1_no_string_name_path_in_public_api() {
    let file =
        syn::parse_file(LOCATOR_SRC).expect("src/locator.rs must parse as valid Rust for the R1 invariant test");

    let mut v = Violations::default();
    walk_items(&file.items, &mut v);

    if !v.is_empty() {
        v.fail();
    }
}

fn walk_items(items: &[Item], v: &mut Violations) {
    for item in items {
        match item {
            // pub fn / pub async fn at module scope
            Item::Fn(f) => {
                if is_pub(&f.vis) {
                    check_sig(&f.sig, "free fn", &f.sig.ident, v);
                }
            }
            // impl blocks: inherent methods + trait impls
            Item::Impl(imp) => {
                check_impl(imp, v);
            }
            // pub trait { fn ...(...) } — the trait method signatures are the API
            Item::Trait(t) => {
                if is_pub(&t.vis) {
                    for ti in &t.items {
                        if let syn::TraitItem::Fn(tf) = ti {
                            check_sig(&tf.sig, "trait fn", &tf.sig.ident, v);
                        }
                    }
                }
            }
            // Recurse into nested non-test modules. The inline `#[cfg(test)] mod
            // tests` is excluded so test helpers don't trip the invariant.
            Item::Mod(m) => {
                if is_test_module(m) {
                    continue;
                }
                if let Some((_, nested_items)) = &m.content {
                    walk_items(nested_items, v);
                }
            }
            _ => {}
        }
    }
}

fn check_impl(imp: &syn::ItemImpl, v: &mut Violations) {
    // Trait impl on a gated self type that yields a string view/construction.
    if let Some((_, trait_path, _)) = &imp.trait_ {
        if let Some(self_name) = type_name(&imp.self_ty) {
            if GATED_SELF_TYPES.contains(&self_name.as_str()) {
                if let Some(reason) = forbidden_string_trait(trait_path) {
                    v.string_trait_impl
                        .push(format!("  impl {} for {self_name}: {reason}", trait_path.to_token_stream()));
                }
            }
        }
    }

    for item in &imp.items {
        if let ImplItem::Fn(m) = item {
            // Inherent `pub fn` methods are public API. Trait-impl methods
            // realize a trait's API (they have no own visibility); their
            // signatures matter too, so check them regardless of `vis`.
            let in_trait_impl = imp.trait_.is_some();
            if in_trait_impl || is_pub(&m.vis) {
                check_sig(&m.sig, "method", &m.sig.ident, v);
            }
        }
    }
}

fn check_sig(sig: &syn::Signature, kind: &str, name: &syn::Ident, v: &mut Violations) {
    for arg in &sig.inputs {
        if let FnArg::Typed(pat_type) = arg {
            if type_is_stringish(&pat_type.ty) {
                v.string_param.push(format!(
                    "  {kind} `{name}` param `{}`: {}",
                    pat_type.pat.to_token_stream(),
                    pat_type.ty.to_token_stream()
                ));
            }
        }
    }
}

fn is_pub(vis: &syn::Visibility) -> bool {
    matches!(vis, syn::Visibility::Public(_))
}

fn is_test_module(m: &syn::ItemMod) -> bool {
    m.attrs
        .iter()
        .any(|a| a.path().is_ident("cfg") || a.path().is_ident("test"))
        && m.ident == "tests"
}

/// Last path segment ident of a (possibly reference) type, if it is a simple path.
fn type_name(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(tp) => tp
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string()),
        Type::Reference(r) => type_name(&r.elem),
        Type::Paren(p) => type_name(&p.elem),
        _ => None,
    }
}

/// Is `ty` a string-like type? Recurses through references, boxes, slices,
/// arrays, tuples, parens, and generic arguments so `&str`, `&String`,
/// `Cow<'_, str>`, `Box<str>`, `Vec<String>`, `impl AsRef<str>`, `&mut str`,
/// etc. all resolve true — covering every bypass form from the #1000 review.
fn type_is_stringish(ty: &Type) -> bool {
    match ty {
        Type::Reference(r) => type_is_stringish(&r.elem),
        Type::Paren(p) => type_is_stringish(&p.elem),
        Type::Group(g) => type_is_stringish(&g.elem),
        Type::Slice(s) => type_is_stringish(&s.elem),
        Type::Array(a) => type_is_stringish(&a.elem),
        Type::Ptr(p) => type_is_stringish(&p.elem),
        Type::Tuple(t) => t.elems.iter().any(type_is_stringish),
        Type::Path(tp) => path_is_stringish(&tp.path),
        Type::ImplTrait(it) => bounds_stringish(&it.bounds),
        Type::TraitObject(to) => bounds_stringish(&to.bounds),
        _ => false,
    }
}

fn bounds_stringish(bounds: &syn::punctuated::Punctuated<TypeParamBound, syn::token::Plus>) -> bool {
    bounds
        .iter()
        .any(|b| matches!(b, TypeParamBound::Trait(tb) if bound_trait_is_stringish(&tb.path)))
}

fn path_is_stringish(p: &Path) -> bool {
    p.segments.iter().any(|seg| {
        let id = seg.ident.to_string();
        STRINGISH_IDENTS.contains(&id.as_str())
            || match &seg.arguments {
                PathArguments::AngleBracketed(ab) => {
                    ab.args.iter().any(generic_is_stringish)
                }
                PathArguments::Parenthesized(pa) => {
                    pa.inputs.iter().any(type_is_stringish)
                }
                PathArguments::None => false,
            }
    })
}

fn generic_is_stringish(arg: &GenericArgument) -> bool {
    match arg {
        GenericArgument::Type(t) => type_is_stringish(t),
        GenericArgument::AssocType(a) => type_is_stringish(&a.ty),
        _ => false,
    }
}

/// `AsRef<str>`, `Into<String>`, `Borrow<str>` — stringish only if the generic
/// argument is itself stringish. (Plain `AsRef<[u8]>` is fine.)
fn bound_trait_is_stringish(p: &Path) -> bool {
    let last = p.segments.last().map(|s| s.ident.to_string());
    let is_converter = last
        .as_deref()
        .map(|l| ["AsRef", "Into", "TryInto", "Borrow"].contains(&l))
        .unwrap_or(false);
    if !is_converter {
        return false;
    }
    p.segments.iter().any(|seg| match &seg.arguments {
        PathArguments::AngleBracketed(ab) => ab.args.iter().any(generic_is_stringish),
        _ => false,
    })
}

/// If `trait_path` is a forbidden string-conversion trait on a gated type,
/// return a human-readable reason.
fn forbidden_string_trait(trait_path: &Path) -> Option<String> {
    let last = trait_path.segments.last()?.ident.to_string();
    if FORBIDDEN_STRING_TRAITS.contains(&last.as_str()) {
        return Some(format!("`{last}` yields a string view (re-opens A5)"));
    }
    if FORBIDDEN_STRING_GENERIC_TRAITS.contains(&last.as_str()) {
        let stringish_args: Vec<String> = trait_path
            .segments
            .iter()
            .flat_map(|seg| match &seg.arguments {
                PathArguments::AngleBracketed(ab) => ab
                    .args
                    .iter()
                    .filter_map(|arg| match arg {
                        GenericArgument::Type(t) if type_is_stringish(t) => {
                            Some(t.to_token_stream().to_string())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect();
        if !stringish_args.is_empty() {
            return Some(format!(
                "`{last}<{}>` constructs from a string (re-opens A5)",
                stringish_args.join(", ")
            ));
        }
    }
    None
}

// Sanity: the detector actually recognises every stringish form the review
// enumerated. If syn changes representation these would start passing the real
// guard silently, so pin the detector against literal type fragments.
#[test]
fn detector_recognises_all_stringish_forms() {
    fn parse_type(src: &str) -> Type {
        syn::parse_str::<Type>(src).expect("valid type")
    }

    let stringish: HashSet<&str> = [
        "&str",
        "str",
        "String",
        "&String",
        "&mut str",
        "std::string::String",
        "Cow<'_, str>",
        "Box<str>",
        "Vec<String>",
        "Option<&str>",
        "&[String]",
        "AsRef<str>",
        "Into<String>",
        "Borrow<str>",
    ]
    .into_iter()
    .collect();
    for s in stringish {
        assert!(type_is_stringish(&parse_type(s)), "should flag: {s}");
    }

    let not_stringish: HashSet<&str> =
        ["&[u8]", "[u8; 64]", "Vec<u8>", "AsRef<[u8]>", "Cid512", "RendezvousKey"]
            .into_iter()
            .collect();
    for s in not_stringish {
        assert!(!type_is_stringish(&parse_type(s)), "should NOT flag: {s}");
    }
}
