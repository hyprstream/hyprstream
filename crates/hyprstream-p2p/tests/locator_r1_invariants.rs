//! Structural enforcement of review rule R1 (story C4 / #892):
//! **no public locator API may take a string-name parameter**, and `Cid512`
//! must not gain a string-conversion trait impl. A name path inside the DHT
//! client re-opens threat A5 (the DHT becomes a trusted name authority).
//!
//! Why parse the AST instead of grepping or using `trybuild`:
//! - A line-by-line grep is bypassable with a multi-line signature,
//!   `impl AsRef<str>`, `Cow<'_, str>`, `&String`, `Box<str>`, `Vec<String>`,
//!   a generic bound (`<S: AsRef<str>>` / `where S: AsRef<str>`), a type alias,
//!   a macro-generated `pub fn`, a re-export, or a trait method.
//! - `trybuild` compile-fail tests can only pin *specific* forbidden call
//!   sites; they cannot prove that *no* public method, present or future, takes
//!   a string.
//!
//! What this walk actually covers (kept honest against the #1000 re-review):
//! - concrete parameter types — recurses through `&`/`Box`/`Cow`/slices/tuples/
//!   generics/`impl Trait`/`dyn Trait` so every `&str`/`String`/`Box<str>`/
//!   `Vec<String>`/`impl AsRef<str>` form is flagged;
//! - **generic params and where-clauses** — `<S: AsRef<str>>`, `<T: Into<String>>`,
//!   `<S: FromStr>`, and `where T: AsRef<str>` are flagged via the same bound
//!   predicate, so the `<T: Trait>` desugaring of `impl Trait` cannot smuggle a
//!   name type in;
//! - **type aliases** — `pub type X = String;` (any visibility) is denied, since
//!   a `String` alias used as a `pub fn` param type would not otherwise look
//!   string-like;
//! - **macros and `pub use`** — these are *forbidden outright* in this module
//!   (not expanded/resolved), because a macro could generate, or a re-export
//!   surface, a name-taking `pub fn` the AST walk would never see. Any such item
//!   fails the test by name so a human must approve it explicitly;
//! - forbidden string-conversion trait impls on `Cid512` et al.
//!
//! Known limitation: a type alias defined *outside* this module (`use`d in)
//! whose target is a string is not resolved here — but `pub use` of the alias
//! is forbidden (above), and the only admissible name source is an out-of-band
//! trusted authority, never this locator.
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
use syn::{
    FnArg, GenericArgument, GenericParam, ImplItem, Item, Path, PathArguments, Type,
    TypeParamBound, WherePredicate,
};

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
    let file = syn::parse_file(LOCATOR_SRC)
        .expect("src/locator.rs must parse as valid Rust for the R1 invariant test");

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
            // `type X = String;` — a string alias used as a pub fn param type
            // would not otherwise look string-like. Deny it at any visibility.
            Item::Type(t) => {
                if type_is_stringish(&t.ty) {
                    v.string_param.push(format!(
                        "  type alias `{}` = {} (string alias smuggles a name path; re-opens A5)",
                        t.ident,
                        t.ty.to_token_stream()
                    ));
                }
            }
            // Macros and `pub use` are forbidden outright: a macro could
            // generate, or a re-export surface, a name-taking `pub fn` this AST
            // walk would never see. Fail by name so a human must approve.
            Item::Macro(m) => {
                let label = m
                    .ident
                    .as_ref()
                    .map(std::string::ToString::to_string)
                    .unwrap_or_else(|| m.mac.path.to_token_stream().to_string());
                v.string_param.push(format!(
                    "  macro `{label}` — macros are forbidden in this module; they could generate an un-walked name-taking pub fn"
                ));
            }
            Item::Use(u) if is_pub(&u.vis) => {
                v.string_param.push(format!(
                    "  {} — `pub use` is forbidden in this module; a re-export could surface a name-taking fn the AST walk cannot see",
                    u.to_token_stream()
                ));
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
                    v.string_trait_impl.push(format!(
                        "  impl {} for {self_name}: {reason}",
                        trait_path.to_token_stream()
                    ));
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
    // Concrete parameter types.
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

    // Generic params + where-clause. `<S: AsRef<str>>(name: S)` and
    // `(s: S) where S: AsRef<str>` are the desugaring of `impl AsRef<str>` and
    // would otherwise pass because the concrete param type `S` is a plain
    // ident. Walk every generic bound and where-predicate with the same
    // stringish predicate used for `impl Trait`.
    if generics_admit_stringish(&sig.generics) {
        v.string_param.push(format!(
            "  {kind} `{name}` generic/where bound admits a string-like type (re-opens A5): {}",
            sig.generics.to_token_stream()
        ));
    }
}

/// Does a generics list admit a string-like type via a bound?
///
/// Catches both `<S: AsRef<str>>` (bounds on type params) and
/// `where S: AsRef<str>` (where-clause predicates), reusing the same
/// [`bound_trait_is_stringish`] predicate as `impl Trait` so neither desugaring
/// can smuggle a name type past the concrete-param check.
fn generics_admit_stringish(generics: &syn::Generics) -> bool {
    let param_bound = generics.params.iter().any(|p| match p {
        GenericParam::Type(tp) => bounds_stringish(&tp.bounds),
        // Lifetime/const params carry no stringish trait bound.
        _ => false,
    });
    if param_bound {
        return true;
    }
    generics
        .where_clause
        .iter()
        .flat_map(|wc| &wc.predicates)
        .any(|pred| match pred {
            WherePredicate::Type(pt) => bounds_stringish(&pt.bounds),
            // Lifetime/`Eq` predicates: no stringish trait bound.
            _ => false,
        })
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
        Type::Path(tp) => tp.path.segments.last().map(|s| s.ident.to_string()),
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

fn bounds_stringish(
    bounds: &syn::punctuated::Punctuated<TypeParamBound, syn::token::Plus>,
) -> bool {
    bounds
        .iter()
        .any(|b| matches!(b, TypeParamBound::Trait(tb) if bound_trait_is_stringish(&tb.path)))
}

fn path_is_stringish(p: &Path) -> bool {
    p.segments.iter().any(|seg| {
        let id = seg.ident.to_string();
        STRINGISH_IDENTS.contains(&id.as_str())
            || match &seg.arguments {
                PathArguments::AngleBracketed(ab) => ab.args.iter().any(generic_is_stringish),
                PathArguments::Parenthesized(pa) => pa.inputs.iter().any(type_is_stringish),
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

/// Is a trait *bound* stringish? Used for `impl Trait` params, `dyn Trait`
/// trait objects, generic-param bounds (`<S: Trait>`), and where-clauses.
///
/// - `FromStr` names string construction with no generic argument.
/// - Converter traits (`AsRef<..>`, `Into<..>`, `TryInto<..>`, `Borrow<..>`)
///   are stringish only when their generic argument is itself stringish, so
///   `AsRef<str>`/`Into<String>`/`Borrow<str>` flag but `AsRef<[u8]>` /
///   `Borrow<[u8]>` (byte views) do not.
fn bound_trait_is_stringish(p: &Path) -> bool {
    let Some(last) = p.segments.last() else {
        return false;
    };
    let name = last.ident.to_string();
    if name == "FromStr" {
        return true;
    }
    ["AsRef", "Into", "TryInto", "Borrow"].contains(&name.as_str())
        && p.segments.iter().any(|seg| match &seg.arguments {
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

    let not_stringish: HashSet<&str> = [
        "&[u8]",
        "[u8; 64]",
        "Vec<u8>",
        "AsRef<[u8]>",
        "Borrow<[u8]>",
        "Cid512",
        "RendezvousKey",
    ]
    .into_iter()
    .collect();
    for s in not_stringish {
        assert!(!type_is_stringish(&parse_type(s)), "should NOT flag: {s}");
    }
}

// Drive `check_sig` directly against signatures whose stringish-ness lives in
// `generics` / the where-clause, not in a concrete param type. This path was
// previously UNwalked (the original detector only read `sig.inputs`), so this
// is the load-bearing regression guard for the #1000 re-review's primary hole.
#[test]
fn check_sig_flags_generic_and_where_clause_string_bounds() {
    fn parse_sig(src: &str) -> syn::Signature {
        syn::parse_str::<syn::ItemFn>(src).expect("valid fn").sig
    }

    fn flagged(src: &str) -> bool {
        let sig = parse_sig(src);
        let mut v = Violations::default();
        check_sig(&sig, "fn", &sig.ident, &mut v);
        !v.is_empty()
    }

    // The exact desugarings from the re-review — must all be caught.
    let should_flag = [
        "fn f<S: AsRef<str>>(s: S) {}",
        "fn resolve<S: AsRef<str>>(name: S) -> () {}",
        "fn f<T>(s: T) where T: AsRef<str> {}",
        "fn f<T: Into<String>>(s: T) {}",
        "fn f<T>(s: T) where T: Borrow<str> {}",
        "fn f<S: FromStr>(s: S) {}",
        // Multi-line generic bound — the original grep could not see this either.
        "fn f<S>(\n    s: S,\n) where\n    S: AsRef<str>,\n{}",
    ];
    for src in should_flag {
        assert!(
            flagged(src),
            "check_sig should flag generic/where string bound: {src}"
        );
    }

    // Negative: non-string bounds must NOT be flagged.
    let clean = [
        "fn f<T: Clone + Send>(s: T) where T: 'static {}",
        "fn f<T: AsRef<[u8]>>(s: T) {}",
        "fn f(cid: &Cid512, key: RendezvousKey) {}",
    ];
    for src in clean {
        assert!(!flagged(src), "check_sig should NOT flag: {src}");
    }
}

// A public type alias whose RHS is a string must be denied — otherwise
// `pub type CapsuleName = String;` used as a `pub fn` param type would not
// look string-like to the concrete-param check.
#[test]
fn walk_flags_string_type_alias_and_clean_alias_passes() {
    fn parse_items(src: &str) -> Vec<Item> {
        syn::parse_file(src).expect("valid module").items
    }

    let mut v = Violations::default();
    walk_items(
        &parse_items("pub type CapsuleName = String; type Peer = PeerContact;"),
        &mut v,
    );
    assert!(
        v.string_param
            .iter()
            .any(|m| m.contains("CapsuleName") && m.contains("string alias")),
        "string type alias must be flagged: {:?}",
        v.string_param
    );
    // Non-string alias is not flagged.
    assert!(
        v.string_param.iter().all(|m| !m.contains("Peer =")),
        "{:?}",
        v.string_param
    );
}
