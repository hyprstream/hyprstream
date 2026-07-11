//! Structural enforcement of review rule R1 (story C4 / #892):
//! **no public locator API may take a string-name parameter**, and `Cid512`
//! must not gain a string-conversion trait impl. A name path inside the DHT
//! client re-opens threat A5 (the DHT becomes a trusted name authority).
//!
//! # Positive allowlist, not whack-a-mole
//!
//! Earlier iterations enumerated *forbidden* string forms (`&str`, `String`,
//! `impl AsRef<str>`, `<S: AsRef<str>>`, type aliases, …). Every review round
//! found another idiomatic name path — that is whack-a-mole. This detector is
//! instead **closed-world / positive**: it enumerates the *approved* public
//! parameter and field types (the CID-keyed newtypes plus primitives and
//! byte views) and rejects **everything else**. A name path therefore cannot
//! sneak in in any new form — struct-wrapped (`pub fn locate(q: NameQuery)`),
//! generic (`<S: ToString>`, `impl Display`), externally aliased
//! (`pub fn resolve(n: CapsuleName)`), or otherwise — because the offending
//! type is simply not on the vetted list. Introducing a new public parameter
//! type is a deliberate, reviewable edit of [`APPROVED_LEAF_TYPES`] /
//! [`APPROVED_WRAPPERS`], which is exactly the "un-vetted public param structs
//! are forbidden" gate the re-review asked for.
//!
//! # What is checked
//!
//! - every public function/method/trait-method **parameter** type must be
//!   approved (recursing through `&`/`Box`/`Option`/`Vec`/slices/arrays);
//! - every **public field** of a public struct/enum must be an approved type
//!   (catches `pub struct NameQuery { pub name: String }` at the definition);
//! - every **type alias** RHS must be approved (catches `type X = String`);
//! - **macros and `pub use`** are forbidden outright in this module (a macro
//!   could generate, or a re-export surface, a name-taking `pub fn` the walk
//!   cannot see);
//! - `Cid512` et al. may not gain a string-conversion trait impl (`FromStr`,
//!   `AsRef`, `Deref`, `Borrow`, `From<&str>`, `TryFrom<&str>`).
//!
//! # Known limitation / follow-up
//!
//! This is a *test*-time check (AST walk), not a *compile* error. The ideal —
//! a build-script or dylint/proc-macro that turns a string-taking `pub fn` into
//! a hard `compile_error!` — is tracked in a follow-up issue; the positive
//! allowlist here is the closed-world interim guard. The only admissible name
//! source for the locator is an out-of-band trusted authority, never this module.
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
use syn::{FnArg, GenericArgument, ImplItem, Item, Path, PathArguments, Type};

/// Public types exported by `locator.rs` whose trait impls are gated by this
/// test (a string-conversion impl on any of them re-opens A5).
const GATED_SELF_TYPES: &[&str] = &["Cid512", "RendezvousKey", "PeerContact", "LookupHints"];

const LOCATOR_SRC: &str = include_str!("../src/locator.rs");

/// Leaf (non-generic) types admissible in a public locator signature. These are
/// the CID-keyed newtypes plus primitives and the one socket-address view. Any
/// other leaf type — `String`, `str`, `NameQuery`, a type-param ident like `S`,
/// an externally-aliased name type — is rejected.
const APPROVED_LEAF_TYPES: &[&str] = &[
    // CID-keyed newtypes (no public string constructor — R1).
    "Cid512",
    "RendezvousKey",
    "PeerContact",
    "LookupHints",
    "SocketAddrV4",
    // primitives
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "usize",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "isize",
    "f32",
    "f64",
    "bool",
    "char",
];

/// Generic wrappers admissible only when every type argument is itself approved
/// (`Option<u16>`, `Vec<PeerContact>`, `&[u8]` via slice, etc.). Anything else
/// generic — `Cow<'_, str>`, `Vec<String>`, `Box<str>` — fails because the
/// inner type is not approved.
const APPROVED_WRAPPERS: &[&str] = &["Option", "Vec", "Box", "Cow"];

/// Trait names whose implementation on a gated self type is forbidden because
/// it yields a string view or string construction.
const FORBIDDEN_STRING_TRAITS: &[&str] = &["FromStr", "AsRef", "Deref", "Borrow"];
/// Trait names that are only forbidden when their generic argument is a string
/// (e.g. `From<&str>`, but `From<[u8; 64]>` is fine).
const FORBIDDEN_STRING_GENERIC_TRAITS: &[&str] = &["From", "TryFrom"];

#[derive(Debug, Default)]
struct Violations {
    unapproved: Vec<String>,
    string_trait_impl: Vec<String>,
}

impl Violations {
    fn is_empty(&self) -> bool {
        self.unapproved.is_empty() && self.string_trait_impl.is_empty()
    }

    fn fail(self) {
        let mut msgs = Vec::new();
        if !self.unapproved.is_empty() {
            msgs.push(format!(
                "R1 violation — public locator surface with an un-vetted (non-allowlisted) \
                 type; only {{Cid512, RendezvousKey, PeerContact, LookupHints, SocketAddrV4, \
                 primitives, byte views}} may cross the public API (re-opens A5):\n{}",
                self.unapproved.join("\n")
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
            // Public struct/enum fields must be approved types — catches a
            // `pub struct NameQuery { pub name: String }` at the definition.
            Item::Struct(s) if is_pub(&s.vis) => {
                // Struct fields carry their own visibility; only `pub` ones cross.
                check_fields(&s.fields, &s.ident, false, v);
            }
            Item::Enum(e) if is_pub(&e.vis) => {
                // Variant fields of a public enum are public regardless of their
                // syntactic visibility (it is always inherited/empty), so force
                // the check — otherwise `pub enum E { Name(String) }` slips
                // through because the tuple field reads as non-`pub`.
                for variant in &e.variants {
                    check_fields(&variant.fields, &e.ident, true, v);
                }
            }
            // `type X = String;` — an unapproved alias used as a pub fn param
            // type would not be caught by name. Deny it at any visibility.
            Item::Type(t) => {
                if !type_is_approved(&t.ty) {
                    v.unapproved.push(format!(
                        "  type alias `{}` = {} (un-vetted type; re-opens A5)",
                        t.ident,
                        t.ty.to_token_stream()
                    ));
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
            // Macros and `pub use` are forbidden outright: a macro could
            // generate, or a re-export surface, a name-taking `pub fn` this AST
            // walk would never see. Fail by name so a human must approve.
            Item::Macro(m) => {
                let label = m
                    .ident
                    .as_ref()
                    .map(std::string::ToString::to_string)
                    .unwrap_or_else(|| m.mac.path.to_token_stream().to_string());
                v.unapproved.push(format!(
                    "  macro `{label}` — macros are forbidden in this module; they could generate an un-walked name-taking pub fn"
                ));
            }
            Item::Use(u) if is_pub(&u.vis) => {
                v.unapproved.push(format!(
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
            // Check only inherent `pub fn` methods. Trait-impl method
            // signatures are *mandated* by their trait (e.g. `Debug::fmt` takes
            // a `Formatter`), so their param types are not the locator's own
            // API; the name-path risk of a trait impl is captured at the impl
            // level (`forbidden_string_trait` forbids string-conversion traits
            // on gated types) and at the trait-definition level (pub traits'
            // method sigs are checked in `walk_items`).
            if imp.trait_.is_none() && is_pub(&m.vis) {
                check_sig(&m.sig, "method", &m.sig.ident, v);
            }
        }
    }
}

fn check_sig(sig: &syn::Signature, kind: &str, name: &syn::Ident, v: &mut Violations) {
    for arg in &sig.inputs {
        if let FnArg::Typed(pat_type) = arg {
            if !type_is_approved(&pat_type.ty) {
                v.unapproved.push(format!(
                    "  {kind} `{name}` param `{}` has un-vetted type {} (re-opens A5)",
                    pat_type.pat.to_token_stream(),
                    pat_type.ty.to_token_stream()
                ));
            }
        }
    }
}

fn check_fields(fields: &syn::Fields, owner: &syn::Ident, force_public: bool, v: &mut Violations) {
    for f in fields {
        if (force_public || is_pub(&f.vis)) && !type_is_approved(&f.ty) {
            v.unapproved.push(format!(
                "  public field on `{owner}` has un-vetted type {} (re-opens A5)",
                f.ty.to_token_stream()
            ));
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
        Type::Path(tp) => tp.path.segments.last().map(|s| s.ident.to_string()),
        Type::Reference(r) => type_name(&r.elem),
        Type::Paren(p) => type_name(&p.elem),
        _ => None,
    }
}

/// **The closed-world R1 predicate.** Is `ty` one of the vetted public locator
/// types? Recurses through references, boxes, slices, arrays, tuples, and
/// generic wrappers so `&Cid512`, `Option<u16>`, `Vec<PeerContact>`, `&[u8]`,
/// `[u8; 64]` are approved, while `&str`, `String`, `NameQuery`, a generic type
/// param `S`, `impl ToString`, `CapsuleName`, `Cow<'_, str>`, `Vec<String>` are
/// not. Generics (`impl Trait`, `dyn Trait`, bare type params) fall through to
/// `_ => false` and are therefore rejected — so no string-ish bound enumeration
/// is needed.
fn type_is_approved(ty: &Type) -> bool {
    match ty {
        Type::Reference(r) => type_is_approved(&r.elem),
        Type::Paren(p) => type_is_approved(&p.elem),
        Type::Group(g) => type_is_approved(&g.elem),
        Type::Slice(s) => type_is_approved(&s.elem),
        Type::Array(a) => type_is_approved(&a.elem),
        Type::Ptr(p) => type_is_approved(&p.elem),
        // Unit `()` is approved; a tuple is approved iff every element is.
        Type::Tuple(t) => t.elems.is_empty() || t.elems.iter().all(type_is_approved),
        // A qualified path (`<Evil as Trait>::Cid512`) is never approved: its
        // resolved type is not one of the local vetted newtypes.
        Type::Path(tp) => tp.qself.is_none() && path_is_approved(&tp.path),
        // ImplTrait, TraitObject, Infer, Macro, Never, Verbatim, bare type
        // params-as-types → not approved (rejected by default).
        _ => false,
    }
}

fn path_is_approved(p: &Path) -> bool {
    // Only a bare, single-segment path can name a local vetted type. A leading
    // `::` or any qualifier (`evil::Cid512`) means the type is *not* the local
    // newtype the allowlist vets — reject it so the closed-world guarantee holds.
    if p.leading_colon.is_some() || p.segments.len() != 1 {
        return false;
    }
    let Some(last) = p.segments.last() else {
        return false;
    };
    let name = last.ident.to_string();
    if APPROVED_LEAF_TYPES.contains(&name.as_str()) {
        return true;
    }
    // A wrapper (Option/Vec/Box/Cow) is approved iff every type argument is.
    if APPROVED_WRAPPERS.contains(&name.as_str()) {
        return match &last.arguments {
            PathArguments::AngleBracketed(ab) => ab.args.iter().all(|arg| match arg {
                GenericArgument::Type(t) => type_is_approved(t),
                GenericArgument::AssocType(a) => type_is_approved(&a.ty),
                // Lifetimes/const args are irrelevant to the name-path check.
                _ => true,
            }),
            _ => false,
        };
    }
    // Any other path — String, str (as a type), NameQuery, CapsuleName, a type
    // param ident, … — is rejected.
    false
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
                        GenericArgument::Type(t) if !type_is_approved(t) => {
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

// ----- self-tests: pin the closed-world allowlist --------------------------------

#[test]
fn allowlist_approves_only_vetted_types() {
    fn parse_type(src: &str) -> Type {
        syn::parse_str::<Type>(src).expect("valid type")
    }

    let approved: HashSet<&str> = [
        "Cid512",
        "RendezvousKey",
        "PeerContact",
        "LookupHints",
        "SocketAddrV4",
        "&Cid512",
        "RendezvousKey",
        "&LookupHints",
        "Option<u16>",
        "Vec<PeerContact>",
        "Box<PeerContact>",
        "&[u8]",
        "[u8; 64]",
        "[u8; 20]",
        "u32",
        "u64",
        "usize",
        "bool",
        "()",
    ]
    .into_iter()
    .collect();
    for s in approved {
        assert!(type_is_approved(&parse_type(s)), "should APPROVE: {s}");
    }

    // Every form the three review rounds used as a bypass must be REJECTED.
    let rejected: HashSet<&str> = [
        // round 1: concrete strings
        "&str",
        "str",
        "String",
        "&String",
        "Cow<'_, str>",
        "Box<str>",
        "Vec<String>",
        // round 2: generic / alias
        "impl AsRef<str>",
        "Into<String>",
        "AsRef<str>",
        // round 3: struct-wrapped, ToString/Display, external alias, generic param
        "NameQuery",
        "CapsuleName",
        "impl ToString",
        "impl Display",
        "S", // a bare generic type param
        "Option<String>",
        "Box<NameQuery>",
        // qualified / associated paths: a foreign `Cid512` (or a `<_ as _>`
        // projection) is NOT the local vetted newtype and must be rejected.
        "evil::Cid512",
        "::Cid512",
        "evil::Option<PeerContact>",
        "<Evil as Trait>::Cid512",
    ]
    .into_iter()
    .collect();
    for s in rejected {
        assert!(!type_is_approved(&parse_type(s)), "should REJECT: {s}");
    }
}

// Drives `check_sig` against the round-3 bypass params directly. The closed
// allowlist rejects them all by *omission* — no per-form enumeration needed.
#[test]
fn check_sig_rejects_unapproved_param_types() {
    fn parse_sig(src: &str) -> syn::Signature {
        syn::parse_str::<syn::ItemFn>(src).expect("valid fn").sig
    }
    fn flagged(src: &str) -> bool {
        let sig = parse_sig(src);
        let mut v = Violations::default();
        check_sig(&sig, "fn", &sig.ident, &mut v);
        !v.is_empty()
    }

    let should_reject = [
        // #1 struct-wrapped string field as a param
        "fn locate(q: NameQuery) {}",
        // #2 impl ToString / impl Display, and their <S: ..> desugarings
        "fn f(s: impl ToString) {}",
        "fn f(s: impl Display) {}",
        "fn f<S: ToString>(s: S) {}",
        "fn f<S: Display>(s: S) {}",
        // #3 private external alias used as a param
        "fn resolve(name: CapsuleName) {}",
        // concrete strings, multi-line, etc.
        "fn f(name: &str) {}",
        "fn f(\n    name: &str,\n) {}",
    ];
    for src in should_reject {
        assert!(flagged(src), "check_sig should REJECT: {src}");
    }

    // The actual locator signatures must stay clean.
    let clean = [
        "fn f(cid: &Cid512) {}",
        "fn f(cid: &Cid512, key: RendezvousKey) {}",
        "fn f(cid: &Cid512, key: RendezvousKey, hints: &LookupHints) {}",
        "fn f(&self, cid: &Cid512, port: Option<u16>) {}",
        "fn f(info_hash: [u8; 20], port: Option<u16>) {}",
    ];
    for src in clean {
        assert!(!flagged(src), "check_sig should NOT reject: {src}");
    }
}

// A public struct with a string field is caught at the definition (hole #1
// defense-in-depth — it is also caught if used as a param, above).
#[test]
fn walk_flags_public_string_struct_field() {
    fn parse_items(src: &str) -> Vec<Item> {
        syn::parse_file(src).expect("valid module").items
    }

    let mut v = Violations::default();
    walk_items(
        &parse_items("pub struct NameQuery { pub name: String }\npub struct Clean { a: u32 }"),
        &mut v,
    );
    assert!(
        v.unapproved
            .iter()
            .any(|m| m.contains("NameQuery") && m.contains("String")),
        "public String field must be flagged: {:?}",
        v.unapproved
    );
    // Private fields and approved public fields are not flagged.
    assert!(
        v.unapproved.iter().all(|m| !m.contains("Clean")),
        "{:?}",
        v.unapproved
    );
}

// A public enum with a string-carrying variant field is caught. Variant fields
// have inherited (non-`pub`) visibility yet are public in a public enum, so the
// walk must force the field check — otherwise `pub enum E { Name(String) }`
// silently bypasses R1.
#[test]
fn walk_flags_public_enum_variant_string_field() {
    fn parse_items(src: &str) -> Vec<Item> {
        syn::parse_file(src).expect("valid module").items
    }

    let mut v = Violations::default();
    walk_items(
        &parse_items(
            "pub enum E { Name(String), Named { name: String }, Clean(u32) }\n\
             pub enum Ok2 { A(Cid512), B(RendezvousKey) }",
        ),
        &mut v,
    );
    assert!(
        v.unapproved
            .iter()
            .filter(|m| m.contains("String"))
            .count()
            >= 2,
        "both tuple and struct-style enum variant String fields must be flagged: {:?}",
        v.unapproved
    );
    assert!(
        v.unapproved.iter().all(|m| !m.contains("u32")),
        "approved variant fields must not be flagged: {:?}",
        v.unapproved
    );
}

// A public type alias whose RHS is unapproved must be denied.
#[test]
fn walk_flags_unapproved_type_alias() {
    fn parse_items(src: &str) -> Vec<Item> {
        syn::parse_file(src).expect("valid module").items
    }

    let mut v = Violations::default();
    walk_items(
        &parse_items("pub type CapsuleName = String; type Peer = PeerContact;"),
        &mut v,
    );
    assert!(
        v.unapproved
            .iter()
            .any(|m| m.contains("CapsuleName") && m.contains("un-vetted")),
        "string type alias must be flagged: {:?}",
        v.unapproved
    );
    assert!(
        v.unapproved.iter().all(|m| !m.contains("Peer =")),
        "{:?}",
        v.unapproved
    );
}
