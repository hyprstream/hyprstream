//! Full method-leaf tree computation and the generated single-decode body
//! decoder (v16 §5.1/§5.2).
//!
//! One shared tree walk ([`collect_method_leaves`]) drives both:
//!
//! - [`generate_body_decoder`] — the generated
//!   `decode_<service>_request_body` function performing the ONE bounded
//!   Cap'n Proto decode and deriving the full nested union-discriminant leaf
//!   path from the signed body; and
//! - the generated method-policy inventory rows (one row per leaf).
//!
//! Because both are produced from the same walk, the decoder can never derive
//! a leaf the inventory does not list, and the inventory can never list a
//! leaf the decoder cannot derive — the agreement is by construction, not by
//! review.
//!
//! The walk descends exactly the unions that select handlers:
//!
//! - the root request union (always);
//! - scoped-client subtrees (`ScopedClient`, recursively through
//!   `nested_clients`) — these are the nested method unions generated scope
//!   dispatch routes on; and
//! - local pure-union struct payloads that are not `Option<T>` wrappers —
//!   these are inner method selectors dispatched from raw bytes.
//!
//! Everything else — struct parameters (even union-bearing ones), imported
//! types, groups — is method *payload*, not method identity, and terminates
//! the leaf.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use hyprstream_rpc_build::schema::dispatch_label::{
    parse_dispatch_mac, parse_dispatch_public_reason, InitialLabelMap, READ_CLASS_ACTIONS,
};

use crate::resolve::ResolvedSchema;
use crate::schema::types::*;
use crate::util::*;

/// One method leaf discovered by the shared tree walk.
pub struct MethodLeaf {
    /// Full union-discriminant chain from the root request union.
    pub path: Vec<u16>,
    /// Dotted human-readable path (review metadata, e.g. `repo.create`).
    pub symbolic: String,
    /// `$scope`/`$capability` action of the leaf variant. Empty when the leaf
    /// is scope-exempt or sits below a hand-dispatched pure union whose
    /// arms carry no annotations.
    pub scope: String,
    /// Whether the leaf variant is `$scopeExempt` (control-plane exemption).
    pub scope_exempt: bool,
    /// `$dispatchMac` label text (strict grammar, v16 §6). Empty when the leaf
    /// is `$dispatchPublic` or the annotation is inherited from a selector.
    pub dispatch_mac: String,
    /// `$dispatchPublic` reason text. Empty for `$dispatchMac` leaves.
    pub dispatch_public: String,
}

/// Per-arm metadata resolved during the walk.
struct ArmMeta<'a> {
    name: &'a str,
    discriminant: u16,
    type_name: &'a str,
    scope: &'a str,
    scope_exempt: bool,
    dispatch_mac: &'a str,
    dispatch_public: &'a str,
}

/// Annotation state inherited from the nearest annotated ancestor selector.
/// Only hand-dispatched pure-union selectors propagate annotations downward
/// (scoped-client dispatchers carry none by rule); arms below them may carry
/// their own. `$dispatchPublic` is **never** inherited — it is legal only on
/// leaves — so only the MAC label flows through here and an inherited public
/// reason is surfaced as the build error it is (see `walk_level`).
#[derive(Clone, Default)]
struct Inherited {
    scope: String,
    scope_exempt: bool,
    dispatch_mac: String,
    dispatch_public: String,
}

/// The arms of one union level, in schema declaration order.
///
/// Root level: `request_variants` + discriminants from `request_struct`
/// (index fallback for text-parsed schemas without wire info, matching the
/// existing dispatch discriminator behavior). Scope level: the scope struct's
/// union fields, with metadata joined from `inner_request_variants`.
fn level_arms<'a>(
    sdef: Option<&'a StructDef>,
    variants: &'a [UnionVariant],
) -> Vec<ArmMeta<'a>> {
    match sdef {
        Some(sdef) => sdef
            .union_fields()
            .map(|f| {
                let meta = variants.iter().find(|v| v.name == f.name);
                ArmMeta {
                    name: &f.name,
                    discriminant: f.discriminant_value,
                    type_name: &f.type_name,
                    scope: meta.map(|v| v.scope.as_str()).unwrap_or(""),
                    scope_exempt: meta.map(|v| v.scope_exempt).unwrap_or(false),
                    dispatch_mac: meta.map(|v| v.dispatch_mac.as_str()).unwrap_or(""),
                    dispatch_public: meta.map(|v| v.dispatch_public.as_str()).unwrap_or(""),
                }
            })
            .collect(),
        None => variants
            .iter()
            .enumerate()
            .map(|(index, v)| ArmMeta {
                name: &v.name,
                discriminant: u16::try_from(index).unwrap_or(u16::MAX),
                type_name: &v.type_name,
                scope: &v.scope,
                scope_exempt: v.scope_exempt,
                dispatch_mac: &v.dispatch_mac,
                dispatch_public: &v.dispatch_public,
            })
            .collect(),
    }
}

/// Whether a union arm's payload is itself a method-selecting union to
/// descend into (a local, non-`Option` pure-union struct).
fn descend_struct<'a>(resolved: &'a ResolvedSchema, type_name: &str) -> Option<&'a StructDef> {
    let sdef = resolved.find_struct(type_name)?;
    if sdef.origin_file.is_some() {
        return None; // imported type: payload, not local method identity
    }
    if sdef.is_pure_union() && sdef.option_inner_type().is_none() {
        Some(sdef)
    } else {
        None
    }
}

/// Recursively collect every method leaf of the service's request tree.
pub fn collect_method_leaves(resolved: &ResolvedSchema) -> Vec<MethodLeaf> {
    let mut out = Vec::new();
    walk_level(
        resolved,
        resolved.raw.request_struct.as_ref(),
        &resolved.raw.request_variants,
        &resolved.raw.scoped_clients,
        &[],
        "",
        &Inherited::default(),
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn walk_level(
    resolved: &ResolvedSchema,
    sdef: Option<&StructDef>,
    variants: &[UnionVariant],
    scopes: &[ScopedClient],
    prefix: &[u16],
    prefix_sym: &str,
    // Annotation inherited from the nearest annotated ancestor selector: the
    // arms below a hand-dispatched pure union carry no `$scope` metadata of
    // their own, so they inherit their selector's annotation instead of
    // producing an unannotated (build-failing) row. `$dispatchMac` inherits
    // the same way; `$dispatchPublic` NEVER does (public is legal only on
    // leaves) — an ancestor public annotation marks `public_ancestor` so the
    // descendant leaf fails the build rather than silently inheriting public.
    inherited: &Inherited,
    out: &mut Vec<MethodLeaf>,
) {
    for arm in level_arms(sdef, variants) {
        let mut path = prefix.to_vec();
        path.push(arm.discriminant);
        let symbolic = if prefix_sym.is_empty() {
            arm.name.to_owned()
        } else {
            format!("{prefix_sym}.{}", arm.name)
        };
        let arm_annotated = !arm.dispatch_mac.is_empty() || !arm.dispatch_public.is_empty();
        let mut effective = inherited.clone();
        if arm.scope.is_empty() && !arm.scope_exempt {
            // scope (and its exemption) inherit from the selector unchanged.
        } else {
            effective.scope = arm.scope.to_owned();
            effective.scope_exempt = arm.scope_exempt;
        }
        if arm_annotated {
            effective.dispatch_mac = arm.dispatch_mac.to_owned();
            effective.dispatch_public = arm.dispatch_public.to_owned();
        } else {
            // An unannotated arm: a `$dispatchMac` selector's label inherits
            // downward (kept above), but `$dispatchPublic` NEVER inherits —
            // the arm is left unannotated and its leaf fails the build (v16
            // §6: public is legal only on leaves, never inherited).
            effective.dispatch_public = String::new();
        }

        if let Some(sc) = scopes.iter().find(|sc| sc.factory_name == arm.name) {
            // A scope selector: descend into its inner method union. Scoped
            // dispatchers carry neither dispatch annotation (v16 §6); the
            // inheritance they propagate is whatever THEY inherited (a scoped
            // selector nested inside a hand-dispatched pure union).
            walk_level(
                resolved,
                resolved.find_struct(arm.type_name),
                &sc.inner_request_variants,
                &sc.nested_clients,
                &path,
                &symbolic,
                &effective,
                out,
            );
        } else if let Some(inner) = descend_struct(resolved, arm.type_name) {
            // A hand-dispatched pure union: its arms are method identity.
            walk_level(resolved, Some(inner), &[], &[], &path, &symbolic, &effective, out);
        } else {
            out.push(MethodLeaf {
                path,
                symbolic,
                scope: effective.scope.clone(),
                scope_exempt: effective.scope_exempt,
                dispatch_mac: effective.dispatch_mac.clone(),
                dispatch_public: effective.dispatch_public.clone(),
            });
        }
    }
}

/// Generate `decode_<service>_request_body`: the ONE bounded decode of the
/// signed request body, deriving the full nested leaf path (v16 §5.2 step 4).
///
/// The generated function performs exactly one `read_message` under the
/// reviewed caps ([`hyprstream_rpc::service::body::bounded_reader_options`]),
/// walks the method unions of the decoded message to the executed leaf, and
/// returns a `DecodedRequestBody` carrying the decoded message itself — the
/// same message generated dispatch later reads the typed request from. An
/// unknown discriminant at any level is an error (deny), never a truncated
/// or coarser path.
pub fn generate_body_decoder(service_name: &str, resolved: &ResolvedSchema) -> TokenStream {
    let pascal = to_pascal_case(service_name);
    let capnp_mod = format_ident!("{}_capnp", service_name);
    let req_snake = format_ident!("{}", to_snake_case(&format!("{pascal}Request")));
    let decode_fn = format_ident!("decode_{}_request_body", to_snake_case(&pascal));
    let root_mod = quote! { crate::#capnp_mod::#req_snake };

    let arms = decoder_arms(
        resolved,
        &capnp_mod,
        &root_mod,
        resolved.raw.request_struct.as_ref(),
        &resolved.raw.request_variants,
        &resolved.raw.scoped_clients,
    );

    let doc = format!(
        "Decode a {pascal} signed request body **exactly once** (bounded) and derive\n\
         the full numeric method leaf path (v16 §5.2).\n\n\
         Returns the one `DecodedRequestBody` that feeds the generated method\n\
         policy, the dispatch MAC PEP, and `dispatch_{}` — there is no second\n\
         `read_message` between admission and handler. An undecodable body, a\n\
         discriminant absent from this schema revision, or a body exceeding the\n\
         reviewed decode caps is an error, which denies — never a coarser path.",
        to_snake_case(&pascal)
    );

    quote! {
        #[doc = #doc]
        pub fn #decode_fn(
            signed_body: &[u8],
        ) -> anyhow::Result<hyprstream_rpc::service::DecodedRequestBody> {
            let message = capnp::serialize::read_message(
                &mut std::io::Cursor::new(signed_body),
                hyprstream_rpc::service::body::bounded_reader_options(),
            )?;
            let mut __path: Vec<u16> = Vec::new();
            {
                let req = message.get_root::<#root_mod::Reader>()?;
                match req.which()? {
                    #(#arms)*
                    #[allow(unreachable_patterns)]
                    _ => anyhow::bail!(
                        "unknown request union discriminant (not in this schema revision)"
                    ),
                }
            }
            hyprstream_rpc::service::DecodedRequestBody::from_message(
                signed_body.to_vec(),
                message,
                __path,
            )
        }
    }
}

/// Generate the service's method-policy inventory rows (v16 §6.1) from the
/// same leaf-tree walk that produced its signed-body decoder.
///
/// Emits `method_policy_rows()` plus an `inventory` submission, so startup
/// installation ([`hyprstream_rpc::proof::policy::install_generated_method_policy`])
/// aggregates every linked service's rows into the one deterministic table.
///
/// Row derivation from the checked schema annotations (v16 §6 — the strict
/// `$dispatchMac`/`$dispatchPublic` pair REPLACES the `$scope`/`$scopeExempt`
/// stand-in as the dispatch-policy source; `$scope` remains the control-plane
/// vocabulary feeding `scope_action`):
///
/// - a `$dispatchPublic("<reason>")` leaf is publicly dispatchable:
///   `UnauthenticatedAllowed` + `UnauthenticatedOrTokenBound` with the
///   mandatory Hybrid suite, target label exactly system low, and the
///   annotation's trimmed reason as its public reason;
/// - a `$dispatchMac("<level>:<assurance>[:<compartments>]")` leaf requires a
///   credential: `CredentialRequired` + `TokenBound` with the mandatory
///   Hybrid suite, and a target label parsed HERE against the checked-in
///   `InitialLabelMap` (stable bit assignments) — never at runtime;
/// - every row carries the fixed system-low `transitional_label` (v16 §7.3)
///   and a `MutationSemantics` derived from the checked `ScopeAction` block
///   structure (`None` for read-class actions, `Some(..)` for mutating ones).
///   (Credentials are Reusable-only per the 2026-08-20 operator deferral;
///   a use-profile field returns only with a future amendment that allocates
///   its claim key — it is not carried dormant here.)
///
/// Any annotation failure — a leaf with neither annotation or both, a public
/// leaf with an empty reason or a coexisting `$scope`, a label that fails the
/// strict grammar (unknown level/assurance, empty components, duplicate or
/// unknown compartments, noncanonical ordering, or system low spelled through
/// `$dispatchMac`) — is a **compile error**: an annotation failure can never
/// produce an unlabeled runtime row.
pub fn generate_method_policy_rows(service_name: &str, resolved: &ResolvedSchema) -> TokenStream {
    let leaves = collect_method_leaves(resolved);

    let label_map = match InitialLabelMap::load() {
        Ok(map) => map,
        Err(e) => {
            let msg = format!("InitialLabelMap failed to load: {e}");
            return quote! { ::core::compile_error!(#msg); };
        }
    };

    let mut row_tokens: Vec<TokenStream> = Vec::new();
    for leaf in &leaves {
        let path = &leaf.path;
        let symbolic = &leaf.symbolic;
        let scope = &leaf.scope;
        let scope_exempt = leaf.scope_exempt;

        let has_mac = !leaf.dispatch_mac.is_empty();
        let has_public = !leaf.dispatch_public.is_empty();
        let (authentication, signature_policy, public_reason, target_label) =
            match (has_mac, has_public) {
                (false, false) => {
                    let msg = format!(
                        "method leaf '{}.{}' has neither a $dispatchMac nor a \
                         $dispatchPublic annotation; every generated dispatch-policy \
                         row must derive from exactly one of the pair (v16 §6). Public \
                         is never inherited from a selector.",
                        service_name, leaf.symbolic
                    );
                    return quote! { ::core::compile_error!(#msg); };
                }
                (true, true) => {
                    let msg = format!(
                        "method leaf '{}.{}' carries BOTH $dispatchMac and $dispatchPublic; \
                         exactly one dispatch annotation per leaf (v16 §6)",
                        service_name, leaf.symbolic
                    );
                    return quote! { ::core::compile_error!(#msg); };
                }
                (false, true) => {
                    let reason = match parse_dispatch_public_reason(&leaf.dispatch_public) {
                        Ok(r) => r,
                        Err(e) => {
                            return quote! { ::core::compile_error!(#e); };
                        }
                    };
                    if !leaf.scope.is_empty() {
                        let msg = format!(
                            "method leaf '{}.{}' is $dispatchPublic but also declares the \
                             $scope({}) action; a control-plane-scoped method cannot be \
                             dispatch-public (v16 §6)",
                            service_name, leaf.symbolic, leaf.scope
                        );
                        return quote! { ::core::compile_error!(#msg); };
                    }
                    (
                        quote! { hyprstream_rpc::proof::policy::AuthenticationRequirement::UnauthenticatedAllowed },
                        quote! {
                            hyprstream_rpc::proof::policy::SignaturePolicy::UnauthenticatedOrTokenBound {
                                suite: hyprstream_rpc::proof::policy::CryptoSuite::Hybrid,
                            }
                        },
                        quote! { Some(#reason) },
                        quote! { hyprstream_rpc::proof::policy::SYSTEM_LOW_LABEL },
                    )
                }
                (true, false) => {
                    let label = match parse_dispatch_mac(&leaf.dispatch_mac, &label_map) {
                        Ok(l) => l,
                        Err(e) => {
                            let msg = format!(
                                "method leaf '{}.{}' $dispatchMac {:?}: {e}",
                                service_name, leaf.symbolic, leaf.dispatch_mac
                            );
                            return quote! { ::core::compile_error!(#msg); };
                        }
                    };
                    let level = level_tokens(label.level);
                    let assurance = assurance_tokens(label.assurance);
                    let bits = &label.compartment_bits;
                    (
                        quote! { hyprstream_rpc::proof::policy::AuthenticationRequirement::CredentialRequired },
                        quote! {
                            hyprstream_rpc::proof::policy::SignaturePolicy::TokenBound {
                                suite: hyprstream_rpc::proof::policy::CryptoSuite::Hybrid,
                            }
                        },
                        quote! { None },
                        quote! {
                            hyprstream_rpc::proof::policy::dispatch_label(#level, #assurance, &[#(#bits),*])
                        },
                    )
                }
            };

        // Mutation semantics derive from the checked ScopeAction block
        // structure (v16 §6.1: "generated from checked schema/policy
        // metadata, not guessed from method names"): Block A (query,
        // subscribe) is side-effect-free; every other action is mutating and
        // requires an explicit semantics — the generated default is
        // NaturallyIdempotent (retry-safe without extra machinery); the
        // idempotency-key and ledger variants arrive with WS-L per method.
        // A public leaf carries no scope action and is exempt from the gate.
        let mutation_semantics = if leaf.scope.is_empty()
            || READ_CLASS_ACTIONS.contains(&leaf.scope.as_str())
        {
            quote! { None }
        } else {
            quote! {
                Some(
                    hyprstream_rpc::proof::policy::MutationSemantics::NaturallyIdempotent
                )
            }
        };

        row_tokens.push(quote! {
            hyprstream_rpc::proof::policy::GeneratedMethodPolicyRow {
                service: #service_name,
                leaf_path: &[#(#path),*],
                symbolic_path: #symbolic,
                scope_action: #scope,
                scope_exempt: #scope_exempt,
                authentication: #authentication,
                signature_policy: #signature_policy,
                mutation_semantics: #mutation_semantics,
                transitional_label: hyprstream_rpc::proof::policy::SYSTEM_LOW_LABEL,
                target_label: #target_label,
                public_reason: #public_reason,
            }
        });
    }

    let doc = format!(
        "Generated dispatch method-policy rows for the {service_name} service (v16 §6.1).\n\n\
         One row per method leaf, derived from the same tree walk as the\n\
         signed-body decoder — the decoder cannot derive a leaf this inventory\n\
         does not list, and vice versa."
    );

    quote! {
        #[doc = #doc]
        pub fn method_policy_rows() -> Vec<hyprstream_rpc::proof::policy::GeneratedMethodPolicyRow> {
            vec![ #(#row_tokens),* ]
        }

        // Aggregated by `install_generated_method_policy()` at startup. Same
        // native-only inventory pattern as `VfsNodeTable` (#1305): linker
        // section registration does not resolve on wasm32.
        #[cfg(not(target_arch = "wasm32"))]
        hyprstream_rpc::metadata::inventory::submit! {
            hyprstream_rpc::proof::policy::GeneratedMethodPolicyProvider {
                service: #service_name,
                rows_fn: method_policy_rows,
            }
        }
    }
}

/// Runtime `Level` enum tokens for one grammar level name.
fn level_tokens(level: &str) -> TokenStream {
    match level {
        "public" => quote! { hyprstream_rpc::auth::mac::Level::Public },
        "internal" => quote! { hyprstream_rpc::auth::mac::Level::Internal },
        "confidential" => quote! { hyprstream_rpc::auth::mac::Level::Confidential },
        "secret" => quote! { hyprstream_rpc::auth::mac::Level::Secret },
        other => unreachable!("grammar parser accepted unknown level {other:?}"),
    }
}

/// Runtime `Assurance` enum tokens for one grammar assurance name.
fn assurance_tokens(assurance: &str) -> TokenStream {
    match assurance {
        "unverified" => quote! { hyprstream_rpc::auth::mac::Assurance::Unverified },
        "classical" => quote! { hyprstream_rpc::auth::mac::Assurance::Classical },
        "pq-hybrid" => quote! { hyprstream_rpc::auth::mac::Assurance::PqHybrid },
        other => unreachable!("grammar parser accepted unknown assurance {other:?}"),
    }
}

/// Build the match arms for one union level of the generated decoder.
fn decoder_arms(
    resolved: &ResolvedSchema,
    capnp_mod: &syn::Ident,
    mod_path: &TokenStream,
    sdef: Option<&StructDef>,
    variants: &[UnionVariant],
    scopes: &[ScopedClient],
) -> Vec<TokenStream> {
    level_arms(sdef, variants)
        .into_iter()
        .map(|arm| {
            let pascal = resolved.name(arm.name).pascal_ident.clone();
            let disc = arm.discriminant;

            let descend_target = if let Some(sc) =
                scopes.iter().find(|sc| sc.factory_name == arm.name)
            {
                resolved.find_struct(arm.type_name).map(|inner_sdef| {
                    (
                        inner_sdef,
                        sc.inner_request_variants.as_slice(),
                        sc.nested_clients.as_slice(),
                    )
                })
            } else {
                descend_struct(resolved, arm.type_name)
                    .map(|inner_sdef| (inner_sdef, &[] as &[UnionVariant], &[] as &[ScopedClient]))
            };

            match descend_target {
                Some((inner_sdef, inner_variants, inner_scopes)) => {
                    let inner_mod_ident =
                        format_ident!("{}", to_capnp_module_name(&inner_sdef.name));
                    let inner_mod = quote! { crate::#capnp_mod::#inner_mod_ident };
                    let inner_arms = decoder_arms(
                        resolved,
                        capnp_mod,
                        &inner_mod,
                        Some(inner_sdef),
                        inner_variants,
                        inner_scopes,
                    );
                    quote! {
                        #mod_path::Which::#pascal(r) => {
                            __path.push(#disc);
                            let r = r?;
                            match r.which()? {
                                #(#inner_arms)*
                                #[allow(unreachable_patterns)]
                                _ => anyhow::bail!(
                                    "unknown request union discriminant (not in this schema revision)"
                                ),
                            }
                        }
                    }
                }
                None => quote! {
                    #mod_path::Which::#pascal(_) => {
                        __path.push(#disc);
                    }
                },
            }
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::resolve::ResolvedSchema;

    fn variant(name: &str, type_name: &str, scope: &str, exempt: bool) -> UnionVariant {
        UnionVariant {
            name: name.to_owned(),
            type_name: type_name.to_owned(),
            description: String::new(),
            scope: scope.to_owned(),
            scope_exempt: exempt,
            cli_hidden: false,
            doc_example: String::new(),
            vfs_path: String::new(),
            vfs_kind: String::new(),
            vfs_bulk: false,
            vfs_hidden: false,
            vfs_mac: String::new(),
            dispatch_mac: String::new(),
            dispatch_public: String::new(),
        }
    }

    fn union_field(name: &str, type_name: &str, disc: u16) -> FieldDef {
        FieldDef {
            name: name.to_owned(),
            type_name: type_name.to_owned(),
            description: String::new(),
            fixed_size: None,
            optional: false,
            slot_offset: 0,
            section: FieldSection::Pointer,
            discriminant_value: disc,
            serde_rename: None,
            domain_type: None,
        }
    }

    fn plain_field(name: &str, type_name: &str) -> FieldDef {
        FieldDef {
            discriminant_value: 0xFFFF,
            ..union_field(name, type_name, 0)
        }
    }

    fn union_struct(name: &str, fields: Vec<FieldDef>) -> StructDef {
        let discriminant_count =
            fields.iter().filter(|f| f.discriminant_value != 0xFFFF).count() as u16;
        StructDef {
            name: name.to_owned(),
            fields,
            has_union: true,
            domain_type: None,
            origin_file: None,
            data_words: 1,
            pointer_words: 2,
            discriminant_count,
            discriminant_offset: 0,
            union_arms: vec![],
        }
    }

    /// A root union with a scope subtree and a nested scope below it derives
    /// full multi-level numeric paths, and every leaf appears exactly once.
    #[test]
    fn nested_scope_leaves_get_full_paths() {
        let schema = ParsedSchema {
            request_variants: vec![
                variant("status", "Void", "query", false),
                variant("repo", "RepositoryRequest", "", false),
            ],
            response_variants: vec![],
            structs: vec![
                union_struct(
                    "RepositoryRequest",
                    vec![
                        plain_field("id", "Text"),
                        union_field("create", "Text", 0),
                        union_field("worktree", "WorktreeRequest", 1),
                    ],
                ),
                union_struct(
                    "WorktreeRequest",
                    vec![
                        plain_field("name", "Text"),
                        union_field("add", "Text", 0),
                        union_field("remove", "Void", 1),
                    ],
                ),
            ],
            scoped_clients: vec![ScopedClient {
                factory_name: "repo".into(),
                client_name: "RepositoryClient".into(),
                scope_fields: vec![plain_field("id", "Text")],
                inner_request_variants: vec![variant("create", "Text", "write", false)],
                inner_response_variants: vec![],
                capnp_inner_response: "repository_response".into(),
                nested_clients: vec![ScopedClient {
                    factory_name: "worktree".into(),
                    client_name: "WorktreeClient".into(),
                    scope_fields: vec![plain_field("name", "Text")],
                    inner_request_variants: vec![
                        variant("add", "Text", "write", false),
                        variant("remove", "Void", "manage", false),
                    ],
                    inner_response_variants: vec![],
                    capnp_inner_response: "worktree_response".into(),
                    nested_clients: vec![],
                }],
            }],
            enums: vec![],
            request_struct: Some(union_struct(
                "RegistryRequest",
                vec![
                    plain_field("id", "UInt64"),
                    union_field("status", "Void", 0),
                    union_field("repo", "RepositoryRequest", 1),
                ],
            )),
            response_struct: None,
        };
        let resolved = ResolvedSchema::from(&schema);
        let leaves = collect_method_leaves(&resolved);

        let by_symbol: std::collections::HashMap<&str, &MethodLeaf> =
            leaves.iter().map(|l| (l.symbolic.as_str(), l)).collect();
        assert_eq!(by_symbol.len(), leaves.len(), "no duplicate symbolic leaves");

        assert_eq!(by_symbol["status"].path, vec![0]);
        assert_eq!(by_symbol["repo.create"].path, vec![1, 0]);
        assert_eq!(by_symbol["repo.worktree.add"].path, vec![1, 1, 0]);
        assert_eq!(by_symbol["repo.worktree.remove"].path, vec![1, 1, 1]);
        assert_eq!(by_symbol["repo.worktree.add"].scope, "write");
        assert!(!by_symbol.contains_key("repo"), "a scope selector is not a leaf");

        // The generated decoder descends the same tree: nested Which paths
        // and every discriminant push must appear in the emitted code.
        let generated = generate_body_decoder("registry", &resolved).to_string();
        for needle in [
            "registry_request :: Which :: Status",
            "repository_request :: Which :: Create",
            "worktree_request :: Which :: Add",
            "bounded_reader_options",
            "from_message",
        ] {
            assert!(
                generated.contains(needle),
                "generated decoder must contain `{needle}`:\n{generated}"
            );
        }
    }

    /// An `Option<T>`-shaped pure union is payload, not method identity.
    #[test]
    fn option_shaped_unions_do_not_extend_the_leaf() {
        let option_struct = StructDef {
            name: "OptionText".into(),
            fields: vec![
                union_field("none", "Void", 0),
                union_field("some", "Text", 1),
            ],
            has_union: true,
            domain_type: None,
            origin_file: None,
            data_words: 1,
            pointer_words: 1,
            discriminant_count: 2,
            discriminant_offset: 0,
            union_arms: vec![],
        };
        let schema = ParsedSchema {
            request_variants: vec![variant("lookup", "OptionText", "query", false)],
            response_variants: vec![],
            structs: vec![option_struct],
            scoped_clients: vec![],
            enums: vec![],
            request_struct: Some(union_struct(
                "LookupRequest",
                vec![union_field("lookup", "OptionText", 0)],
            )),
            response_struct: None,
        };
        let resolved = ResolvedSchema::from(&schema);
        let leaves = collect_method_leaves(&resolved);
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].path, vec![0]);
        assert_eq!(leaves[0].symbolic, "lookup");
    }

    // ── Strict dispatch pair (v16 §6, WS-D) ────────────────────────────────

    const MAC: &str = "internal:pq-hybrid";

    fn dispatch_variant(
        name: &str,
        type_name: &str,
        scope: &str,
        mac: &str,
        public: &str,
    ) -> UnionVariant {
        let mut v = variant(name, type_name, scope, false);
        v.dispatch_mac = mac.to_owned();
        v.dispatch_public = public.to_owned();
        v
    }

    fn simple_schema(variants: Vec<UnionVariant>) -> ResolvedSchema<'static> {
        // Leak a box: test schemas live for the process lifetime anyway.
        let schema = Box::leak(Box::new(ParsedSchema {
            request_variants: variants,
            response_variants: vec![],
            structs: vec![],
            scoped_clients: vec![],
            enums: vec![],
            request_struct: None,
            response_struct: None,
        }));
        ResolvedSchema::from(schema)
    }

    /// A leaf with neither dispatch annotation is a compile error — an
    /// unannotated runtime row can never exist, and nothing inherits public.
    #[test]
    fn an_unannotated_leaf_is_a_compile_error() {
        let resolved = simple_schema(vec![dispatch_variant("load", "Void", "write", "", "")]);
        let generated = generate_method_policy_rows("model", &resolved).to_string();
        assert!(generated.contains("compile_error"), "{generated}");
        assert!(generated.contains("neither a $dispatchMac"), "{generated}");
    }

    #[test]
    fn both_annotations_on_one_leaf_is_a_compile_error() {
        let resolved = simple_schema(vec![dispatch_variant(
            "load",
            "Void",
            "write",
            MAC,
            "reason",
        )]);
        let generated = generate_method_policy_rows("model", &resolved).to_string();
        assert!(generated.contains("BOTH $dispatchMac"), "{generated}");
    }

    /// `$dispatchPublic` is legal only on leaves: an unannotated arm below a
    /// public selector does NOT inherit public — it fails the build instead.
    #[test]
    fn public_never_inherits_through_a_hand_dispatched_selector() {
        let selector = union_struct(
            "InnerRequest",
            vec![
                union_field("a", "Text", 0),
                union_field("b", "Void", 1),
            ],
        );
        let schema = Box::leak(Box::new(ParsedSchema {
            request_variants: vec![dispatch_variant("outer", "InnerRequest", "", "", "leaf-level reason")],
            response_variants: vec![],
            structs: vec![selector],
            scoped_clients: vec![],
            enums: vec![],
            request_struct: Some(union_struct(
                "SvcRequest",
                vec![union_field("outer", "InnerRequest", 0)],
            )),
            response_struct: None,
        }));
        let resolved = ResolvedSchema::from(schema);
        let leaves = collect_method_leaves(&resolved);
        assert_eq!(leaves.len(), 2, "both pure-union arms are leaves");
        for leaf in &leaves {
            assert!(
                leaf.dispatch_mac.is_empty() && leaf.dispatch_public.is_empty(),
                "public must not flow to {}",
                leaf.symbolic
            );
        }
        let generated = generate_method_policy_rows("svc", &resolved).to_string();
        assert!(generated.contains("compile_error"), "{generated}");
        assert!(generated.contains("never inherited"), "{generated}");
    }

    /// A `$dispatchMac` label DOES inherit through a hand-dispatched pure
    /// union: unannotated arms resolve to their selector's label.
    #[test]
    fn a_mac_label_inherits_through_a_hand_dispatched_selector() {
        let selector = union_struct(
            "InnerRequest",
            vec![
                union_field("a", "Text", 0),
                union_field("b", "Void", 1),
            ],
        );
        let schema = Box::leak(Box::new(ParsedSchema {
            request_variants: vec![dispatch_variant("outer", "InnerRequest", "", MAC, "")],
            response_variants: vec![],
            structs: vec![selector],
            scoped_clients: vec![],
            enums: vec![],
            request_struct: Some(union_struct(
                "SvcRequest",
                vec![union_field("outer", "InnerRequest", 0)],
            )),
            response_struct: None,
        }));
        let resolved = ResolvedSchema::from(schema);
        let leaves = collect_method_leaves(&resolved);
        assert_eq!(leaves.len(), 2);
        assert!(leaves.iter().all(|l| l.dispatch_mac == MAC));

        // And both inherited leaves generate real rows — no compile_error.
        let generated = generate_method_policy_rows("svc", &resolved).to_string();
        assert!(!generated.contains("compile_error"), "{generated}");
        assert!(
            generated.contains("Level :: Internal"),
            "{generated}"
        );
    }

    /// Grammar failures at codegen are compile errors, never runtime rows:
    /// unknown axes, malformed grammar, and system low through `$dispatchMac`.
    #[test]
    fn invalid_label_grammar_is_a_compile_error() {
        for bad in [
            "internal",
            "internal:",
            "topsecret:pq-hybrid",
            "internal:pq",
            "internal:pq-hybrid:",
            " internal:pq-hybrid",
            "public:unverified", // system low through the MAC path
        ] {
            let resolved = simple_schema(vec![dispatch_variant("load", "Void", "write", bad, "")]);
            let generated = generate_method_policy_rows("model", &resolved).to_string();
            assert!(generated.contains("compile_error"), "{bad}: {generated}");
        }
    }

    /// A public leaf needs a nonempty trimmed reason and no `$scope`.
    #[test]
    fn a_public_leaf_requires_a_real_reason_and_no_scope() {
        let empty_reason = simple_schema(vec![dispatch_variant("ping", "Void", "", "", "   ")]);
        let generated = generate_method_policy_rows("mcp", &empty_reason).to_string();
        assert!(generated.contains("compile_error"), "{generated}");

        let scoped_public =
            simple_schema(vec![dispatch_variant("ping", "Void", "query", "", "reason")]);
        let generated = generate_method_policy_rows("mcp", &scoped_public).to_string();
        assert!(generated.contains("cannot be dispatch-public"), "{generated}");
    }

    /// The generated row carries every §6.1 field: auth requirement, Hybrid
    /// signature policy, credential use, mutation semantics keyed off the
    /// checked scope-action block, system-low transitional label, parsed
    /// target label, and (only for public) the reason.
    #[test]
    fn generated_rows_carry_the_full_section_6_1_policy() {
        let resolved = simple_schema(vec![
            dispatch_variant("status", "Void", "query", MAC, ""),
            dispatch_variant("commit", "Text", "write", "secret:pq-hybrid", ""),
            dispatch_variant("ping", "Void", "", "", "genuinely unauthenticated leaf"),
        ]);
        let generated = generate_method_policy_rows("model", &resolved).to_string();

        assert!(!generated.contains("compile_error"), "{generated}");
        for needle in [
            "AuthenticationRequirement :: CredentialRequired",
            "AuthenticationRequirement :: UnauthenticatedAllowed",
            "SignaturePolicy :: TokenBound",
            "SignaturePolicy :: UnauthenticatedOrTokenBound",
            "CryptoSuite :: Hybrid",
            "MutationSemantics :: NaturallyIdempotent",
            "SYSTEM_LOW_LABEL",
            "dispatch_label",
            "Level :: Internal",
            "Assurance :: PqHybrid",
            "Level :: Secret",
            "\"genuinely unauthenticated leaf\"",
        ] {
            assert!(generated.contains(needle), "missing `{needle}`:\n{generated}");
        }

        // Read-class rows carry no mutation semantics; the mutating one does.
        // (Asserted structurally by the runtime validator over the real
        // inventory; here the counts must line up with the three leaves.)
        let mutation_count = generated.matches("MutationSemantics :: NaturallyIdempotent").count();
        assert_eq!(mutation_count, 1, "only `commit` (write) is mutating");
    }

    /// Opus #1518 F-J drift coverage: an IMPORTED pure-union payload
    /// (`origin_file` set) terminates the leaf at its annotated arm — the
    /// decoder and the inventory share that one row by construction, so the
    /// coarsening can never produce an unannotated or unlisted leaf.
    #[test]
    fn an_imported_pure_union_terminates_on_the_annotated_arm() {
        let imported = StructDef {
            name: "ImportedChoice".into(),
            fields: vec![
                union_field("x", "Text", 0),
                union_field("y", "Void", 1),
            ],
            has_union: true,
            domain_type: None,
            origin_file: Some("streaming".into()),
            data_words: 1,
            pointer_words: 1,
            discriminant_count: 2,
            discriminant_offset: 0,
            union_arms: vec![],
        };
        let schema = Box::leak(Box::new(ParsedSchema {
            request_variants: vec![dispatch_variant("fetch", "ImportedChoice", "query", MAC, "")],
            response_variants: vec![],
            structs: vec![imported],
            scoped_clients: vec![],
            enums: vec![],
            request_struct: Some(union_struct(
                "FetchRequest",
                vec![union_field("fetch", "ImportedChoice", 0)],
            )),
            response_struct: None,
        }));
        let resolved = ResolvedSchema::from(schema);
        let leaves = collect_method_leaves(&resolved);
        assert_eq!(leaves.len(), 1, "imported union arms are payload, not identity");
        assert_eq!(leaves[0].path, vec![0]);
        assert_eq!(leaves[0].dispatch_mac, MAC);

        // The shared tree walk drives the decoder identically: the imported
        // arm is a terminal decoder arm, and its single inventory row derives
        // from the same walk — one row, one decoder arm, no drift.
        let decoder = generate_body_decoder("fetch", &resolved).to_string();
        assert!(
            decoder.contains("fetch_request :: Which :: Fetch"),
            "{decoder}"
        );
        let rows = generate_method_policy_rows("fetch", &resolved).to_string();
        assert!(!rows.contains("compile_error"), "{rows}");
    }
}
