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
    /// is `$scopeExempt` or sits below a hand-dispatched pure union whose
    /// arms carry no annotations.
    pub scope: String,
    /// Whether the leaf variant is `$scopeExempt` (explicitly public).
    pub scope_exempt: bool,
}

/// Per-arm metadata resolved during the walk.
struct ArmMeta<'a> {
    name: &'a str,
    discriminant: u16,
    type_name: &'a str,
    scope: &'a str,
    scope_exempt: bool,
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
        ("", false),
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
    // producing an unannotated (build-failing) row.
    inherited: (&str, bool),
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
        let effective: (&str, bool) = if arm.scope.is_empty() && !arm.scope_exempt {
            inherited
        } else {
            (arm.scope, arm.scope_exempt)
        };

        if let Some(sc) = scopes.iter().find(|sc| sc.factory_name == arm.name) {
            // A scope selector: descend into its inner method union.
            walk_level(
                resolved,
                resolved.find_struct(arm.type_name),
                &sc.inner_request_variants,
                &sc.nested_clients,
                &path,
                &symbolic,
                effective,
                out,
            );
        } else if let Some(inner) = descend_struct(resolved, arm.type_name) {
            // A hand-dispatched pure union: its arms are method identity.
            walk_level(resolved, Some(inner), &[], &[], &path, &symbolic, effective, out);
        } else {
            out.push(MethodLeaf {
                path,
                symbolic,
                scope: effective.0.to_owned(),
                scope_exempt: effective.1,
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
/// Row derivation from the checked schema annotations:
/// - a `$scopeExempt` leaf is publicly dispatchable:
///   `UnauthenticatedAllowed` + `UnauthenticatedOrTokenBound` with the
///   mandatory Hybrid suite, with the exemption recorded as its public
///   reason;
/// - every `$scope`-annotated leaf requires a credential:
///   `CredentialRequired` + `TokenBound` with the mandatory Hybrid suite.
///
/// A leaf with neither annotation (and no annotated ancestor selector to
/// inherit from) is a **compile error** — an unannotated runtime row can
/// never exist (v16 §6).
pub fn generate_method_policy_rows(service_name: &str, resolved: &ResolvedSchema) -> TokenStream {
    let leaves = collect_method_leaves(resolved);

    let mut row_tokens: Vec<TokenStream> = Vec::new();
    for leaf in &leaves {
        if leaf.scope.is_empty() && !leaf.scope_exempt {
            let msg = format!(
                "method leaf '{}.{}' has neither a $scope/$capability annotation nor \
                 $scopeExempt; every generated dispatch-policy row must derive from a \
                 checked annotation (v16 §6)",
                service_name, leaf.symbolic
            );
            return quote! { ::core::compile_error!(#msg); };
        }
        let path = &leaf.path;
        let symbolic = &leaf.symbolic;
        let scope = &leaf.scope;
        let (authentication, signature_policy, public_reason) = if leaf.scope_exempt {
            (
                quote! { hyprstream_rpc::proof::policy::AuthenticationRequirement::UnauthenticatedAllowed },
                quote! {
                    hyprstream_rpc::proof::policy::SignaturePolicy::UnauthenticatedOrTokenBound {
                        suite: hyprstream_rpc::proof::policy::CryptoSuite::Hybrid,
                    }
                },
                quote! { Some("declared $scopeExempt in the service schema") },
            )
        } else {
            (
                quote! { hyprstream_rpc::proof::policy::AuthenticationRequirement::CredentialRequired },
                quote! {
                    hyprstream_rpc::proof::policy::SignaturePolicy::TokenBound {
                        suite: hyprstream_rpc::proof::policy::CryptoSuite::Hybrid,
                    }
                },
                quote! { None },
            )
        };
        row_tokens.push(quote! {
            hyprstream_rpc::proof::policy::GeneratedMethodPolicyRow {
                service: #service_name,
                leaf_path: &[#(#path),*],
                symbolic_path: #symbolic,
                scope_action: #scope,
                authentication: #authentication,
                signature_policy: #signature_policy,
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
}
