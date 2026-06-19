//! Built-in policy templates for bootstrapping access control
//!
//! Templates provide pre-configured policy rules that can be applied
//! via CLI or PolicyService RPC.
//!
//! Also defines [`SERVICE_BASE_POLICIES`] — mandatory service-to-service
//! authorization rules that are always injected into the Casbin enforcer.

/// A single Casbin policy rule with named fields.
///
/// Casbin format: `p, subject, domain, resource, action, effect`
/// This struct makes field order impossible to get wrong.
#[derive(Clone, Copy)]
pub struct ServicePolicyRule {
    pub subject: &'static str,
    pub domain: &'static str,
    pub resource: &'static str,
    pub action: &'static str,
    pub effect: &'static str,
}

impl ServicePolicyRule {
    /// Convert to the Vec<String> Casbin expects for `add_policy()` / `add_policies()`.
    pub fn to_vec(&self) -> Vec<String> {
        vec![
            self.subject.into(),
            self.domain.into(),
            self.resource.into(),
            self.action.into(),
            self.effect.into(),
        ]
    }

    /// Convert to a CSV line: `p, subject, domain, resource, action, effect`
    pub fn to_csv_line(&self) -> String {
        format!(
            "p, {}, {}, {}, {}, {}",
            self.subject, self.domain, self.resource, self.action, self.effect
        )
    }
}

/// A role grouping rule.
///
/// Casbin format: `g, user, role`
pub struct ServiceGrouping {
    pub user: &'static str,
    pub role: &'static str,
}

impl ServiceGrouping {
    pub fn to_vec(&self) -> Vec<String> {
        vec![self.user.into(), self.role.into()]
    }

    pub fn to_csv_line(&self) -> String {
        format!("g, {}, {}", self.user, self.role)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Base service-to-service rules (always injected into the enforcer)
// ─────────────────────────────────────────────────────────────────────────────

/// Base service-to-service authorization rules.
///
/// These rules are ALWAYS present in the Casbin enforcer — injected at init
/// and re-injected after every reload. They grant each local service the
/// permissions it needs to operate. Templates and user-facing rules are
/// added ON TOP of these rules, never replacing them.
///
/// Because they're structured Rust data, field count/order is compile-time
/// validated. No CSV parsing needed.
pub const SERVICE_BASE_POLICIES: &[ServicePolicyRule] = &[
    // PolicyService: CA — issues tokens, manages policy rules, assigns roles
    ServicePolicyRule { subject: "service:policy", domain: "*", resource: "*", action: "*", effect: "allow" },
    // Token issuance: any service can call PolicyService::issueToken (capnp type name = IssueToken)
    ServicePolicyRule { subject: "service:*", domain: "*", resource: "policy:IssueToken", action: "manage", effect: "allow" },
    // Services that perform policy authorization checks
    ServicePolicyRule { subject: "service:registry", domain: "*", resource: "policy:*", action: "check", effect: "allow" },
    ServicePolicyRule { subject: "service:model", domain: "*", resource: "policy:*", action: "check", effect: "allow" },
    ServicePolicyRule { subject: "service:worker", domain: "*", resource: "policy:*", action: "check", effect: "allow" },
    ServicePolicyRule { subject: "service:oai", domain: "*", resource: "policy:*", action: "check", effect: "allow" },
    ServicePolicyRule { subject: "service:tui", domain: "*", resource: "policy:*", action: "check", effect: "allow" },
    ServicePolicyRule { subject: "service:discovery", domain: "*", resource: "policy:*", action: "check", effect: "allow" },
    ServicePolicyRule { subject: "service:notification", domain: "*", resource: "policy:*", action: "check", effect: "allow" },
    ServicePolicyRule { subject: "service:metrics", domain: "*", resource: "policy:*", action: "check", effect: "allow" },
    ServicePolicyRule { subject: "service:mcp", domain: "*", resource: "policy:*", action: "*", effect: "allow" },
    // Service key resolution: any service can resolve other service keys via policy
    ServicePolicyRule { subject: "service:oauth", domain: "*", resource: "policy:*", action: "query", effect: "allow" },
    ServicePolicyRule { subject: "service:registry", domain: "*", resource: "policy:*", action: "query", effect: "allow" },
    ServicePolicyRule { subject: "service:model", domain: "*", resource: "policy:*", action: "query", effect: "allow" },
    ServicePolicyRule { subject: "service:worker", domain: "*", resource: "policy:*", action: "query", effect: "allow" },
    ServicePolicyRule { subject: "service:oai", domain: "*", resource: "policy:*", action: "query", effect: "allow" },
    ServicePolicyRule { subject: "service:tui", domain: "*", resource: "policy:*", action: "query", effect: "allow" },
    ServicePolicyRule { subject: "service:discovery", domain: "*", resource: "policy:*", action: "query", effect: "allow" },
    ServicePolicyRule { subject: "service:notification", domain: "*", resource: "policy:*", action: "query", effect: "allow" },
    ServicePolicyRule { subject: "service:metrics", domain: "*", resource: "policy:*", action: "query", effect: "allow" },
    // Registry access (services that create RegistryClient)
    ServicePolicyRule { subject: "service:model", domain: "*", resource: "registry:*", action: "*", effect: "allow" },
    ServicePolicyRule { subject: "service:oai", domain: "*", resource: "registry:*", action: "*", effect: "allow" },
    ServicePolicyRule { subject: "service:flight", domain: "*", resource: "registry:*", action: "*", effect: "allow" },
    // Model/inference access (services that create ModelClient)
    ServicePolicyRule { subject: "service:model", domain: "*", resource: "model:*", action: "*", effect: "allow" },
    ServicePolicyRule { subject: "service:oai", domain: "*", resource: "model:*", action: "*", effect: "allow" },
    // Discovery: services announce and query endpoints
    ServicePolicyRule { subject: "service:discovery", domain: "*", resource: "discovery:*", action: "*", effect: "allow" },
    ServicePolicyRule { subject: "service:model", domain: "*", resource: "discovery:*", action: "*", effect: "allow" },
    ServicePolicyRule { subject: "service:oai", domain: "*", resource: "discovery:*", action: "*", effect: "allow" },
    ServicePolicyRule { subject: "service:worker", domain: "*", resource: "discovery:*", action: "*", effect: "allow" },
    // NOTE: Federation (`federation:register` resource) is deny-by-default.
    // This is the single, atproto-style trust gate that governs both:
    //   - third-party clients registering via CIMD (Client ID Metadata
    //     Documents — published metadata at an HTTPS origin), and
    //   - remote peer servers whose entity statements we'd resolve via
    //     FederationKeyResolver / DiscoveryService.
    //
    // Operators opt in by applying the `federation-open` template, or
    // by allowlisting specific origins:
    //   p, https://app.partner.org,        *, federation:register, check, allow
    //   p, https://*.partner.org,          *, federation:register, check, allow
    //   p, https://hyprstream.partner.org, *, federation:register, check, allow
    //
    // The same rule shape covers a CIMD client at app.partner.org and
    // a peer hyprstream instance at hyprstream.partner.org. Wizard may
    // suggest `federation-open` at setup time for operators who want
    // to join the broader MCP/atproto network.
    //
    // Infra services — scoped to their own domain
    ServicePolicyRule { subject: "service:oauth", domain: "*", resource: "oauth:*", action: "*", effect: "allow" },
    ServicePolicyRule { subject: "service:streams", domain: "*", resource: "streams:*", action: "*", effect: "allow" },
    ServicePolicyRule { subject: "service:event", domain: "*", resource: "event:*", action: "*", effect: "allow" },
];

/// Convert structured base policies to the Vec<Vec<String>> Casbin expects.
pub fn base_policies_to_vec() -> Vec<Vec<String>> {
    SERVICE_BASE_POLICIES.iter().map(ServicePolicyRule::to_vec).collect()
}

/// Convert structured base policies to a CSV string (for default_policy_csv only).
pub fn base_policies_to_csv() -> String {
    SERVICE_BASE_POLICIES
        .iter()
        .map(ServicePolicyRule::to_csv_line)
        .collect::<Vec<_>>()
        .join("\n")
}

// ─────────────────────────────────────────────────────────────────────────────
// Policy templates
// ─────────────────────────────────────────────────────────────────────────────

/// Built-in policy template
pub struct PolicyTemplate {
    pub name: &'static str,
    pub description: &'static str,
    /// Static policy rules, or None if generated dynamically.
    pub policies: Option<&'static [ServicePolicyRule]>,
    /// Static grouping rules (g, user, role).
    pub groupings: Option<&'static [ServiceGrouping]>,
}

impl PolicyTemplate {
    /// Get expanded policy rules for this template. All templates now
    /// carry static `policies`; the dynamic-OS-username "local" template
    /// was removed because its expansion (`anonymous → *`) was
    /// indistinguishable from `public-*` templates and the name
    /// misrepresented its scope.
    pub fn expanded_policies(&self) -> Vec<ServicePolicyRule> {
        self.policies.map(<[_]>::to_vec).unwrap_or_default()
    }

    /// Convert all rules (policies + groupings) to a CSV string.
    ///
    /// Used only for generating initial policy.csv content and for
    /// CLI display. The enforcer path uses structured data directly.
    pub fn to_csv(&self) -> String {
        let mut lines = Vec::new();
        let policies = self.expanded_policies();
        for p in &policies {
            lines.push(p.to_csv_line());
        }
        if let Some(groupings) = self.groupings {
            for g in groupings {
                lines.push(g.to_csv_line());
            }
        }
        lines.join("\n")
    }
}

/// Get all available policy templates.
///
/// Templates are **composable and additive**: each `policy apply-template`
/// adds the template's rules to the running policy via Casbin's
/// `add_policies` / `add_grouping_policies` (deduplicating, so repeat
/// application is idempotent). Operators can layer templates as
/// needed — e.g. `cimd-open` + `public-inference` + `ttt.user`.
///
/// No template removes rules. To revoke a previously-applied template
/// operators use `policy edit` to remove the lines, or
/// `policy rollback <ref>` to restore an earlier commit.
pub fn get_templates() -> &'static [PolicyTemplate] {
    &[
        PolicyTemplate {
            name: "federation-open",
            description: "Open federation: accept third-party apps and remote peer servers from any HTTPS origin (atproto-style, recommended for MCP/peer compatibility)",
            policies: Some(&[
                // Unified federation trust gate. Covers BOTH:
                //   - CIMD third-party clients (their HTTPS metadata URL)
                //   - Remote peer servers (their entity-statement URL)
                // Operators wanting tighter posture replace with origin-
                // specific allow rules; the matcher is the same.
                ServicePolicyRule {
                    subject: "*",
                    domain: "*",
                    resource: "federation:register",
                    action: "check",
                    effect: "allow",
                },
            ]),
            groupings: None,
        },
        PolicyTemplate {
            name: "public-inference",
            description: "Anonymous users can infer and query models",
            policies: Some(&[
                ServicePolicyRule { subject: "anonymous", domain: "*", resource: "model:*", action: "infer.generate", effect: "allow" },
                ServicePolicyRule { subject: "anonymous", domain: "*", resource: "model:*", action: "query.status", effect: "allow" },
            ]),
            groupings: None,
        },
        PolicyTemplate {
            name: "public-read",
            description: "Anonymous users can query the registry (read-only)",
            policies: Some(&[
                ServicePolicyRule { subject: "anonymous", domain: "*", resource: "registry:*", action: "query.status", effect: "allow" },
            ]),
            groupings: None,
        },
        PolicyTemplate {
            name: "ttt.user",
            description: "TTT user: infer, query, persist on all models",
            policies: Some(&[
                ServicePolicyRule { subject: "ttt.user", domain: "*", resource: "model:*", action: "infer.generate", effect: "allow" },
                ServicePolicyRule { subject: "ttt.user", domain: "*", resource: "model:*", action: "query.status", effect: "allow" },
                ServicePolicyRule { subject: "ttt.user", domain: "*", resource: "model:*", action: "query.delta", effect: "allow" },
                ServicePolicyRule { subject: "ttt.user", domain: "*", resource: "model:*", action: "persist.save", effect: "allow" },
                ServicePolicyRule { subject: "ttt.user", domain: "*", resource: "model:*", action: "persist.export", effect: "allow" },
                ServicePolicyRule { subject: "ttt.user", domain: "*", resource: "model:*", action: "persist.snapshot", effect: "allow" },
            ]),
            groupings: None,
        },
        PolicyTemplate {
            name: "ttt.agent",
            description: "TTT agent: alias for ttt.user",
            policies: None,
            groupings: Some(&[
                ServiceGrouping { user: "ttt.agent", role: "ttt.user" },
            ]),
        },
        // ─────────────────────────────────────────────────────────────────
        // Inference mesh (#319) — host↔host pipeline RPC authorization.
        //
        // SECURITY MODEL (deny-by-default, fail-closed, least-privilege):
        //   * The mesh AUTHORITY actions (`infer.stage`, `delta.submit`, and the
        //     `mesh.rpc` umbrella invoke right) are NEVER granted to `*`,
        //     `service:*`, or `anonymous`. They are granted ONLY to specific,
        //     non-wildcard `service:inference:host-<id>` subjects (#328), and
        //     ONLY within a single tenant's domain. A host enrolled for tenant
        //     `acme` thus cannot stage on tenant `beta` (the Casbin matcher does
        //     an exact `r.dom == p.dom` check unless the rule's domain is `*`,
        //     which mesh-authority rules MUST NOT use).
        //   * The READ ability (`query.status`) MAY be granted to a *group* of
        //     hosts via Casbin grouping (`g`) — see the `mesh-host-group`
        //     template — so an operator can give a fleet read visibility without
        //     per-host duplication. Authority stays per-host.
        //
        // RESOURCE SHAPE — `mesh://<tenant>/<job>/<…>` (UCAN-translatable):
        //   The tenant is BOTH the Casbin domain (for ambient isolation) AND the
        //   first path segment (so the caveat survives translation to a future
        //   UCAN delegation, which has no domain column). The 1:1 mapping is:
        //
        //     Casbin: p, service:inference:host-7, acme, mesh://acme/job/j7, infer.stage, allow
        //     UCAN:   { iss: <policy-root>, aud: host-7, sub: <policy-root>,
        //               cmd: /mesh/infer/stage,
        //               pol: [["==", ".tenant", "acme"], ["==", ".job", "j7"]] }
        //
        //   action ⇄ cmd, subject ⇄ aud, and the tenant/job caveats live in the
        //   resource path so a future UCAN layer maps without loss. `query.status`
        //   ⇄ /mesh/query/status, `delta.submit` ⇄ /mesh/delta/submit, the
        //   `mesh.rpc` umbrella ⇄ /mesh/rpc.
        //
        // This template is a WORKED EXAMPLE for two hosts in tenant `acme`.
        // Operators copy it and substitute their real host labels / tenant /
        // job via `policy edit` (the per-host authority lines are intentionally
        // explicit, never wildcarded). It is NOT injected into the base policy.
        PolicyTemplate {
            name: "mesh-host",
            description: "Inference mesh (#319): per-host pipeline RPC grants for tenant `acme` (worked example — substitute real host labels/tenant; authority is strictly non-wildcard per-host)",
            policies: Some(&[
                // Router → host: ModelService stages activations onto enrolled
                // hosts within the tenant. `service:model` is the router identity.
                ServicePolicyRule { subject: "service:model", domain: "acme", resource: "mesh://acme/*", action: "infer.stage", effect: "allow" },
                // Host → host: peer activation hand-off + status read.
                ServicePolicyRule { subject: "service:inference:host-1", domain: "acme", resource: "mesh://acme/*", action: "mesh.rpc", effect: "allow" },
                ServicePolicyRule { subject: "service:inference:host-1", domain: "acme", resource: "mesh://acme/*", action: "infer.stage", effect: "allow" },
                ServicePolicyRule { subject: "service:inference:host-1", domain: "acme", resource: "mesh://acme/*", action: "query.status", effect: "allow" },
                ServicePolicyRule { subject: "service:inference:host-2", domain: "acme", resource: "mesh://acme/*", action: "mesh.rpc", effect: "allow" },
                ServicePolicyRule { subject: "service:inference:host-2", domain: "acme", resource: "mesh://acme/*", action: "infer.stage", effect: "allow" },
                ServicePolicyRule { subject: "service:inference:host-2", domain: "acme", resource: "mesh://acme/*", action: "query.status", effect: "allow" },
                // Host → aggregator: TTT delta submission (authority).
                ServicePolicyRule { subject: "service:inference:host-1", domain: "acme", resource: "mesh://acme/*", action: "delta.submit", effect: "allow" },
                ServicePolicyRule { subject: "service:inference:host-2", domain: "acme", resource: "mesh://acme/*", action: "delta.submit", effect: "allow" },
            ]),
            groupings: None,
        },
        // Group-policy variant: grant the READ ability to a whole fleet via a
        // Casbin role, WITHOUT per-host duplication, while authority stays
        // per-host (granted by `mesh-host`). The grouping rule `g,
        // service:inference:host-<id>, mesh-readers` makes each host inherit the
        // role's grants; the role gets ONLY `query.status` (read), never an
        // authority action. Operators add `g, <host>, mesh-readers` per host.
        PolicyTemplate {
            name: "mesh-host-group",
            description: "Inference mesh (#319): grant the read ability (query.status) to the `mesh-readers` host group for tenant `acme` (authority stays per-host via `mesh-host`)",
            policies: Some(&[
                ServicePolicyRule { subject: "mesh-readers", domain: "acme", resource: "mesh://acme/*", action: "query.status", effect: "allow" },
            ]),
            groupings: Some(&[
                ServiceGrouping { user: "service:inference:host-1", role: "mesh-readers" },
                ServiceGrouping { user: "service:inference:host-2", role: "mesh-readers" },
            ]),
        },
        PolicyTemplate {
            name: "ttt.privileged",
            description: "TTT privileged: extends ttt.user with ttt.* operations",
            policies: Some(&[
                ServicePolicyRule { subject: "ttt.privileged", domain: "*", resource: "model:*", action: "ttt.train", effect: "allow" },
                ServicePolicyRule { subject: "ttt.privileged", domain: "*", resource: "model:*", action: "ttt.writeback", effect: "allow" },
                ServicePolicyRule { subject: "ttt.privileged", domain: "*", resource: "model:*", action: "ttt.evict", effect: "allow" },
                ServicePolicyRule { subject: "ttt.privileged", domain: "*", resource: "model:*", action: "ttt.zero", effect: "allow" },
            ]),
            groupings: Some(&[
                ServiceGrouping { user: "ttt.privileged", role: "ttt.user" },
            ]),
        },
    ]
}

/// Get a template by name
pub fn get_template(name: &str) -> Option<&'static PolicyTemplate> {
    get_templates().iter().find(|t| t.name == name)
}
