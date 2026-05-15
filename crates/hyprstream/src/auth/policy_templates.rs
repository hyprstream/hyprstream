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
    /// Get expanded policy rules for this template.
    ///
    /// For the "local" template, dynamically expands to the current OS username.
    /// For static templates, returns the predefined rules.
    /// Returns owned Vec because the "local" template generates rules at runtime.
    #[allow(deprecated)]
    pub fn expanded_policies(&self) -> Vec<ServicePolicyRule> {
        if let Some(rules) = self.policies {
            rules.iter().map(|r| ServicePolicyRule {
                subject: r.subject,
                domain: r.domain,
                resource: r.resource,
                action: r.action,
                effect: r.effect,
            }).collect::<Vec<_>>()
        } else if self.name == "local" {
            let user = hyprstream_rpc::envelope::RequestIdentity::anonymous().user().to_owned();
            vec![ServicePolicyRule {
                subject: user.leak(),
                domain: "*",
                resource: "*",
                action: "*",
                effect: "allow",
            }]
        } else {
            Vec::new()
        }
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

/// Get all available policy templates
pub fn get_templates() -> &'static [PolicyTemplate] {
    &[
        PolicyTemplate {
            name: "local",
            description: "Full access for the current local user",
            policies: None, // Dynamic: expands to current OS username
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
