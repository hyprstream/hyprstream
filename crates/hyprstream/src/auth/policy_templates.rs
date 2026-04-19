//! Built-in policy templates for bootstrapping access control
//!
//! Templates provide pre-configured policy rules that can be applied
//! via CLI or PolicyService RPC.
//!
//! Also defines [`SERVICE_BASE_RULES`] — mandatory service-to-service
//! authorization rules that are always present in every policy.csv.

/// Base service-to-service authorization rules.
///
/// These rules MUST be present in every policy.csv — they grant each local
/// service the permissions it needs to operate. Templates and user-facing
/// rules are appended ON TOP of these rules, never replacing them.
///
/// Applied by:
/// - `DEFAULT_POLICY_CSV` on first-time init
/// - `handle_apply_template()` when templates are applied
/// - `BootstrapManager::apply_template()` during wizard
pub const SERVICE_BASE_RULES: &str = r#"# Service-to-service base rules (do not remove)
# PolicyService: CA — issues tokens, manages policy rules, assigns roles
p, service:policy, *, *, *, allow

# Token self-renewal: any service can request its own token renewal
p, service:*, *, policy:issue-token, manage, allow

# Services that perform policy authorization checks
p, service:registry, *, policy:*, check, allow
p, service:model, *, policy:*, check, allow
p, service:worker, *, policy:*, check, allow
p, service:oai, *, policy:*, check, allow
p, service:tui, *, policy:*, check, allow
p, service:discovery, *, policy:*, check, allow
p, service:notification, *, policy:*, check, allow
p, service:metrics, *, policy:*, check, allow
p, service:mcp, *, policy:*, *, allow

# Registry access (services that create RegistryClient)
p, service:model, *, registry:*, *, allow
p, service:oai, *, registry:*, *, allow
p, service:flight, *, registry:*, *, allow

# Model/inference access (services that create ModelClient)
p, service:model, *, model:*, *, allow
p, service:oai, *, model:*, *, allow

# Discovery: services announce and query endpoints
p, service:discovery, *, discovery:*, *, allow
p, service:model, *, discovery:*, *, allow
p, service:oai, *, discovery:*, *, allow
p, service:worker, *, discovery:*, *, allow

# Infra services — scoped to their own domain
p, service:oauth, *, oauth:*, *, allow
p, service:streams, *, streams:*, *, allow
p, service:event, *, event:*, *, allow
"#;

/// Built-in policy template
pub struct PolicyTemplate {
    pub name: &'static str,
    pub description: &'static str,
    /// Static rules, or None if rules are generated dynamically.
    pub rules: Option<&'static str>,
}

impl PolicyTemplate {
    /// Get the rules content, expanding dynamic templates.
    pub fn expanded_rules(&self) -> String {
        if let Some(rules) = self.rules {
            rules.to_owned()
        } else if self.name == "local" {
            let user = hyprstream_rpc::envelope::RequestIdentity::anonymous().user().to_owned();
            format!("# Full access for {user}\np, {user}, *, *, *, allow\n")
        } else {
            String::new()
        }
    }
}

/// Get all available policy templates
pub fn get_templates() -> &'static [PolicyTemplate] {
    &[
        PolicyTemplate {
            name: "local",
            description: "Full access for the current local user",
            rules: None, // Dynamic: expands to current OS username
        },
        PolicyTemplate {
            name: "public-inference",
            description: "Anonymous users can infer and query models",
            rules: Some(r#"# Public inference access
p, anonymous, *, model:*, infer.generate, allow
p, anonymous, *, model:*, query.status, allow
"#),
        },
        PolicyTemplate {
            name: "public-read",
            description: "Anonymous users can query the registry (read-only)",
            rules: Some(r#"# Public read access (registry only)
p, anonymous, *, registry:*, query.status, allow
"#),
        },
        PolicyTemplate {
            name: "ttt.user",
            description: "TTT user: infer, query, persist on all models",
            rules: Some(
                r#"# TTT user — infer, query, persist on all models
p, ttt.user, *, model:*, infer.generate, allow
p, ttt.user, *, model:*, query.status, allow
p, ttt.user, *, model:*, query.delta, allow
p, ttt.user, *, model:*, persist.save, allow
p, ttt.user, *, model:*, persist.export, allow
p, ttt.user, *, model:*, persist.snapshot, allow
"#,
            ),
        },
        PolicyTemplate {
            name: "ttt.agent",
            description: "TTT agent: alias for ttt.user",
            rules: Some(
                r#"# ttt.agent is an alias for ttt.user
g, ttt.agent, ttt.user
"#,
            ),
        },
        PolicyTemplate {
            name: "ttt.privileged",
            description: "TTT privileged: extends ttt.user with ttt.* operations",
            rules: Some(
                r#"# ttt.privileged inherits ttt.user and adds ttt.* operations
g, ttt.privileged, ttt.user
p, ttt.privileged, *, model:*, ttt.train, allow
p, ttt.privileged, *, model:*, ttt.writeback, allow
p, ttt.privileged, *, model:*, ttt.evict, allow
p, ttt.privileged, *, model:*, ttt.zero, allow
"#,
            ),
        },
    ]
}

/// Get a template by name
pub fn get_template(name: &str) -> Option<&'static PolicyTemplate> {
    get_templates().iter().find(|t| t.name == name)
}
