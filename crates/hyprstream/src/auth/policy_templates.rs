//! Built-in policy templates for bootstrapping access control
//!
//! Templates provide pre-configured policy rules that can be applied
//! via CLI or PolicyService RPC.

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
            let user = hyprstream_rpc::envelope::RequestIdentity::local().user().to_owned();
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
p, anonymous, *, model:*, infer, allow
p, anonymous, *, model:*, query, allow
"#),
        },
        PolicyTemplate {
            name: "public-read",
            description: "Anonymous users can query the registry (read-only)",
            rules: Some(r#"# Public read access (registry only)
p, anonymous, *, registry:*, query, allow
"#),
        },
        PolicyTemplate {
            name: "ttt.user",
            description: "TTT user: infer, query, write, and manage access on all models",
            rules: Some(
                r#"# TTT user — infer, query, write, manage on all models
# NOTE: 'manage' scope covers both ttt.zero (resetDelta) and shutdown.
# Task 11 will replace this with per-operation policies that gate shutdown
# separately from delta management operations.
p, ttt.user, *, model:*, infer, allow
p, ttt.user, *, model:*, query, allow
p, ttt.user, *, model:*, write, allow
p, ttt.user, *, model:*, manage, allow
"#,
            ),
        },
        PolicyTemplate {
            name: "ttt.agent",
            description: "TTT agent: alias for ttt.user (compound ops via generateStream; no direct delta manipulation until Task 11)",
            rules: Some(
                r#"# ttt.agent is an alias for ttt.user (interim; Task 11 will restrict to compound ops only)
g, ttt.agent, ttt.user
"#,
            ),
        },
        PolicyTemplate {
            name: "ttt.privileged",
            description: "TTT privileged: extends ttt.user with train scope for direct delta manipulation",
            rules: Some(
                r#"# ttt.privileged inherits ttt.user and adds train scope
g, ttt.privileged, ttt.user
p, ttt.privileged, *, model:*, train, allow
"#,
            ),
        },
    ]
}

/// Get a template by name
pub fn get_template(name: &str) -> Option<&'static PolicyTemplate> {
    get_templates().iter().find(|t| t.name == name)
}
