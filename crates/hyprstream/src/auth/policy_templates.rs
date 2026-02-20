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
    ]
}

/// Get a template by name
pub fn get_template(name: &str) -> Option<&'static PolicyTemplate> {
    get_templates().iter().find(|t| t.name == name)
}
