//! Compile-time scope registration via inventory pattern.
//!
//! Services can register scopes at compile time using inventory::submit!
//! This ensures all scopes are discoverable and documented.

/// Scope definition for compile-time registration.
pub struct ScopeDefinition {
    pub action: &'static str,
    pub resource: &'static str,
    pub example: &'static str,
    pub description: &'static str,
}

impl ScopeDefinition {
    pub const fn new(
        action: &'static str,
        resource: &'static str,
        example: &'static str,
        description: &'static str,
    ) -> Self {
        Self {
            action,
            resource,
            example,
            description,
        }
    }
}

// Collect all registered scopes
inventory::collect!(ScopeDefinition);

/// Get all registered scopes.
pub fn registered_scopes() -> impl Iterator<Item = &'static ScopeDefinition> {
    inventory::iter::<ScopeDefinition>()
}
