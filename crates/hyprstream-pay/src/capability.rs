//! Capability model for the settlement/tariff protocol (#1399).
//!
//! These are the JWT scope strings that gate the RPC handlers. The
//! `#[authorize]` macro on the AGPL service impl references these constants.
//!
//! Convention: `ai.hyprstream.pay.<domain>.<action>` — matching the
//! `ai.hyprstream.ledger.*` lexicon already in use.

/// Scope: invoke `SettlementIssuer::issue`. This is the purchased-credit
/// issuance path — it requires a verified settlement attestation AND this
/// capability. Neither alone is sufficient.
pub const SCOPE_SETTLEMENT_ISSUE: &str = "ai.hyprstream.pay.settlement.issue";

/// Scope: invoke `SettlementIssuer::status` (query a prior issuance's
/// outcome for retry/recovery).
pub const SCOPE_SETTLEMENT_STATUS: &str = "ai.hyprstream.pay.settlement.status";

/// Scope: invoke `TariffProvider::quote` (get a priced quote for a resource).
pub const SCOPE_TARIFF_QUOTE: &str = "ai.hyprstream.pay.tariff.quote";

/// Scope: invoke `TariffProvider::resolve_unit` (resolve a unit reference for
/// cross-cell federation).
pub const SCOPE_TARIFF_RESOLVE: &str = "ai.hyprstream.pay.tariff.resolve";

/// All settlement/tariff scopes, for registration with the scope registry.
pub const ALL_SCOPES: &[&str] = &[
    SCOPE_SETTLEMENT_ISSUE,
    SCOPE_SETTLEMENT_STATUS,
    SCOPE_TARIFF_QUOTE,
    SCOPE_TARIFF_RESOLVE,
];
