//! Shared types for the settlement/tariff protocol (#1399).
//!
//! These types are the **MIT-licensed protocol surface** that
//! `cyberdione-product` (and any other consumer) links against. The AGPL
//! service implementation in `hyprstream::services::pay` consumes these traits
//! and provides the real settlement-store + ledger-credit path.

use serde::{Deserialize, Serialize};

/// A resource unit reference — the unit names its issuer (INV-1 from the
/// ledger crate). Carried in the protocol so a tariff quote can name the exact
/// credit unit that will be issued.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnitRef {
    /// The issuer's DID (the liability holder for this unit).
    pub issuer_did: String,
    /// e.g. `"gpu.h100.seconds"`.
    pub resource_class: String,
}

/// A request to issue credits from a verified settlement (#1399, PAY-06).
///
/// The `attestation` is a PQ-hybrid-signed blob proving a committed settlement
/// row exists in the restricted settlement store. The server re-verifies it
/// against the issuer's key material before calling the ledger's internal
/// credit path. A client assertion of "paid" is never sufficient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueRequest {
    /// Internal settlement row id (opaque to the product layer; the server
    /// resolves it against its own restricted settlement store).
    pub settlement_id: String,
    /// PQ-hybrid-signed settlement attestation (COSE composite). The server
    /// verifies this before issuing any credits.
    pub attestation: Vec<u8>,
    /// The unit to issue.
    pub unit: UnitRef,
    /// The DID of the credit destination (the purchaser's pseudonymous
    /// identity — never legal identity, per PAY-00 #6).
    pub destination_did: String,
    /// Amount in minor units (u128 as hi/lo since JSON/capnp don't natively
    /// carry u128). The server re-derives this from the settlement row; a
    /// client-supplied value is never trusted.
    pub amount_minor_lo: u64,
    pub amount_minor_hi: u64,
    /// The allocation grant CID this issuance backs (opaque bytes — the ledger
    /// correlates but never parses it).
    pub grant_cid: Vec<u8>,
}

impl IssueRequest {
    /// The full u128 amount.
    pub fn amount(&self) -> u128 {
        ((self.amount_minor_hi as u128) << 64) | (self.amount_minor_lo as u128)
    }
}

/// The response from an issuance request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IssueResponse {
    /// The deterministic transfer id (idempotent: same `settlement_id` → same
    /// transfer id across retries/crashes).
    pub transfer_id_lo: u64,
    pub transfer_id_hi: u64,
    /// The journal sequence that recorded this outcome.
    pub outcome_seq: u64,
    /// Whether the issuance succeeded.
    pub ok: bool,
    /// Human-readable error detail if `!ok`.
    pub error: Option<String>,
}

impl IssueResponse {
    /// The full u128 transfer id.
    pub fn transfer_id(&self) -> u128 {
        ((self.transfer_id_hi as u128) << 64) | (self.transfer_id_lo as u128)
    }
}

/// A tariff quote request — the product asks "what does this cost?" and the
/// server owns the price, catalog, and ceiling (#1399, PAY-00 #3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TariffRequest {
    /// e.g. `"gpu.h100.seconds"`.
    pub resource_class: String,
    /// How many units of the resource.
    pub quantity: u64,
    /// The DID of the subject who will pay (for tier/allowance resolution).
    pub subject_did: String,
    /// Pinned catalog version (the product may carry a cached version; the
    /// server validates it's still current).
    pub catalog_version: String,
}

/// A server-owned priced quote with a finite maximum quantum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TariffQuote {
    /// The unit that will be issued/debited.
    pub unit: UnitRef,
    /// Price in minor units (u128 as hi/lo).
    pub price_minor_lo: u64,
    pub price_minor_hi: u64,
    /// Quote expiry (unix seconds). The reservation must be placed before this.
    pub expires_at: u64,
    /// The catalog version this quote was computed against.
    pub catalog_version: String,
    /// Server-imposed ceiling on the quantum for a single operation.
    pub max_quantum: u64,
}

impl TariffQuote {
    /// The full u128 price.
    pub fn price(&self) -> u128 {
        ((self.price_minor_hi as u128) << 64) | (self.price_minor_lo as u128)
    }

    /// Whether this quote is still valid at the given unix time.
    pub fn is_valid_at(&self, now_unix: u64) -> bool {
        now_unix < self.expires_at
    }
}
