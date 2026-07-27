//! # hyprstream-pay — settlement/tariff RPC contract (MIT)
//!
//! The **protocol surface** for the pay-wave: `SettlementIssuer` +
//! `TariffProvider` traits, shared types, capability model, and a client stub.
//!
//! This crate is MIT-only (not dual MIT/AGPL like `hyprstream`). It contains
//! ONLY the protocol definitions. The AGPL service implementation — which
//! verifies PQ-hybrid attestations server-side, checks the restricted
//! settlement store, and calls the ledger's internal credit path — lives in
//! `hyprstream::services::pay`.
//!
//! ## Why a separate MIT crate (#1399)?
//!
//! `cyberdione-product` (and any other consumer) must link against the
//! settlement/tariff contract without inheriting the AGPL. The protocol/stub
//! crate is MIT; the service that actually moves money is AGPL. This is the
//! standard protocol/impl split: the wire format is free; the implementation
//! is copyleft.
//!
//! ## The two contracts
//!
//! - [`SettlementIssuer`] — verified settlement → deterministic credit
//!   issuance. Idempotent: same settlement id → same transfer id, same
//!   outcome, across retries and crashes. The only purchased-credit issuance
//!   surface.
//! - [`TariffProvider`] — resource → priced quote with a finite server-imposed
//!   ceiling. The server owns the price, catalog, and maximum quantum; clients
//!   cannot set these.
//!
//! ## Security model
//!
//! Peer auth is transport-level (TLS 1.3 on QUIC, UDS peer credentials on
//! iroh). Then capability-gated at the RPC layer (`#[authorize]` on handlers).
//! The attestation itself is PQ-hybrid-signed (EdDSA + ML-DSA-65 COSE
//! composite) and **re-verified server-side** — never trusts a client
//! assertion. This reuses the #1129 spend-authorization verification path.

pub mod capability;
pub mod types;

pub use capability::ALL_SCOPES;
pub use types::{IssueRequest, IssueResponse, TariffQuote, TariffRequest, UnitRef};

use async_trait::async_trait;

/// Errors from the settlement/tariff protocol.
#[derive(Debug, thiserror::Error)]
pub enum PayError {
    /// The settlement attestation failed PQ-hybrid signature verification or
    /// the issuer's key material is unknown.
    #[error("settlement attestation verification failed: {0}")]
    AttestationInvalid(String),

    /// No committed settlement row was found for the given settlement id.
    #[error("no committed settlement for id {0}")]
    SettlementNotFound(String),

    /// The settlement row exists but is not in the `settlement_committed`
    /// state — issuance requires a fully verified, server-validated
    /// settlement.
    #[error("settlement {0} is not committed (state: {1})")]
    SettlementNotCommitted(String, String),

    /// The settlement row exists and is committed, but credits were already
    /// issued for it. This is a success case (idempotent replay) — the
    /// original transfer id and outcome are returned.
    #[error("settlement {0} already issued")]
    AlreadyIssued(String),

    /// The ledger rejected the issuance (insufficient issuer liability,
    /// unknown destination account, internal error, etc.).
    #[error("ledger issuance failed: {0}")]
    LedgerError(String),

    /// The tariff request named an unknown resource class or catalog version.
    #[error("unknown resource class or catalog: {0}")]
    UnknownResourceClass(String),

    /// The quote has expired.
    #[error("quote expired")]
    QuoteExpired,

    /// The requested quantum exceeds the server-imposed maximum.
    #[error("quantum {requested} exceeds server maximum {max}")]
    QuantumExceedsMaximum { requested: u64, max: u64 },

    /// An internal error (transport, serialization, etc.).
    #[error("internal pay error: {0}")]
    Internal(String),
}

/// The settlement-issuer contract: only a verified, server-revalidated
/// settlement can cause purchased-credit issuance.
///
/// **Production impl (AGPL, `hyprstream::services::pay`):**
/// 1. Verify the attestation's PQ-hybrid signature against the issuer's key
///    material (reuse #1129 `verify_composite`).
/// 2. Check the restricted settlement store for a committed settlement row
///    matching `settlement_id`.
/// 3. Derive the deterministic transfer ID (blake3 of `settlement_id` +
///    `catalog_version` + line index — idempotent across retries/crashes).
/// 4. Call the ledger's internal `credit()` (the restricted path, NOT the
///    public `LedgerHandle::credit`).
/// 5. Return the idempotent outcome.
///
/// A client redirect, client-supplied `paid` field, Checkout completion
/// without a settled PaymentIntent, or unverified webhook can never reach
/// issuance. The externally-reachable generic ledger `credit` is NOT on this
/// path.
#[async_trait]
pub trait SettlementIssuer: Send + Sync {
    /// Issue credits from a verified settlement attestation.
    ///
    /// Idempotent: replaying the same `settlement_id` returns the original
    /// `IssueResponse` verbatim (same transfer id, same outcome). The server
    /// never generates a second issuance after an ambiguous timeout — it
    /// retries the same deterministic transfer id.
    async fn issue(&self, req: IssueRequest) -> Result<IssueResponse, PayError>;

    /// Query a prior issuance's status. Used for retry/recovery when the
    /// original `issue` response was lost (e.g. network failure after the
    /// ledger committed).
    async fn status(&self, settlement_id: &str) -> Result<IssueResponse, PayError>;
}

/// The tariff-provider contract: resource → priced quote.
///
/// The server owns the price, catalog, and maximum quantum. A client cannot
/// set the amount, price, or ceiling — those are derived server-side from the
/// resource class, catalog version, and subject tier.
#[async_trait]
pub trait TariffProvider: Send + Sync {
    /// Get a priced quote for a resource quantity. The quote has a finite
    /// expiry and a server-imposed maximum quantum.
    async fn quote(&self, req: TariffRequest) -> Result<TariffQuote, PayError>;

    /// Resolve a unit reference for cross-cell federation (map an issuer DID +
    /// resource class to the canonical `UnitRef`).
    async fn resolve_unit(
        &self,
        issuer_did: &str,
        resource_class: &str,
    ) -> Result<UnitRef, PayError>;
}
