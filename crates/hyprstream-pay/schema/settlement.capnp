# hyprstream-pay settlement/tariff RPC schema (#1399)
#
# The MIT-licensed protocol surface. The AGPL service impl in
# hyprstream::services::pay implements these interfaces.
#
# Capability scopes: ai.hyprstream.pay.settlement.{issue,status},
#                    ai.hyprstream.pay.tariff.{quote,resolve}

@0xbf8c3a91e5f27a09;

# ─── Shared types ────────────────────────────────────────────────────────────

struct UnitRef {
    issuerDid       @0 :Text;    # the liability-holder DID for this unit
    resourceClass   @1 :Text;    # e.g. "gpu.h100.seconds"
}

# ─── SettlementIssuer ────────────────────────────────────────────────────────

struct IssueRequest {
    settlementId    @0 :Text;    # internal settlement row id (opaque)
    attestation     @1 :Data;    # PQ-hybrid-signed settlement attestation
    unit            @2 :UnitRef; # target unit to issue
    destinationDid  @3 :Text;    # credit destination DID (pseudonymous)
    amountMinorLo   @4 :UInt64;  # u128 minor units (low 64 bits)
    amountMinorHi   @5 :UInt64;  # u128 minor units (high 64 bits)
    grantCid        @6 :Data;    # allocation grant CID (opaque)
}

struct IssueResponse {
    transferIdLo    @0 :UInt64;  # u128 transfer id (low 64 bits)
    transferIdHi    @1 :UInt64;  # u128 transfer id (high 64 bits)
    outcomeSeq      @2 :UInt64;  # journal sequence of the outcome
    ok              @3 :Bool;    # whether issuance succeeded
    error           @4 :Text;    # human-readable error detail if !ok
}

interface SettlementIssuer {
    # Issue credits from a verified settlement attestation.
    # Idempotent: same settlementId → same transferId, same outcome.
    issue @0 (req :IssueRequest) -> IssueResponse;

    # Query the status of a prior issuance (retry/recovery).
    status @1 (settlementId :Text) -> IssueResponse;
}

# ─── TariffProvider ──────────────────────────────────────────────────────────

struct TariffRequest {
    resourceClass   @0 :Text;    # e.g. "gpu.h100.seconds"
    quantity        @1 :UInt64;  # how much
    subjectDid      @2 :Text;    # who's paying (for tier/allowance)
    catalogVersion  @3 :Text;    # pinned catalog version
}

struct TariffQuote {
    unit            @0 :UnitRef; # unit to issue/debit
    priceMinorLo    @1 :UInt64;  # u128 minor units (low 64 bits)
    priceMinorHi    @2 :UInt64;  # u128 minor units (high 64 bits)
    expiresAt       @3 :UInt64;  # unix seconds — reservation deadline
    catalogVersion  @4 :Text;    # catalog this quote was computed against
    maxQuantum      @5 :UInt64;  # server-imposed ceiling
}

interface TariffProvider {
    # Get a priced quote for a resource quantity.
    quote @0 (req :TariffRequest) -> TariffQuote;

    # Resolve a unit reference for cross-cell federation.
    resolveUnit @1 (issuerDid :Text, resourceClass :Text) -> UnitRef;
}
