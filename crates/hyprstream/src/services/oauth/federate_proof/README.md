# Federate COSE proof dispatch

`federate_proof.rs` is the production Registry/Model admission adapter for the
Federate host-signed credential. The RPC verifier defers identity publication
only for the exact reserved issuer/client and profile, with a canonical,
nonzero signed signer-inventory digest. It rejects a missing digest or a mixed
session-generation claim; ordinary JWT and envelope-key behavior is unchanged.

The adapter accepts only the already signature-verified direct `at+jwt`. It
loads the Ed25519 confirmation key from `cnf.jwk` and the ML-DSA-65 public key,
inventory digest, and verified Dex provenance from signed claims. It validates
the hybrid-suite thumbprint and the original holder's COSE proof over the exact
credential, service, schema, request body, and expiry. Proof material in the
credential does not authorize the request.

Before dispatch, authenticated Policy validates the serving service, the fixed
profile and signer-inventory digest, the exact action/resource scope, and the
current DID tenant grant/suspension state. This decision is repeated for each
protected request. Only after Policy allows it does the adapter consume the
request ID in the existing shared PostgreSQL proof-replay store. Replay or
storage failure denies dispatch. The store retains replay digest/expiry only;
there is no session-row lookup, session URL, or database-backed `sid` contract.
`sid` is correlation, and credential expiry is at most five minutes.

Static service/approver enrollment remains independently resolved from the
operator-owned inventory; a static enrolled primary can never replace the
request's holder keys. The runtime factory installs this adapter only when the
authenticated Policy client and trusted proof enrollment are available. Missing
Policy/replay authority fails closed.

The relevant local compile/lint checks use the BuildQ wrapper. Exact-head CI
executes the Federate unit suite; compilation alone does not execute those
tests. Neither local tests nor CI prove live Dex claims, browser login, or the
staging Registry/Model path. Acceptance remains a sanitized live receipt for
Federate login, authenticated Registry RPC, pinned Model→Inference, and denial
after grant revocation. Worker validation is out of scope.
