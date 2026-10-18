# Federate runtime wiring — implementation plan

## Current source status
The OAuth runtime now installs `FederateIssuer` from `OAuthService::run` when the
`postgres` feature is enabled. It uses the existing authenticated `PolicyClient`
for prepare/commit and the OAuth service's active EdDSA access-token signing
key, resolving the current rotation slot for every token and falling back to
the startup CA JWT key as the existing OAuth path does. Startup fails if no
signing source is available; it does not silently leave the issuer disabled.
The host challenge/exchange routes are already mounted.

## Remaining proof
1. Exact-head review and CI for the factory/admission/signer wiring.
2. Local integration with the real Policy implementation: source-token verification,
   challenge/dual-proof exchange, DID-bound session persistence, and current grant
   enforcement on Registry and pinned Model→Inference requests.
3. Negative controls for wrong issuer/DID, missing grant, challenge replay, and
   revocation/suspension; then repeat against staging after protected deployment.
