# SPIFFE in hyprstream

> **Status: design direction — SPIFFE proper is NOT implemented.** hyprstream issues no SPIFFE IDs, serves no SPIFFE bundle endpoint, and has no `[spiffe]` config section. What *does* ship today is the SVID **concept** under different standards: workload identity is a WIMSE Workload Identity Token (`typ: "wit+jwt"`, minted by `encode_service_jwt` in `crates/hyprstream-rpc/src/auth/jwt.rs`; browser WITs via `POST /oauth/wit`, `crates/hyprstream/src/services/oauth/wit_bootstrap.rs`), node identity is did:web/did:key, and envelope signing is hybrid-PQC COSE. This page describes how a SPIFFE-compatible surface *would* layer over that reality if a SPIFFE-consumer integration ever needs it. Nothing below is a configuration you can turn on today except where explicitly noted.

The intended model: hyprstream acts as a **SPIFFE trust-domain authority** — issuing SPIFFE-compatible identities for its own workloads and publishing a trust bundle that other SPIFFE consumers can verify against, with no SPIRE installation required on the hyprstream host.

This page is for operators considering integrating hyprstream with a SPIFFE-aware environment. If you don't run any SPIFFE consumers, you can skip it entirely — the WIT/did:web identity system works without any of this.

## The short version (proposed model)

- **Trust domain** = hyprstream instance FQDN (e.g. `hyprstream.acme.com`)
- **SPIFFE IDs** would look like `spiffe://hyprstream.acme.com/service/model`
- **Trust bundle** would be served at `https://hyprstream.acme.com/.well-known/spiffe/bundle`
- **Federation** with other trust domains would be gated by the same `federation:register` policy that already controls OAuth/CIMD federation — one trust decision, multiple protocols

## Why FQDN-aligned

The same FQDN already serves OAuth issuer metadata, OIDC discovery, and did:web documents, and is the policy subject for client/peer federation rules. Adding SPIFFE as one more identity protocol over that DNS name would mean:

- No new DNS to provision, no new cert to manage
- The TLS chain that validates OAuth/OIDC also validates the SPIFFE bundle fetch
- One Casbin rule covers everything:

  ```
  p, https://*.partner.org, *, federation:register, check, allow
  ```

  ...allows partner.org's OAuth issuer, their CIMD-served clients, their did:web users — and, under this design, their SPIFFE-issued workloads.

This matches the atproto compatibility model: one FQDN = one identity entity, multiple protocols layered over it.

## What exists today instead

The SVID role is filled by WIMSE Workload Identity Tokens:

- **Service WITs** — `encode_service_jwt` (`crates/hyprstream-rpc/src/auth/jwt.rs`) mints an EdDSA-signed JWT with `typ: "wit+jwt"` for hyprstream services.
- **Browser WITs** — `POST /oauth/wit` (`crates/hyprstream/src/services/oauth/wit_bootstrap.rs`) exchanges a PKCE-issued `at+jwt` for a short-lived `wit+jwt` bound to the caller's Ed25519 pubkey.
- **Node identity** — did:web documents at the FQDN; envelope signatures use hybrid-PQC COSE (EdDSA + ML-DSA-65).

A future SPIFFE surface would be a re-encoding of these primitives (WIT ≈ JWT-SVID, did:web JWKS ≈ trust bundle), not a parallel identity system.

## Proposed SPIFFE ID scheme

Under this design, hyprstream services would receive SPIFFE IDs structured as:

```
spiffe://<trust_domain>/service/<service-name>
```

with the trust domain defaulting to the host portion of `oauth.issuer_url`. Per-user or per-workload SPIFFE IDs would not be issued — SPIFFE is workload identity; *user* identity lives in did:web (`did:web:hyprstream.acme.com:users:alice`). The two systems address different layers.

> [!warning] Trust-domain choice would be sticky
> Once SVIDs have been issued under a trust domain, changing it breaks every existing SVID. Pick deliberately at setup.

## Federation policy (this part works today)

The `federation:register` policy gate is real and is the single trust decision that governs whether hyprstream accepts identities from a given remote origin, regardless of protocol. To allow a partner origin:

```bash
hyprstream quick policy edit
# add: p, https://your-trust-domain.example.com, *, federation:register, check, allow
hyprstream quick policy apply
```

Apply the `federation-open` template (or run `hyprstream wizard --enable-federation`) to enable open federation; allowlist specific origins for stricter posture. A SPIFFE integration would sit behind this same gate.

## Scope limits (would apply even if implemented)

- **JWT-SVID only.** No X.509-SVID issuance is planned. If you need mTLS with X.509-SVIDs (e.g., service mesh sidecars), run SPIRE alongside hyprstream and federate the trust domains.
- **No upstream-authority mode.** hyprstream would not be a SPIRE upstream CA; other SPIRE deployments would federate with it as peers.
- **No SPIFFE for MCP clients.** MCP clients use OAuth flows (CIMD or DCR).

## Relationship to other identity protocols

| Layer | Identifier | Protocol | Status |
|---|---|---|---|
| User identity | `did:web:hyprstream.acme.com:users:alice` | did:web + OAuth/OIDC | shipped |
| OAuth client (with FQDN) | `https://app.example.com/client.json` | CIMD | shipped |
| OAuth client (no FQDN) | UUID issued by hyprstream | DCR (RFC 7591) | shipped |
| hyprstream workload | WIMSE WIT (`wit+jwt`) | EdDSA JWT + COSE envelopes | shipped |
| hyprstream workload (SPIFFE view) | `spiffe://hyprstream.acme.com/service/model` | SPIFFE JWT-SVID | **design only** |
| Peer hyprstream | `https://hyprstream.partner.org` | OAuth/OIDC issuer + did:web | shipped (gated by `federation:register`) |

## References

- SPIFFE specifications: https://github.com/spiffe/spiffe/tree/main/standards
- SPIFFE Workload API: https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE_Workload_API.md
- SPIFFE Federation spec: https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE_Federation.md
- WIMSE (Workload Identity in Multi-System Environments): https://datatracker.ietf.org/wg/wimse/about/
