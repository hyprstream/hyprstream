# SPIFFE in hyprstream

> **Status:** design — see [`docs/plans/phase-4-spiffe-fqdn-aligned.md`](../docs/plans/phase-4-spiffe-fqdn-aligned.md) for the rollout plan. This document describes the *intended* model; capabilities ship in stages (4a/4b/4c).

hyprstream acts as a **SPIFFE trust-domain authority**: it issues SPIFFE-compatible identities for its own workloads and publishes a trust bundle that other SPIFFE consumers can verify against. No SPIRE installation is required on the hyprstream host.

This page is for operators integrating hyprstream with a SPIFFE-aware environment. If you don't run any SPIFFE consumers today, you can skip it — hyprstream services use SPIFFE identities internally only when configured to.

## The short version

- **Trust domain** = hyprstream instance FQDN (e.g. `hyprstream.acme.com`)
- **SPIFFE IDs** look like `spiffe://hyprstream.acme.com/service/model`
- **Trust bundle** is served at `https://hyprstream.acme.com/.well-known/spiffe/bundle`
- **Federation** with other trust domains is gated by the same `federation:register` policy that controls OAuth/CIMD federation — one trust decision, four protocols

## Why FQDN-aligned

The same FQDN already serves OAuth issuer metadata, OIDC discovery, did:web documents, and is the policy subject for client/peer federation rules. Adding SPIFFE as one more identity protocol over that DNS name means:

- No new DNS to provision, no new cert to manage
- The TLS chain that validates OAuth/OIDC also validates the SPIFFE bundle fetch
- One Casbin rule covers everything:

  ```
  p, https://*.partner.org, *, federation:register, check, allow
  ```

  ...allows partner.org's OAuth issuer, their CIMD-served clients, their did:web users, AND their SPIFFE-issued workloads.

This matches the atproto compatibility model: one FQDN = one identity entity, multiple protocols layered over it.

## Trust-domain configuration

The trust domain defaults to the host portion of `oauth.issuer_url`:

```toml
[oauth]
issuer_url = "https://hyprstream.acme.com"
# → trust domain = hyprstream.acme.com
# → SPIFFE IDs = spiffe://hyprstream.acme.com/...
```

To override (e.g., split-horizon deployments where the public name differs from the internal mesh name):

```toml
[spiffe]
trust_domain = "hyprstream.internal.acme.com"
```

> [!warning] Trust-domain choice is sticky
> Once SVIDs have been issued under a trust domain, changing it breaks every existing SVID. External systems federating with hyprstream will reject SVIDs from the old domain until they update their federation config. Pick deliberately at setup.

## SPIFFE ID scheme for hyprstream workloads

hyprstream services receive SPIFFE IDs structured as:

```
spiffe://<trust_domain>/service/<service-name>
```

Examples:
- `spiffe://hyprstream.acme.com/service/policy`
- `spiffe://hyprstream.acme.com/service/model`
- `spiffe://hyprstream.acme.com/service/registry`
- `spiffe://hyprstream.acme.com/service/oauth`
- `spiffe://hyprstream.acme.com/service/discovery`

Per-user or per-workload SPIFFE IDs are not currently issued — SPIFFE is workload identity; *user* identity lives in did:web (`did:web:hyprstream.acme.com:users:alice`). The two systems address different layers.

## Bundle endpoint

The trust bundle (the public key material consumers need to verify our SVIDs) is served at:

```
GET https://<trust_domain>/.well-known/spiffe/bundle
```

The response is JWKS-shaped, conforming to the SPIFFE Trust Domain and Bundle spec. It includes all active and grace-period signing keys, so consumers that cache the bundle gracefully handle our key rotations.

Caching behaviour mirrors our OAuth JWKS endpoint: standard HTTP cache headers, suitable for downstream CDNs and SPIRE Server's bundle refresh logic.

## Federating with hyprstream from SPIRE

If you run SPIRE Server in another trust domain and want to verify hyprstream's SVIDs:

```hcl
# spire-server.conf
federates_with "hyprstream.acme.com" {
  bundle_endpoint_url = "https://hyprstream.acme.com/.well-known/spiffe/bundle"
  bundle_endpoint_profile "https_web" {}
}
```

`https_web` profile means SPIRE relies on web PKI (standard TLS chain validation) to authenticate the bundle endpoint. Since hyprstream's bundle is served from the FQDN that also presents the trust-domain DNS cert, this Just Works.

For mutual federation (your workloads also need to be verifiable by hyprstream), apply the `federation:register` policy on the hyprstream side:

```bash
hyprstream quick policy edit
# add: p, https://your-trust-domain.example.com, *, federation:register, check, allow
hyprstream quick policy apply
```

Then provide hyprstream with your bundle URL via configuration (Phase 4c — coming).

## Workload identity ingestion

When Phase 4b lands, hyprstream services *optionally* fetch their identity through a local SPIFFE Workload API socket instead of reading a key file at startup:

```toml
[spiffe.workload_api]
enabled = true
socket = "/run/hyprstream/workload-api.sock"
```

The PolicyService implements the Workload API and uses lightweight attestors to verify caller identity:

- **Unix attestor** — peer credentials over the Unix domain socket (UID/GID, executable path)
- **systemd unit attestor** — when the caller is a systemd-managed process, map the unit name to a SPIFFE ID

This replaces the "whoever can read the key file is the service" model with runtime attestation: a stolen key file alone no longer impersonates a service.

If you already run SPIRE Agent on the host and want hyprstream services to use it instead, point them at the SPIRE socket:

```toml
[spiffe.workload_api]
enabled = true
socket = "/run/spire/sockets/agent.sock"
```

In that case hyprstream is a SPIFFE *consumer* on the host; PolicyService still acts as the trust-domain authority for federation purposes.

## What hyprstream does NOT do (today)

- **No X.509-SVID issuance.** We issue **JWT-SVID** only. If you need mTLS with X.509-SVIDs (e.g., for service mesh sidecars expecting them), run SPIRE alongside hyprstream and federate the trust domains. X.509-SVID support is a possible future extension, not in the current plan.
- **No upstream-authority mode.** hyprstream is not a SPIRE upstream CA. Other SPIRE deployments cannot source their root from us; they federate with us as peers.
- **No SPIFFE for MCP clients.** MCP clients use OAuth flows (CIMD or DCR). SPIFFE for clients would be unusual; reach out if you have a use case.

## Relationship to other identity protocols

| Layer | Identifier | Protocol | Trust gate |
|---|---|---|---|
| User identity | `did:web:hyprstream.acme.com:users:alice` | did:web + OAuth/OIDC | per-user provisioning + Casbin |
| OAuth client (with FQDN) | `https://app.example.com/client.json` | CIMD | `federation:register` policy |
| OAuth client (no FQDN) | UUID issued by hyprstream | DCR (RFC 7591) | DCR capacity bound |
| Hyprstream workload | `spiffe://hyprstream.acme.com/service/model` | SPIFFE JWT-SVID | local attestation + bundle federation |
| Peer hyprstream | `https://hyprstream.partner.org` | OAuth/OIDC issuer + SPIFFE bundle | `federation:register` policy |

The `federation:register` policy is the single trust gate that governs whether hyprstream accepts identities from a given remote origin, regardless of which protocol they arrive over. Apply the `federation-open` template (or run `hyprstream wizard --enable-federation`) to enable open federation across all four protocols; allowlist specific origins for stricter posture.

## References

- SPIFFE specifications: https://github.com/spiffe/spiffe/tree/main/standards
- SPIFFE Workload API: https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE_Workload_API.md
- SPIFFE Federation spec: https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE_Federation.md
- Phase 4 implementation plan: [`docs/plans/phase-4-spiffe-fqdn-aligned.md`](plans/phase-4-spiffe-fqdn-aligned.md)
