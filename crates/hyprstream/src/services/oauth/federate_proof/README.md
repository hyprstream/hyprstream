# H3b.1: disabled request-local proof primitive

This is a partial H3b slice, stacked on H3a. It is compiled only with the existing
`hyprstream/postgres` feature. It has no route, runtime provider installer, or
production credential-handle constructor. `Consumer::default()` denies. The only
provider implementation and handle construction are in offline tests.

The primitive repeats the full H3a response bindings against immutable credential
facts, checks collision inventory/keys and the positive database epoch, constructs
a stack-local primary resolver, and invokes the existing v16 hybrid verifier.
Static approver/service resolution is delegated; static primary fallback is absent.
It compares the exact token hash, body, schema and service before returning evidence.
One bounded provider call is required on every valid handle use, including reuse;
provider failure never recovers via subject or key lookup. Component kids reuse
H4a's framing implementation; v16 cnf and replay encodings remain unchanged.

**Success is not dispatch admission.** There is intentionally no handler callback
or permit. A duplicate proof still verifies cryptographically in this primitive.
The existing dispatch method-policy, MAC and replay gates remain unchanged, and
no dynamic profile is connected to them by this patch. Account/grant provenance
is not permission.

## Remaining H3b boundary

1. Add strict signed generation/profile handling to the verified credential path,
   and a connection-owned handle constructor that rejects delegated/service/bearer
   substitutes. Today's `Claims` has no generation field; direct cnf validation
   binds to the envelope key in `RequestService::verify_claims`. Do not weaken that
   static path to accommodate separate browser proof keys.
2. Implement `CurrentPrimary` through authenticated, capability-scoped Policy RPC
   (Policy itself uses its direct reader). Include current jti revocation, bounded
   response decoding, the complete trusted inventory loader and no positive cache.
   The fixture provider is not evidence of RPC authentication or DB revocation.
3. Integrate the primitive into dispatch before the existing method/MAC/replay
   admission gates. Implement session-bound activation instead of adapting the
   current subject-only cache, with one fresh lookup on every use. Preserve original
   holder evidence through mediation.
4. Run F3 at that admission boundary: two verifier processes, H1 persisted epoch,
   disposable PostgreSQL replay adapter with unique `(namespace, request_id)`,
   duplicate denial after restart, fresh-ID control and missing-history denial.
   This patch does not implement or claim F3, durable replay, or handler isolation.
5. Apply the operator-selected H2 checks at each API request and before each
   distinct tool call. A previously authorized stream may finish, including after
   later revocation or credential/session expiry; no per-token, chunk or timer
   polling. The next request/tool call must recheck. The amended contract awaits
   K3 review; no H2 grants or streaming behavior are implemented here. Neither
   this primitive nor H3a authorizes enablement.

## Local gates

Use the host's BuildQ/hyprstream-build wrapper for all Cargo commands and hooks:

```
cargo test -p hyprstream --features postgres --lib --release h3b_
cargo test -p hyprstream --features postgres --lib --release federate_source
cargo clippy -p hyprstream --features postgres --all-targets --release -- -D warnings
```

Tests use generated fixture keys and opaque fixture credential bytes, not live
tokens. They do not verify a host JWT or call a live issuer/Policy/database. No
production-ready or complete-H3b claim should be based on these tests.
