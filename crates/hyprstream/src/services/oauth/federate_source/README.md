# Federate verifier

The accepted A0 direct-exchange contract uses the exact API origin from the
server's configured OAuth issuer URL (the same identity as the RPC expected
audience). Enabling Federate requires a canonical HTTPS origin without a path,
trailing slash, credentials, query or fragment. Request headers, browser inputs
and JWT claims cannot choose this authority. `Verifier::default()` is disabled;
production installs `for_host()` with the validated configuration.

The upstream issuer remains `https://login.federate.to`; the website client,
origin and callback remain the existing staging values. The browser must use
the selected API origin as `resource` and in both host/resource fields of its
nonce commitment, and verify that origin and its `/oauth/token` endpoint in the
possession transcript. No wire field names change. Existing staging vectors
remain valid for the configured staging origin. These offline checks do not
prove a live browser callback, launch, SSH or inference workflow.

The fixed HTTPS JWKS URL uses normal TLS verification, no redirects, a 2-second
deadline, 64 KiB body limit, at most 16 RSA signing keys and a 60-second cache.
An unknown kid triggers at most one refresh per lookup; global ingress/rate
limits are still required before exposing this component. Tokens cannot choose
the authority URL. Header URLs/embedded keys, private JWK members, duplicate
JSON members (including nested unknown claims), noncanonical base64url and
unsupported algorithms fail closed. No stale-key fallback after cache expiry.

The caller must supply trusted wall-clock seconds, fresh CSPRNG challenge IDs,
and Policy-derived `AuthorityBinding`; the latter is transcript input, not a
grant this module can authorize. Grant revision is opaque pending H2's decision.
The consuming challenge verifies a separately verified source at redemption,
exact source/code/commitment/challenge binding and both possession signatures.
Only transcript bytes are exposed, never a generic signing API. Keys must be
dedicated fresh browser request-proof keys, never login/envelope/service keys.

This host verifier cannot prove browser callback execution, state comparison,
PKCE redemption, key freshness or that a code was actually redeemed. It checks
the signed commitment and Dex's required `c_hash`. If the token includes
`at_hash`, a supplied access token is checked; a host call without it only
checks canonical hash encoding as allowed by A0. Access tokens are not accepted
as source credentials. Callback validation and browser key storage are separate
work. Typed evidence has private fields and no browser deserialization or
credential serialization.

Consuming a Rust challenge prevents reuse of that value only. This is not
durable replay prevention: H1 must still atomically consume issuer/client/jti,
nonce and source hash with admission. A re-created challenge or another process
may verify the same source again until that transaction. No working-login claim.

## Offline checks

Run Cargo through the fleet BuildQ helper:

```
cargo test -p hyprstream --lib federate_source
cargo clippy -p hyprstream --lib --tests -- -D warnings
```

`framing-vectors.json` copies the published encoding-only A0 vector. Its patterned
PQ bytes and fake source token are not a signing identity. `signature-vector.json`
contains actual RS256, Ed25519 and pure ML-DSA-65 signatures over the same framing
with real test keys. Its signed source token includes the required Federate
`atproto` connector DID claim. Native verification consumes the generated vector.
`test-rsa.pem` and deterministic seeds 7/8 are public disposable test material.

To regenerate the signature vector, install tooling-only dependencies with
scripts disabled: `@noble/post-quantum@0.7.1` and `@noble/curves@2.4.0` into an
isolated directory, then run `node generate-vector.mjs /path/to/node_modules`.
The generator emits JSON on stdout and self-verifies both signatures using the
browser-compatible libraries. No npm dependency is added to the application;
Node execution does not constitute a live-browser login acceptance test.
