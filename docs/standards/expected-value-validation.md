# Expected-value validation

Security verification must distinguish an exact expected value from a deliberate
protocol-specific exemption. An absent value is a configuration error and must
reject; it must never mean that the verifier has no constraint.

For JWT audience validation, use
`decode_with_expectation(..., AudienceExpectation::Exact(value))`. The only
way to omit that check is `AudienceExpectation::ExplicitlyUnchecked { reason }`,
where the reason documents the different protocol binding being verified. The
legacy `Option<&str>` compatibility entry points map `None` to a rejection so
that old callers fail closed while they are migrated.

Apply the same rule to issuer, nonce, `kid`, `cnf`/`jkt`, and every future
expected-value verifier:

1. Model the verifier's expectation as a required closed type, not `Option<T>`.
2. Make every bypass opt-in and named at the call site with its security reason.
3. Treat missing security configuration as an error before accepting input.
4. Add a regression test that proves the absent expectation is rejected.

## #1145 audit

| Site | Verdict | Disposition |
| --- | --- | --- |
| `server/middleware.rs` | Safe | Already supplies exact resource and issuer. |
| `services/xet.rs` and registry deployment credential validation | Safe | Already supply exact audiences. |
| `crypto::cose_sign::verify_composite` | Safe | A missing PQ anchor is explicitly rejected when Hybrid verification requires one. |
| Response-envelope optional signer keys | Intentional test-only compatibility | Compiled only for classical compatibility tests; production verification binds the service domain and key. |
| Refresh-token DPoP thumbprint | Conditional binding | An expectation exists only for a DPoP-bound refresh token; an unbound token cannot silently lose an existing binding. |
| `RegistryService::handle_list` policy check | Live fail-open | Fixed here: a policy RPC error now denies listing instead of exposing every repository. |
| `RequestService::expected_audience()` and MCP stdio configuration | Latent fail-open | Fixed here: a missing audience now reaches the decoder as `Missing` and rejects. |
| `services/oauth/auth.rs` and `services/oauth/introspection.rs` | Live once OAuth algorithm routing accepts production tokens | Owned by #1146 Lane B / PR #1147; deliberately not modified here. |
| `services/oauth/token_exchange.rs` | Mixed: ID-token verification intentionally uses the OIDC client audience; access-token downscoping was fail-open | Owned by #1146 Lane B / PR #1147; deliberately not modified here. |
| `services/policy.rs` service-JWT registration | Latent fail-open | Owned by #1146 Lane B / PR #1147; the compatibility path now fails closed until its exact audience is supplied. |

The broad `unwrap_or(true)`, `is_none()`, optional-config, and skipped-branch
search also found runtime tuning defaults, account-profile presentation defaults,
cache/rotation expiry sentinels, and deliberately anonymous/bootstrap transport
paths. They do not represent an absent *expected security value* and therefore
are outside this rule. The only policy-decision default-allow was registry
listing, above.

The durable rule is: **an expected security value is a required constraint;
absence rejects; exemptions are explicit, named, and justified.**
