# hyprstream-rpc-std

Cap'n Proto service schemas and generated RPC clients for all standard hyprstream services.

## What it does

This crate bundles:

- **Cap'n Proto generated modules** for every first-party service: `inference_capnp`, `model_capnp`, `registry_capnp`, `policy_capnp`, `mcp_capnp`, `metrics_capnp`, `notification_capnp`, `discovery_capnp`, and more.
- **Generated typed clients** (`InferenceClient`, `ModelClient`, `PolicyClient`, etc.) produced by `hyprstream-rpc-derive`'s `generate_rpc_service!` macro.
- **WASM bindings** (`wasm_api.rs`): `wasm_bindgen` exports for the browser RPC client — `send_rpc`, `verify_signed_envelope`, `register_pq_trust`, `unregister_pq_trust`, `clear_pq_trust`.

## Targets

Compiles to both native (`rlib`) and `wasm32-unknown-unknown` (`cdylib`). The WASM build is the browser client bundle used by `www-cyberdione-ai`. The native build is used by `hyprstream` for embedding client logic.

## Architecture position

```
hyprstream-rpc           (transport, envelope, Cap'n Proto codec)
    ↑
hyprstream-rpc-std       ← you are here (schemas + generated clients + WASM)
    ↑
hyprstream               (embeds generated clients)
www-cyberdione-ai        (loads the WASM cdylib in a Web Worker)
```

## Key exports

| Export | Description |
|--------|-------------|
| `InferenceClient` | Typed client for the inference service |
| `ModelClient` | Typed client for the model registry |
| `PolicyClient` | Typed client for the policy/authz service |
| `wasm_api::send_rpc` | WASM: make a Cap'n Proto call over WebTransport |
| `wasm_api::verify_signed_envelope` | WASM: verify COSE envelope signature |
| `wasm_api::register_pq_trust` | WASM: bind an Ed25519 pubkey → ML-DSA-65 vk for hybrid enforcement |

## Browser npm package publication

CI packages this crate's `wasm-pack --target web` output as the scoped public npm package `@hyprstream/rpc`. **Merging a PR to `main` never publishes.** A push to `main` only builds, smoke-tests, and exposes the exact deterministic staging artifact identity (version, SHA-256, SRI, npm shasum, file inventory) in a read-only job summary — nothing is written to the registry, and the build/identity jobs hold no `id-token` and cannot mint an npm identity.

Staging publication is **manual only**: an operator re-runs the workflow on the target commit via the GitHub Actions UI, supplying the exact `expected_version` and `expected_sha256` copied from the exposed identity. The dispatch rebuilds, and the verifier inspects the **immutable tarball's own `package/package.json`** (not just the evidence file): it requires the tarball manifest's `name` and `version` to exactly equal the evidence and the operator pre-statement, enforces a commit-deterministic dev prerelease for staging (and exact crate/tag semver for production), refuses lifecycle scripts and runtime dependencies in the manifest, and only then publishes under the `staging` dist-tag. This prevents a build whose evidence claims a staging version from fronting a tarball whose manifest is a production semver. The staging version is commit-deterministic (`0.1.0-dev.<commit-count>.<short-sha>`), so the same main commit reproduces the same version across independent runs.

### External prerequisites (not facts)

These controls are **not** created by this repository's source and are **not** assumed to exist. The release owner must create and re-verify them (via the GitHub/npm API) before any publish can succeed; until then every publish fails closed:

- **npm trusted publisher** for `@hyprstream/rpc`, bound to repository `hyprstream/hyprstream`, workflow filename `.github/workflows/publish-rpc-std-wasm.yml`, and the `npm-staging` **environment**. npm allows one trusted publisher per package, so this single binding authorizes only the staging publish lane; the production lane is not authorized by it and will fail at npm until separately configured.
- **GitHub `npm-staging` environment** with required independent reviewers (no self-review/bypass), restricted to the `main` branch. A job referencing an environment that does not yet exist simply creates an unprotected one, so the environment and its reviewer rule must be established explicitly before use.
- For production only (not currently authorized): a separate `npm-production` environment and an **active repository tag ruleset** protecting the `hyprstream-rpc-v*` namespace. No tag ruleset exists today.

CI accepts no npm token and has no fallback registry or package namespace. Tarball provenance attestation runs in a **separate workflow file** (`.github/workflows/attest-rpc-std-wasm.yml`) whose workflow identity cannot satisfy the npm trusted publisher; npm publication additionally uses `--provenance` for registry-side Sigstore provenance. Each build emits the tarball plus `integrity.json` (SHA-256, npm SHA-512 SRI, file inventory, source commit/ref, toolchain). Downstream consumers should pin an exact version and lockfile integrity rather than a dist-tag.
