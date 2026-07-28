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

CI packages this crate's `wasm-pack --target web` output as the unscoped public npm package `hyprstream-rpc-std`. Pull requests build, pack, and install the tarball into a clean browser-bundled consumer but never publish it. A push to `main` publishes a unique `0.1.0-dev.<run>.<attempt>.<commit>` prerelease only under the `staging` dist-tag. Production publication requires the exact tag `hyprstream-rpc-std-v0.1.0` (kept in sync with this crate's Cargo version) and publishes immutable version `0.1.0` under `latest`.

Publication is fail-closed and credentialless: npmjs must already own the package and configure GitHub Actions trusted publishing for repository `hyprstream/hyprstream` and workflow `.github/workflows/publish-rpc-std-wasm.yml`. The GitHub `npm-production` environment and `hyprstream-rpc-std-v*` tag must be protected before production use. CI accepts no npm token and has no fallback registry or package namespace.

Each build emits the npm tarball plus `integrity.json` containing its SHA-256, npm SHA-512 SRI, file inventory, source commit/ref, and toolchain. Push builds also receive a GitHub artifact attestation, while npm publication uses `--provenance` for registry-side Sigstore provenance. Downstream consumers should pin an exact version and lockfile integrity rather than a dist-tag.
