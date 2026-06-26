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
