# hyprstream-rpc-derive

Proc-macro crate for Cap'n Proto serialization and RPC service code generation.

## What it does

Provides four derive macros and one declarative macro:

| Macro | Purpose |
|-------|---------|
| `#[derive(ToCapnp)]` | Generate `impl ToCapnp for T` — serialize a Rust struct into a Cap'n Proto builder |
| `#[derive(FromCapnp)]` | Generate `impl FromCapnp for T` — deserialize a Cap'n Proto reader into a Rust struct |
| `#[derive(authorize)]` | Attach scope/policy checks to RPC handler methods |
| `#[service_factory]` | Generate service factory boilerplate (handler registration, dispatch table) |
| `generate_rpc_service!("name", scope)` | Emit a complete typed client + handler + dispatch fn from a compiled schema |
| `generate_rpc_client!("name")` | Emit portable client/data/metadata code only |
| `generate_rpc_server!("name", types_crate = contract_crate)` | Emit handler/dispatch/metadata only against a contract crate's generated types |

## Usage

```rust
use hyprstream_rpc::prelude::*;

#[derive(ToCapnp)]
#[capnp(registry_capnp::clone_request)]
pub struct CloneRequest {
    pub url: String,
    pub name: String,
    #[capnp(rename = "maxDepth")]
    pub depth: u32,
}

#[derive(FromCapnp)]
#[capnp(registry_capnp::clone_response)]
pub struct CloneResponse {
    pub repo_id: String,
    pub success: bool,
}
```

## Field attributes

- `#[capnp(schema::path)]` on the struct — required, maps to the generated schema type.
- `#[capnp(skip)]` on a field — omit from serialization.
- `#[capnp(rename = "camelCase")]` on a field — use a different Cap'n Proto field name.

## Architecture position

```
hyprstream-rpc-derive    ← you are here (proc macros)
    ↓ (used by)
hyprstream-rpc           (re-exports the derive macros)
hyprstream-rpc-std       (generate_rpc_client! for each public service schema)
hyprstream-{services}    (generate_rpc_server! against rpc-std contracts)
hyprstream-{discovery,workers,...}
```
