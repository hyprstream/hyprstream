# hyprstream-rpc-build

Build-time helpers: Cap'n Proto schema compilation, annotation extraction, TypeScript codegen, and RPC diagram generation.

## What it does

### Library (`lib.rs`)

- `compile_schemas(schema_dir, out_dir, import_paths, names)` — compile `.capnp` files via `capnpc`, save raw `CodeGeneratorRequest` (`.cgr`) files, and extract annotation metadata to JSON. Used in `build.rs` of every crate with Cap'n Proto schemas.
- Schema types (`ParsedSchema`, `StructDef`, `FieldDef`, `EnumDef`, …) — shared between `build.rs` code and the proc-macro crate so annotation metadata can be merged at compile time.

### Binaries

| Binary | Purpose |
|--------|---------|
| `hyprstream-ts-codegen` | Read compiled `.cgr` + metadata JSON → emit TypeScript client types for `www-cyberdione-ai` |
| `generate-rpc-diagrams` | Scan service schemas → emit Mermaid/PlantUML RPC topology diagrams |

## Usage (in a crate's `build.rs`)

```rust
fn main() {
    let schema_dir = Path::new("src/schemas");
    let out_dir = Path::new(&std::env::var("OUT_DIR").unwrap());
    hyprstream_rpc_build::compile_schemas(
        schema_dir,
        out_dir,
        &[],
        &["inference", "model", "registry"],
    );
}
```

## Architecture position

Build-time only — not linked into any runtime binary. The TypeScript codegen binary is run as part of the `www-cyberdione-ai` build to keep the browser client in sync with Cap'n Proto schema changes.
