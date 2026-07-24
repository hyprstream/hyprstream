# PEP #1268 Status

**Branch:** feat/mac-rpc-pep
**Issue:** #1268 (MAC activation-gate T3, epic #1267)
**Status:** PR open — ready for kimi-k3 review (FINAL gate)

## Delivered

- `.fleet-coord/mac-pep-contract.md` — shared PEP contract published EARLY (consumed by b-1269/1270/1271/1272)
- `crates/hyprstream-rpc/src/auth/mac/dispatch_pep.rs` — `MacDispatchPep` trait, `MacDecision`/`MacDenyReason`, `RpcObjectLabelResolver`, fail-closed defaults (`DenyAllMacPep`, `DenyAllObjectResolver`), process-global install (`install_mac_dispatch_pep`/`global_mac_dispatch_pep`)
- `crates/hyprstream-rpc/src/service/dispatch.rs` — mandatory PEP gate wired between `verify_claims` and `handle_request` in `process_request`
- 10 unit tests, 921 workspace tests pass in release (`cargo nextest run --release --profile ci -p hyprstream-rpc`)

## Design decisions

- **Dormant-until-activated**: PEP gate is dormant (pass-through) until `install_mac_dispatch_pep` is called. This matches the documented MAC enforcement state and avoids breaking 921 existing tests. Activation is a deliberate operator choice (#1267 gate).
- **Clearance seam**: uses the existing `EnvelopeContext::security_context()` (Claims × VerifiedKeyMaterial, S1 invariant). #698 dependency flagged: until production clearance is wired, `security_context()` resolves to `None` → every request denies `NoClearance`.
- **Object-label seam**: `RpcObjectLabelResolver` trait with `DenyAllObjectResolver` default. S3/#569 schema annotations are the production resolver.
- **No permissive mode** (#547): missing PEP (when activated), missing clearance, missing label, or lattice-floor deny all block the handler.

## Dependencies flagged

- **#698** (production clearance provenance): the PEP structure is fail-closed with the clearance seam stubbed via `EnvelopeContext::security_context()`. Until #698 wires the `clearance` field on `Claims`, the PEP denies `NoClearance` for every request — correct fail-closed, not a bug.
- **S3/#569** (schema annotations): production `RpcObjectLabelResolver` needs schema-declared labels for static RPC methods.
