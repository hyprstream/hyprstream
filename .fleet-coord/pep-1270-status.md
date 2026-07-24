# PEP-1270 Status — CAS ingest/read MAC PEP

**Branch:** `feat/mac-cas-pep`
**Issue:** [#1270](https://github.com/hyprstream/hyprstream/issues/1270) · T3 · epic [#1267](https://github.com/hyprstream/hyprstream/issues/1267)
**Status:** PR open, pending kimi-k3 review (final gate).

## What landed

- **NEW `mac/cas_pep.rs`**: CAS-native MAC PEP — consumes the canonical `MacDecision`, `MacDenyReason`, and `RpcObjectLabelResolver` contract; provides `CasPep`, the authority-bound `CasClearanceSource` seam, `DenyAllClearanceSource`, `MacCasAuthorizer` (implements `CasMountAuthorizer`), and `domain_label()` + `seal_label()` (trusted content-bound label derivation from `DedupDomain`). `MacDispatchPep` remains RPC-plane-only.
- **Substrate (`storage/cas/substrate.rs`)**: `put()` now derives the trusted label via `seal_label(domain, hint)` — no longer caller-asserted plumb-through. The hint parameter is a D1 restrict-only input from our own staging path.
- **Registry (`services/registry.rs`)**: read continuation (`execute_get_blob_stream`) performs a MAC PEP recheck before streaming any bytes. Verified subject and `verified_tenant` are captured from `EnvelopeContext` and threaded through the continuation; the registry CAS/provenance store remains in its node-global dedup domain.
- **Xet HTTP (`services/xet.rs`)**: `BootstrapCasAuthorizer` replaced with `MacCasAuthorizer::fail_closed()` at the xorb read handler.
- **Manifest (`storage/cas/manifest.rs`)**: doc updated — carrier field is now the trusted derived label, not plumb-through.
- **Mount (`storage/cas/mount.rs`)**: seal_slot comment updated; test updated for BLP join semantics (assurance degrades to Classical — weakest link).

## Dependencies / open gaps

- **CAS clearance adapter**: #698's clearance primitives have landed, but the CAS plane still needs an authority-backed `CasClearanceSource` adapter. Until installed, `DenyAllClearanceSource` makes the explicitly active CAS PEP fail closed.
- **RPC activation boundary**: the canonical #1288 dispatcher remains dormant/pass-through (`Permit`) until a `MacDispatchPep` is installed. The CAS lane does not reinterpret absence as `NoPepInstalled`.
- **Lattice internment**: `domain_label()` uses `CompartmentSet::EMPTY` (level-axis enforcement only). Compartment isolation is structural (distinct physical storage roots per domain). Enriching the seal label with interned compartment bits requires threading the `Lattice` to the CAS layer.

## Validation

- `cargo metadata --all-features`: ✓
- `cargo check -p hyprstream --lib --tests`: ✓
- `cargo test -p hyprstream mac::cas_pep --lib`: ✓ 16 passed, 0 failed
- `cargo test --release -p hyprstream services::xet::tests::get_xorb --lib`: ✓ 2 passed, 0 failed
- Cross-package release nextest: pending final installation-seam candidate.
