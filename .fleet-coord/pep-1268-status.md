# PEP #1268 Status

**Branch:** feat/mac-rpc-pep
**Issue:** #1268 (MAC activation-gate T3, epic #1267)
**PR:** #1288
**Status:** Revised per kimi-k3 review — ready for re-review

## kimi-k3 review fixes applied

1. **Installed PEP fail-closed:** `check_dispatch_mac()` is the single dispatch seam. The uninstalled activation state is dormant and preserves pre-PEP RPC behavior; once installed, the PEP is mandatory and missing clearance, missing labels, and lattice-floor denials fail closed. The temporary per-test `DormantMacPep` workaround was removed after it exposed the `hyprstream` metrics regression.
2. **Contract-doc sync:** Removed nonexistent `ClearanceProvenance` trait and `AuditFailClosed` variant. Noted `MacDispatchPep::check` takes `&EnvelopeContext` (RPC-only — other lanes reuse `MacDecision`/`MacDenyReason`/`RpcObjectLabelResolver` but NOT the PEP trait). Noted `method: Option<u16>` is browser-only.
3. **Streaming continuations:** Amended contract to state `StaleAuthority` is deferred (reserved variant, not used today). Filed follow-up #1291.

## Validation

`cargo nextest run --release --profile ci -p hyprstream-rpc` → 923 passed, 3 skipped, 0 failed.
