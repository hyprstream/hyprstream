# PEP #1268 Status

**Branch:** feat/mac-rpc-pep
**Issue:** #1268 (MAC activation-gate T3, epic #1267)
**PR:** #1288
**Status:** Revised per kimi-k3 review — ready for re-review

## kimi-k3 review fixes applied

1. **BLOCKING fail-open → fail-closed:** Replaced `if let Some(pep)` pass-through with `check_dispatch_mac()` (denies `NoPepInstalled` when no PEP is installed). No pass-through path exists in dispatch. Integration tests install `DormantMacPep` via `ensure_dormant_mac_pep()`. Global PEP changed from `OnceLock` → `RwLock` (swappable for tests). Removed misleading "dormant = pass-through" and "AVC generation counter" comments.
2. **Contract-doc sync:** Removed nonexistent `ClearanceProvenance` trait and `AuditFailClosed` variant. Noted `MacDispatchPep::check` takes `&EnvelopeContext` (RPC-only — other lanes reuse `MacDecision`/`MacDenyReason`/`RpcObjectLabelResolver` but NOT the PEP trait). Noted `method: Option<u16>` is browser-only.
3. **Streaming continuations:** Amended contract to state `StaleAuthority` is deferred (reserved variant, not used today). Filed follow-up #1291.

## Validation

`cargo nextest run --release --profile ci -p hyprstream-rpc` → 923 passed, 3 skipped, 0 failed.
