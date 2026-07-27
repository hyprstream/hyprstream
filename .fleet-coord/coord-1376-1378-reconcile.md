# Coordination: #1376 (pglite backend) ↔ #1378 (wiring)

## Status: RECONCILED (2026-07-27)

#1378 (feat/1378-pglite-wiring, f453dcd82) modified `pglite_store.rs` to fix
the SQL from nonexistent `profile` column → real schema columns. #1376's final
backend (feat/1376-pglite-userstore) is a **complete superset** of that fix:

- Same real-column SQL (no `profile` blob column)
- Same `resolve_or_bind_external_idp` atomic implementation
- PLUS: complete hosted provisioning/activation (#1370 R4 hardening),
  all pubkey operations, hybrid PQ validation, external-identity reads

## Rebase instruction for #1378

When #1378 rebases onto the merged #1376:
1. **DROP** #1378's `pglite_store.rs` diff entirely — #1376 supersedes it
2. **KEEP** all wiring changes: `oauth/mod.rs`, `config/mod.rs`,
   `oidc_callback.rs`, `scim.rs`, `token.rs`, `userinfo.rs`, etc.
3. The interface #1378 depends on is unchanged:
   - `PgliteUserStore::open(path)` → standalone factory ✓
   - `PgliteUserStore::from_database(Arc<PGlite>)` → shared #1351 handle ✓
   - `PgliteUserStore` implements full `UserStore` trait ✓

## Open wiring concern (#1378's scope, not #1376's)

#1378 currently opens PGlite at `credentials_dir/pglite/` via `open()` rather
than sharing the #1351 AppView handle via `from_database()`. The charter says
"share the substrate; do NOT stand up a second identity DB." #1378 should
switch to `from_database(shared_handle)` when it rebases. #1376 provides both
constructors; the choice is #1378's wiring decision.
