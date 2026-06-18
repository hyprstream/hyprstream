# Spike: policy_manager test failures — wildcard validation regression

**Branch:** `ewindisch/spike-policy-manager-tests` (off `ewindisch/stack-test-fixes`, moq epic tip)
**Date:** 2026-06-17
**Question:** Are the 7 failing `auth::policy_manager` tests valid issues or stale tests? Security-review the tests and their importance.

## Verdict

**Valid issue — a real, production-affecting regression**, not stale tests. The tests are correct and security-important; they caught a hardening change that broke the runtime authorization-mutation API.

## Root cause

`516b53581` (2026-06-16, *"fix(validation): … policy wildcard …"*, stack-only) added
`validate_policy_component` (`crates/hyprstream/src/auth/policy_manager.rs:89`), which rejects
`*` in **every** policy component:

```rust
if component.contains('*') {
    return Err(ValidationError("… contains wildcard '*' (not allowed in user-supplied policy subjects)"));
}
```

It is called by **all** mutators: `add_policy_with_domain`, `remove_policy_with_domain`,
`add_role_for_user`, `remove_role_for_user` (and `add_policy`/`remove_policy`, which delegate).

But the Casbin model is **built on wildcards** (`DEFAULT_MODEL_CONF`):
- `m = … (p.dom == "*" || r.dom == p.dom) && keyMatch(r.obj, p.obj) && (p.act == "*" || keyMatch(r.act, p.act))`
- Every documented rule is `p, alice, *, *, *, allow` / `p, alice, *, model:*, infer, allow`.
- `domain="*"` is the **standard global domain** used by every base rule and every production call.

So the wildcard ban contradicts the model it guards.

### Two concrete defects

1. **`add_policy()` can never succeed.** It hardcodes `domain="*"` then calls
   `add_policy_with_domain`, which validates the domain and rejects `*`. Any 3-arg
   `add_policy(user, resource, op)` returns `ValidationError` regardless of arguments
   (proven: `test_policy_manager_from_dir` fails on a wildcard-free call).
2. **`domain="*"` / `subject="*"` / `action="*"` are all rejected**, breaking legitimate
   runtime grants.

## Production impact (all silently swallowed via `let _ =`)

| Call site | What breaks |
|---|---|
| `services/policy.rs:1023-1031` `handle_set_branch_visibility` | **Make model public/private (#276)** — adds `("*","*",resource,…)`; both `*` rejected → model never becomes public. Error swallowed; `save()` persists nothing. |
| `services/oauth/oidc_callback.rs:566,587` | **OIDC-login self-ownership grant** — `(subject,"*",ns,"*","allow")` → rejected; a newly-authenticated user does not receive their namespace grant. |
| `cli/bootstrap_manager.rs:357,374` | Bootstrap per-user grants — `(username,"*",…)` → rejected. |
| `services/factories.rs:349` | TUI anonymous grant `("anonymous","*","tui:*","*",…)` → rejected, but **masked** by the identical static rule in `default_policy_csv` (`p, anonymous, *, tui:*, *, allow`). |

Production boots and mostly works only because bootstrap loads `policy.csv` and injects
`SERVICE_BASE_POLICIES` **directly into the enforcer**, bypassing the validator. The
**authorization decision path (`check`/`check_with_domain`) is unaffected** — it does not
validate, and wildcard matching works (`test_base_rules_injected_on_init` passes). Only
**runtime mutations** are broken, and every production caller ignores the error, so the
breakage is silent (an authz grant that silently no-ops).

## Security review of the tests

All 7 are **valid and important** — they encode the authz contract and must not regress:

| Test | Covers | Verdict |
|---|---|---|
| `test_policy_manager_from_dir` | deny-by-default; default files; grant-then-allow | keep |
| `test_add_and_check_policy` | remove permissive rule → deny; scoped grant; op isolation | keep |
| `test_role_based_access` | RBAC role inheritance; non-member denied | keep |
| `test_act_wildcard_keymatch` | action-namespace matching (`ttt.*` matches `ttt.writeback`, not `infer.*`) | keep |
| `test_set_branch_visibility_adds_removes_rules` | public/private model visibility (#276) | keep |
| `test_format_policy` | policy/role introspection | keep |
| `test_input_validation_accepts_valid_input` | validator must **accept** legitimate input incl. `model:qwen3-*` | keep |

The CSV-injection validations (null byte, newline/CR, comma, length cap) are **correct and
valuable** — their tests (`rejects_null_bytes`, `rejects_line_breaks`, `rejects_long_strings`)
pass and should stay.

## Assessment of the `*` ban as a security control

The `*` rejection is **mis-targeted**:
- It does not actually prevent privilege escalation — a caller able to reach `add_policy`
  could still grant broad non-`*` patterns (e.g., `keyMatch` prefixes).
- The correct control is **service-level authorization on the policy-mutation RPC** (scope
  *which* subjects/resources a caller may grant — partly present via `SERVICE_BASE_POLICIES`
  and the PolicyService's own authz), **not** a syntactic ban on `*` in policy content.
- It breaks core, intended features (public models, global-domain rules, RBAC resource
  wildcards) — i.e., it is both **incomplete as security** and **breaks functionality**.

## Recommended fix (not yet applied — security-sensitive, needs sign-off)

1. **Remove the `*` check** from `validate_policy_component`; keep null/newline/CR/comma/length
   (the genuine CSV-injection + DoS protections).
2. Enforce grant-scoping at the **PolicyService RPC layer** (who may grant what), where the
   caller identity is known — the right place for anti-escalation.
3. Re-run the 7 tests (expected: all pass) and add a regression test asserting
   `add_policy(user, resource, op)` succeeds (guards the `domain="*"` self-failure).

Until then the failures should **not** be dismissed as "pre-existing/ignore" — they mark a
live, silent authorization-mutation outage (most visibly: **public/private models and OIDC
self-grant do not work at runtime**).
