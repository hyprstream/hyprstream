# PEP #1271 status — MAC per-object PEP for events + MoQ pub/sub

PR: https://github.com/hyprstream/hyprstream/pull/1296 (branch `feat/mac-moq-event-pep`). Package: `hyprstream-rpc`.

## Reconciled with the canonical #1288 contract

- `auth/mac/pep.rs` consumes `MacDecision`, `MacDenyReason`, and
  `RpcObjectLabelResolver` from `dispatch_pep.rs`; it does not define a
  parallel verdict, deny-reason, or resolver contract.
- `MoqEventPep` is the active event/MoQ PEP. An installed instance fails
  closed on missing verified-subject clearance, missing object labels, and
  lattice-floor denial. Construction requires an audit sink; the parent
  `MoqAuditSinkAdapter` writes denials into the canonical signed MAC WAL.
- Dormant event/MoQ enforcement remains pass-through. Public event publishers
  and subscribers retain the pre-activation `AllowAllEventAuthz`, and MoQ
  transports retain `MoqAuthzConfig::default()` with no installed authorizer.
  Installing `MacEventAuthz` or `MacSubscribeAuthorizer` activates fail-closed
  enforcement. No deny-on-uninstalled sentinel is used.

## Event and MoQ enforcement

- `MacEventAuthz` applies the label ceiling to public and confidential
  publishing, subscribing, and encrypted join/decrypt.
- Confidential object lookups are tenant-qualified using main's verified
  tenant binding; public/node-global event sources retain their global prefix.
- `EventSubscriber` carries an injected authz and independently verified
  subject. Key release is checked before the HyKEM epoch grant is opened, and
  received objects are checked before passthrough/decrypt. `try_recv` returns
  an explicit error for a subscribe denial; it never reports that denial as an
  empty queue.
- `MacSubscribeAuthorizer` adapts the same PEP to the MoQ authorization
  contract. The existing structural verified-tenant consumer scoping remains
  intact.

## #276 transport dependency and current fail-closed gate

`moq_net::Server` still has no per-subscribe callback
([#276](https://github.com/hyprstream/hyprstream/issues/276)). Consequently
this PR does **not** claim live per-track policy decisions at the transport
surface. Instead, both iroh and quinn reject the entire MoQ session whenever a
track authorizer is installed; the MAC adapter records that denial through the
same WAL path. Dormant transports with no installed authorizer retain their
pre-activation behavior. Once #276 supplies the callback, the coarse admission
gate can call `authorize(peer, track)` per track.

Event-plane publishing, subscribing, and join/decrypt are independently
mediated when `MacEventAuthz` is installed.

## Review gate

kimi-k3 review is the final gate before merge. Do not self-merge.
