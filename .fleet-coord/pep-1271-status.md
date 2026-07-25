# PEP #1271 status — MAC per-object PEP for events + MoQ pub/sub

PR: https://github.com/hyprstream/hyprstream/pull/1296 (branch `feat/mac-moq-event-pep`). Package: `hyprstream-rpc`.

## Reconciled with the canonical #1288 contract

- `auth/mac/pep.rs` consumes `MacDecision`, `MacDenyReason`, and
  `RpcObjectLabelResolver` from `dispatch_pep.rs`; it does not define a
  parallel verdict, deny-reason, or resolver contract.
- `MoqEventPep` is the active event/MoQ PEP. An installed instance fails
  closed on missing verified-subject clearance, missing object labels, and
  lattice-floor denial.
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
  received objects are checked before passthrough/decrypt.
- `MacSubscribeAuthorizer` adapts the same PEP to the MoQ subscribe surface.
  The existing structural verified-tenant consumer scoping remains intact.

## Known external seam

`moq_net::Server` still has no per-subscribe callback (#276), so the transport
authorizer is available for installation and unit-tested but is not invoked
per-track by `moq_net` itself. Event-plane enforcement is active at the event
publisher/subscriber boundary.

## Review gate

kimi-k3 review is the final gate before merge. Do not self-merge.
