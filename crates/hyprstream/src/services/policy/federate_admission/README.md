# Federate production identity and request authorization

The ATProto DID from Dex's signature-verified `federated_claims` connector
assertion is the Hyprstream principal and Policy subject. The issuer-signed
host credential keeps the opaque Dex `(issuer, sub)` as provenance; neither a
handle, email, browser-supplied DID, local username, nor hosted `did:web` is
used to establish the principal. Login alone grants no access.

Policy prepares a short-lived challenge from the DID's current tenant grants,
then repeats the Policy decision when the one-use challenge is redeemed. The
OAuth service signs a credential with DID `sub`, audience, expiry, correlation
`sid`, exact scope ceiling, Ed25519 `cnf.jwk`, ML-DSA-65 public key, ordered
hybrid-key thumbprint, signer-inventory digest, and verified OIDC provenance.
The complete challenge/signature binding proves possession of both proof
keys. Credential lifetime is capped at five minutes. `sid` is correlation only:
there is no session-row lookup or immediate per-session logout contract.

Registry and Model independently verify the host credential, the original
holder's COSE request proof, and its binding to the exact service, schema,
method/body, credential, and expiry. Before dispatch they ask authenticated
Policy to recheck the DID's current tenant/resource/action grant and signed
scope ceiling. A revoke or suspension therefore denies the next protected
request/tool call. Model-to-Inference remains inside the same logical Model
request and retains the original caller; no delegated bearer is created.

After Policy allows the action, the serving adapter admits the request ID into
the existing shared PostgreSQL replay store. It stores only a
domain-separated replay digest and expiry; the request body, bearer, and
credential are not stored. Database failure or replay denies dispatch. This
is the only PostgreSQL state required by the Federate Registry/Model path;
the v5 session/provenance schema, session URL, and profile bootstrap are not
runtime prerequisites.

The Policy factory installs the runtime from the trusted service enrollment
and a complete local signer inventory. It derives the signed inventory digest
from that inventory, so a credential from an old inventory fails the next
Policy action check. Missing enrollment/inventory or replay authority remains
fail-closed. Historical session-store migration code is not used by this
runtime path.

This source change still requires local focused tests, exact-head CI/review,
and live acceptance. None of those alone proves the full delivery objective:
the receipt must show Federate browser login from the staging website, an
authenticated Registry RPC, a pinned Model-to-Inference request, and denial
after a Policy grant is revoked. Worker validation is out of scope.
