# Deferred Federate recipient-binding source extension

This unpublished source-slice extension closes first-use recipient
substitution when the staging rendezvous is treated as blind and untrusted.
It does not alter `common.capnp` and is not a reservation or amendment to the
public/frozen v16 label registry.

Deferred Federate request proofs may carry private CWT claim `-70009`, whose
value is a closed three-entry map: key `1` is the unary response-recipient
commitment and key `2` is the optional hybrid-KEM stream-recipient commitment.
Each is either `null` when the matching envelope field is absent or
`SHA-256(RecipientPublic::encode())`. The stream KEM recipient remains
optional in this slice; a holder may instead select the existing legacy
DH-only stream path. That absence is signed, so a relay cannot add a KEM
recipient to a request whose claim contains `null`. Key `3` is the legacy
`clientDhPublic` value itself, an exact 32-byte string or `null`. The proof
producer must sign all three values along with the request's existing
credential, audience, schema, and exact-body commitments. Deferred Federate
dispatch requires the claim and checks all three fields against the
authenticated envelope before calling service admission.

Generic request proofs may omit `-70009`; their existing proof-v1 behavior is
unchanged. A deferred Federate client using this source extension must create
the response recipient and, when using hybrid streaming, the stream recipient
before building its proof, include the canonical commitments/nulls, and send
exactly those values in the envelope. Existing clients which omit the claim,
or use a relay-selected key, fail closed before handler entry. The current
transport still encrypts generic denial responses
to the envelope's response recipient; no successful handler result or
Inference stream key-release can occur after a mismatch.

Only the dedicated deferred-Federate proof parser accepts `-70009`; the
generic proof-v1 parser continues to reject it as an unknown claim, preserving
the frozen closed v16 claim set for all other paths.

The extension intentionally uses `-70009` only in this unpublished source
slice. Publishing it requires separate standards/registry review because the
current v16 proof claim set is frozen and closed.
