# did:at9p accepted-state transaction boundary

Issue #1004 is owned by the daemon's sole read-write PDS process. That process
constructs one `At9pStateIngest` beside `PdsPublisher`, backed by the same
RocksDB instance and by a `DuplicityGuard::with_durable_alarm` using dedicated,
purpose-derived Ed25519 and ML-DSA-65 audit keys. The derivation starts from
the deployment's stable node/service root, so alarm verification survives both
process restart and unrelated OAuth signing-key rotation.

Each DID has one versioned accepted-state value. The value contains the head
kind and canonical head bytes together with the epoch, H512 head digest, and
terminal flag. A daemon-authenticated envelope also contains the purpose-derived
audit public key, a certificate for that key from the deployment's stable
node/service identity, and an audit-key signature over the complete state.
Thus an internally valid attacker-selected fork cannot be substituted in
RocksDB and mistaken for the fork this daemon accepted. Genesis ingest first
runs the complete GATE pipeline and derives the seed exclusively from the
verified capsule. Successor ingest calls `admit_successor`; the guard reloads
and decodes that durable head and derives its own predecessor. Callers cannot
provide a seed or predecessor state.

The store replaces the complete value with one synchronous RocksDB write. A
RocksDB write is atomic, its WAL is enabled, and `sync=true` makes successful
ingest crash-durable before it is returned. Consequently recovery sees either
the prior complete value or the replacement complete value. There are no
separate body and watermark keys, so a new watermark with an old body (or the
reverse) is not representable. Reads decode the head, recompute its digest and
derived epoch/terminal fields, verify the deployment certificate and daemon
acceptance signature, and reject any mismatch before returning the typed
`AcceptedAt9pState`.

Duplicity alarms use the existing separately fsynced, hybrid-signed alarm WAL.
An alarm never advances accepted state. Restart eagerly reopens and verifies
the alarm WAL before ingest becomes available. Accepted-state values are
verified lazily but fail closed on every load, before they can anchor an ingest
or be returned to a consumer.

This boundary remembers only the state this node accepted. It does not claim
global-latest consensus or first-contact fork detection, does not treat service
entries as reach/liveness/admission authority, and never falls back to a caller,
configuration seed, or genesis keys after an accepted rotation.
