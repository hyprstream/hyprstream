# did:at9p accepted-state transaction boundary

Issue #1004 is owned by the daemon's sole read-write PDS process. That process
constructs one `At9pStateIngest` beside `PdsPublisher`, backed by the same
RocksDB instance and by a `DuplicityGuard::with_durable_alarm` using dedicated,
purpose-derived Ed25519 and ML-DSA-65 audit keys. The derivation starts from
the deployment's stable node/service root, so alarm verification survives both
process restart and unrelated OAuth signing-key rotation.

Each DID has a versioned accepted-state envelope and a distinct monotonic-head
checkpoint key. The envelope contains the head
kind and canonical head bytes together with the epoch, H512 head digest, and
terminal flag. A daemon-authenticated envelope also contains the purpose-derived
audit public key, a certificate for that key from the deployment's stable
node/service identity, and an audit-key signature over the complete state. The
deployment-signed checkpoint binds `(epoch, head digest, terminal)` to the H512
digest of that exact envelope.
Thus an internally valid attacker-selected fork cannot be substituted in
RocksDB and mistaken for the fork this daemon accepted. Genesis ingest first
runs the complete GATE pipeline and derives the seed exclusively from the
verified capsule. Successor ingest calls `admit_successor`; the guard reloads
and decodes that durable head and derives its own predecessor. Callers cannot
provide a seed or predecessor state.

The store advances envelope and checkpoint in one synchronous RocksDB
`WriteBatch`. Before the commit point a snapshot exposes the complete old pair;
after it, the complete new pair. Missing keys and every cross-generation,
signature, envelope-digest, epoch, head-digest, or terminal mismatch fail
closed. Thus replaying an older otherwise-valid envelope alone cannot make it
visible after the checkpoint advances.

Admission has no unconditional write API. The store commits only when its
verified durable head still equals the expected `(epoch, digest, terminal)`.
All production guards share one `Arc<PdsRecordStore>` owner and store-level CAS
lock; RocksDB excludes a second read-write owner for the directory. A loser
re-reads and is classified against the durable winner, never `Advanced`.

The ordinary registry RPC schema exposes authenticated
`ingestAt9pCandidate`. Network/PDS callers supply untrusted DID and record bytes
only; genesis authority comes solely from GATE and successor authority solely
from the verified durable predecessor. Consumers read the typed verified PDS
seam.

Duplicity alarms use the existing separately fsynced, hybrid-signed alarm WAL.
An alarm never advances accepted state. Restart eagerly reopens and verifies
the alarm WAL before ingest becomes available. Accepted-state values are
verified lazily but fail closed on every load, before they can anchor an ingest
or be returned to a consumer.

This boundary remembers only the state this node accepted. It does not claim
global-latest consensus or first-contact fork detection, does not treat service
entries as reach/liveness/admission authority, and never falls back to a caller,
configuration seed, or genesis keys after an accepted rotation.

This rollback guarantee protects the production store API and recovery
protocol. An attacker able to atomically roll back every trusted external/local
monotonic anchor, including both signed keys, requires an off-host anchor and is
outside this local-store guarantee.
