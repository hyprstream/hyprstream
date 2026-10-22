# Read-only accepted service roster

`hyprstream pds inspect-services --service event,discovery,policy,registry,streams,model,oai,oauth`

This OS-owned local inspection command reads existing accepted service state and
emits one complete `hyprstream/verified-service-roster@1` JSON document to stdout.
It uses the same public schema as `pds provision-services --roster-export`, with
service name, DID, epoch, expiry and accepted head digest. All requested members
must verify before any document bytes are emitted. Errors return nonzero without
a document; diagnostics go to stderr. A stdout I/O error can still interrupt a
write, so consumers must require successful exit and a complete valid document.

The existing deployment trust loader authenticates the OS-owned hybrid deployment
CA, authority log/checkpoint and current Registry credential. The accepted-state
store opens read-only at the existing deployment path (`models/.registry/pds-store`
beneath the deployment data directory; `/var/lib/hyprstream/models/.registry/pds-store`
with `XDG_DATA_HOME=/var/lib`). Missing or invalid stores fail without initialization.
The command never opens the writer, ingest WAL, provisioner, renewal, genesis,
checkpoint advance or key-generation paths.

Retained service signing keys are still required to reuse the existing hybrid
service-key binding validator. The selected config `[secrets].path` and existing
`HYPRSTREAM__SECRETS__PATH` override resolve that directory. As with the offline
provisioner, it uses the shared-directory profile: Policy's canonical flat key,
other services' `SERVICE/signing-key`. It does not generate missing keys, sign
successors or print secret material. Existing trusted-artifact path and ownership
requirements, Registry JWT validity and accepted-state freshness remain mandatory.

Missing, ambiguous, invalid, expired or genesis-only roster members fail the whole
read. Checkpoint/signature failures are never interpreted as absent state or first
boot. A fresh `generated_at` records inspection time, not renewal; unchanged
accepted epochs, digests and expiries remain unchanged in the output.

The early dispatch precedes logging initialization, local endpoint initialization,
service bootstrap and ordinary CLI signing-key/resolver startup. No output-file
option exists: capture stdout using a caller-owned temporary file and publish it
only after successful exit. Redirecting directly over an old good file truncates
it in the shell before verification and is not failure-preserving publication.

This is verified local readback, not a remote trust root, caller credential,
client-admission mechanism or service-readiness probe. It can inspect beside a
live writer through existing read-only RocksDB handles, but it does not lock out
concurrent renewal or promise an all-roster transaction snapshot. Maintenance must
still own its existing exclusion and compare accepted heads/epochs/expiry before
mutation; live-service readiness still requires authenticated network probes.
