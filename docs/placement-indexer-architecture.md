# Placement Record Indexer / Discovery AppView

This document records the design settled by #564 for the P1 placement directory
implemented by #524. It separates durable placement capabilities from volatile
load, defines the trust and freshness boundaries, and records the EventService
AppView shape that supersedes the original proposed `moq_event::BackfillSource`
widening.

## Scope and outcome

The placement directory is an in-process index owned by `DiscoveryService`.
`queryCandidates` uses it to discover a bounded set of candidate node DIDs;
it is not a global scheduler or cross-cell directory (that work remains #522).

P1 materializes records in these collections:

- `ai.hyprstream.placement.node`: a node's labels, declared capacities, and
  group-consent URIs.
- `ai.hyprstream.placement.group`: group metadata owned by the publishing DID.
- `ai.hyprstream.placement.groupItem`: the group owner's membership claim.
- `ai.hyprstream.placement.workload`: decoded as part of a repository snapshot,
  but not used to filter P1 candidates.

The candidate identity is a `Did`; there is no separate `nodeId`. A selected
candidate includes the node record's `at://` URI so a caller that needs a proof
can retrieve the signed source record with `getRecord`.

## Day-one ingestion: authenticated repository snapshots

Day one deliberately does **not** use `moq_event::BackfillSource` or introduce
a placement-specific firehose consumer. `BackfillSource` is a legacy,
OID-scoped event API; widening it to `{ Oid | Did }` would preserve the wrong
key model. The existing P1 implementation instead polls the injected
`RecordResolver` for a bootstrap set of node DIDs:

```text
bootstrap DID (discovery announce; final admission policy at implementation review)
  -> RecordResolver::resolve_repo(did)
  -> CARv1: commit + complete MST + record blocks
  -> validate, decode placement collections
  -> replace that DID's snapshot in PlacementIndex
  -> rebuild derived directory maps
```

A refresh replaces a DID's complete snapshot, never merges records indefinitely.
Thus removal of a source record removes its derived labels and membership claims
at the next successful poll. A missing repo likewise clears the older snapshot;
a resolver, CAR, signature, or decoding failure clears it and does not admit
partially decoded facts.

### Ingest trust boundary

The resolver is a source adapter, not the integrity root. Before any record can
enter the index, the ingestion path must:

1. Require exactly one CAR root and locate the commit block.
2. Recompute and compare the CID of every CAR block, including the commit,
   MST nodes, and records.
3. Require `commit.did == did` to prevent a valid repo from one DID being
   replayed as another.
4. When `RecordResolver::resolve_verifying_key` supplies the DID's `#atproto`
   P-256 key, verify the commit signature before materializing facts.
5. Walk the MST with a visited-set and bounded visit budget, then decode only
   known placement record schemas.

The CID check matters even when the commit signature verifies: the commit signs
the root CID, not arbitrary bytes handed to the index under a declared CID.
Without content binding, substituted MST or record blocks could forge group
membership.

`Ok(None)` from `resolve_verifying_key` retains the day-one trusted-resolver
posture: structural and CID validation are still mandatory, but authentication
of the remote DID is delegated to the in-process resolver boundary. Resolving
foreign DID-document `#atproto` methods is a federation-hardening follow-up.
Until then, placement data is scheduling metadata, while the requester's
per-candidate policy authorization remains the hard query boundary.

## Materialized state and query path

`PlacementIndex` retains a raw snapshot per source DID and rebuilds the derived
maps after each refresh:

```text
DID -> { node facts, owned groups, group-item claims, workload count }
DID -> node { record URI, labels, declared resources, consented group URIs }
group at-URI -> group metadata
group at-URI -> claimed member DIDs
```

This representation makes delete/replace semantics simple and avoids retaining
stale materialized facts. The current P1 fleet size is small enough that selector
evaluation scans the known, live node set. A future high-cardinality AppView may
add an inverted `label key/value -> DID set` index, but must preserve the same
selector semantics for `In`, `NotIn`, `Exists`, `DoesNotExist`, `Gt`, and `Lt`.
It must not hard-code CPU, GPU, or memory names: labels and resource names are
generic key/value data.

A group is effective only with **bidirectional consent**:

```text
group owner publishes GroupItemRecord(group, subject = node DID)
AND
node's NodeRecord.groups contains that exact group at-URI
```

The index exposes each effective membership as synthetic label
`group/<group-at-uri>=true`. Using a group-specific label key is intentional:
the label selector model has one value per key, so multiple `group=<uri>`
entries would cause ambiguous or silently incomplete matching.

`queryCandidates` performs these steps in order:

1. Start with DIDs that have a durable indexed `NodeRecord` **and** an unexpired
   liveness entry.
2. Evaluate label selectors, including synthetic group labels.
3. Evaluate generic resource requests against the liveness report's
   `allocatable` resources, not the durable declared capacity.
4. Perform a fail-closed authorization check for each surviving
   `placement:candidate:<did>` resource.
5. Rank the authorized candidates by lower load and stable DID tie-break,
   then apply the caller/default result bound. `total_matching` is computed
   before that bound so truncation remains observable.

Ranking, affinity, and stickiness are deliberately outside P1.

## Freshness model

Durable capability facts and live scheduling facts have distinct freshness
contracts:

| Layer | Source | Freshness | Failure behavior |
| --- | --- | --- | --- |
| Labels, declared capacity, group records | polled signed repository snapshot | bounded by poll interval / later event lag | old facts are replaced or cleared on refresh failure; they cannot make an unlived node eligible |
| Allocatable capacity and load | `reportNodeLiveness` | real time within TTL | `TtlCache` expiry hard-excludes the node from candidates |

The hard exclusion is the ratified default: a node without a fresh liveness
advertisement is unavailable, rather than returned with a stale marker. A
soft/stale result mode, if ever added, must be opt-in and must not silently
become the scheduling default.

`reportNodeLiveness` may trigger first-seen ingestion for a node, but reporting
liveness does not itself establish durable placement capabilities. The two
layers remain independent by design.

## Memory and operational bounds

The P1 index is process-local and bounded by the cell's admitted/bootstrap DID
set, not the full federation. Operators must bound the bootstrap input (the
recommended source is discovery announcements), repository/CAR read size,
maximum records and labels per DID, MST visit budget, polling concurrency, and
query result count. P1 already bounds the returned candidate set; an AppView
implementation must retain that bound and reject/skip oversized source
snapshots rather than allowing a peer-controlled record graph to consume
unbounded DiscoveryService memory.

Raw snapshots are needed for atomic replace semantics. Derived maps are rebuilt
from those snapshots after an ingest, keeping read-side queries lock-bounded and
preventing deletion leaks. Metrics should expose poll age/failure, indexed DID
count, decoded record counts, rejected snapshots, liveness-cache size/expiry,
and query truncation.

## Day-two EventService AppView

The live tail uses the public-profile EventService firehose defined by EV3
(#603). Its canonical subject is:

```text
CanonicalSubject { did, nsid, rkey }
```

The AppView consumes only placement NSIDs for admitted DIDs. Its cold-start/live
seam is:

```text
record-store snapshot (RecordResolver -> CAR/MST validation)
  -> establish EventService firehose cursor
  -> apply commits whose (DID, NSID) match placement collections
  -> atomically publish an updated per-DID snapshot
```

Cursor handoff must avoid a gap between the snapshot and tail: subscribe or
capture a high-water cursor first, read the snapshot, then replay from that
cursor before accepting the live edge. Each event still passes the same schema,
content-binding, DID/authority, and replace/delete validation as a polled
snapshot; firehose delivery is not an authority bypass.

This replaces, rather than widens, the old OID-keyed `BackfillSource`. The
read-snapshot-then-tail concept in `BackfillMode` remains useful, but the
bespoke OID naming is retired in favor of canonical record subjects. Longer
term, the snapshot read converges with #705's subject-authorized VFS/CAS record
store; the EventService firehose remains the live event plane.

## Non-goals and follow-ups

- Cross-cell/global directory fan-out, affinity, and stickiness: #522.
- Firehose cursoring and EventService AppView implementation: day two after the
  EV3 subject model; not a `BackfillSource` API change.
- Foreign DID `#atproto` key resolution: federation-hardening follow-up.
- A soft stale-liveness policy: only as an explicit future configuration.

This design answers #564 while preserving the P1 implementation boundary set
by #524: local, bounded, authenticated repository materialization plus a
separate, TTL-bounded volatile load layer.
