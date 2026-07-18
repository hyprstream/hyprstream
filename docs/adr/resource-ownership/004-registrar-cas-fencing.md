# ADR RO-004: Registrar CAS and fencing

**Status:** Accepted for review (#1067)

## Decision

The registrar is the only lifecycle writer. Every transition supplies the expected predecessor manifest/version. Finalization performs compare-and-swap and increments a registrar-allocated, per-resource monotonic fencing generation within a registrar term; the term changes on leadership/authority epoch, so a deposed primary's fence is distinguishable from the current primary's. A stale fence or predecessor loses even if both attestations are otherwise valid. The fence is threaded to the effectors: materialization staging/sealing and namespace publish/withdraw carry the fencing token, and effectors MUST reject effects whose `(term, generation)` is stale relative to the newest durably applied state.

## Consequences

At most one successor advances a predecessor. High availability requires a registrar backend capable of linearizable per-resource CAS; an external coordinator does not become title authority. A stale primary surviving failover cannot keep projecting or staging writes that downstream gates accept. The loser voids an unposted reservation or records a compensating transfer after post. Split-brain cannot be hidden by PDS last-write-wins behavior.
