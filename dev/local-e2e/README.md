# Disposable causal validation harness

This directory is an **offline smoke and contract harness**, not an end-to-end
deployment. It does not install or start units, containers, daemons, cloud
resources, DNS, credentials, or trust roots. In particular, passing this
harness is not evidence that the ingest staging stack, an immutable image,
ACME, DNS, SELinux, signup, or authenticated inference works.

The old native `systemd --user` deployment graph was removed. The deployable
service graph belongs to ingest MR !100. This harness consumes that graph only
after ingest emits concrete rendered Quadlets and an independent exact-head
review records `PASS`.

## What is causal today

`tests/run.sh` creates a fresh owner-only task root and exercises:

- cleanup armed before the first run mutation;
- unique run IDs, XDG roots, unit projections, and held loopback ports;
- mode-0600, no-follow secret files and symlink rejection;
- cleanup that requires an exact regular-file ownership marker;
- exact rendered core graph validation for Event, Policy, Discovery, Registry,
  Streams, Model, OAI, and OAuth;
- the separate versioned two-replica inference contract;
- wrong-image, wrong-port, missing-service, dirty-source, and non-PASS-review
  negatives;
- exact transport/HTTP/RPC response classes and JSON schemas.

`run-offline-smoke.sh` additionally runs focused Rust tests against the current
checkout with `Cargo.lock` and `--locked` through the shared build helper. The
tests cover positive and negative trust-directory selection, symlink/writable
trust rejection, positive and wrong-pin TLS transport behavior, exact
service-auth/session response behavior, authenticated session success/replay
rejection, inference readiness semantics, and application output accounting.
They are in-process tests; no persistent daemon is started.

## Owned run wrapper

Run any offline command beneath disposable roots:

```bash
mkdir -m 0700 dev/local-e2e/runs/task-1383
python3 dev/local-e2e/causal_harness.py owned-run \
  --task-root dev/local-e2e/runs/task-1383 -- \
  env
```

The child receives `XDG_CONFIG_HOME`, `XDG_STATE_HOME`, `XDG_DATA_HOME`,
`XDG_CACHE_HOME`, `XDG_RUNTIME_DIR`, `HYPRSTREAM_CAUSAL_CONTEXT`, and a unique
unit prefix. The context is a mode-0600 JSON file. Ports are bound and held by
the parent for the child's lifetime, so another process cannot claim them.
Secrets must be written with the harness API or through protected files; they
must never be placed in arguments or environment variables.

## Ingest contract gate

The final adapter is deliberately fail-closed:

```bash
python3 dev/local-e2e/causal_harness.py verify-ingest \
  --source-root /path/to/clean/ingest-checkout \
  --expected-sha FULL_SHA \
  --review-record /path/to/first-independent-review.md \
  --review-record /path/to/second-independent-review.md \
  --render-dir /path/to/ingest-produced/rendered-quadlets \
  --inference-contract /path/to/ingest-produced/inference-contract.json
```

Requirements:

- the source checkout is clean and exactly at `FULL_SHA`;
- two distinct independent reviews each start with exactly `PASS FULL_SHA`;
- every path component and input is non-symlink;
- all eight concrete core Quadlets use one immutable digest and the exact
  ingest service commands/shared IPC volume;
- no template placeholder remains;
- the separate inference contract is version `v1`, pins the reviewed
  implementation/image evidence, and contains exactly replicas 0/1 with
  ports 7440/7441 and the reviewed authenticated readiness/health schemas.

Until ingest !100 has a reviewed final artifact, this command must fail and PR
#1383 is not merge-ready. Inference application output remains a separate
runtime E2E requirement: it needs an authenticated landed client and a
non-empty expected output. This smoke harness does not manufacture credentials
or relabel socket existence as E2E proof.

## Full E2E definition

The term E2E is reserved for a later authorized run that proves, in order:

1. every rendered ingest service is active and belongs to the unique run;
2. a wrong CA fails TLS and the intended CA succeeds;
3. a wrong deployment trust directory fails startup and every required trust
   leaf is consumed from the intended directory;
4. signup, DPoP-bound service auth, one-use session exchange, and whoami return
   their exact status/header/body contracts;
5. authenticated inference returns `isReady=true`, then
   `modelLoaded=true,status=ok`, and finally the expected application output.

No script in this directory currently claims that proof.
