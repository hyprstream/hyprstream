# CPU inference service deployment contract

This is the handoff from hyprstream #1236/#1247 to metal #1309. A deployment
runs exactly two independent `inference` service processes. Each process loads
one immutable model checkout on CPU, owns its libtorch engine and KV cache, and
publishes encrypted `generateStream` output on its own QUIC/WebTransport `/moq`
origin.

## Artifact and command

Use the multi-architecture CPU image and pin the release digest in metal:

```text
ghcr.io/hyprstream/hyprstream@sha256:<release-cpu-index-digest>
```

`latest-cpu` and `edge-cpu` are discovery tags, not deployment pins. The OCI
index selects native `linux/amd64` or `linux/arm64`. The container command is:

```text
/hyprstream service start inference --foreground --ipc
```

The image is distroless. Do not specify a shell health check or expect CUDA
tools in the container.

## Two required metal entries

Metal should model these as two named entries, not a scalar replica count:

| Metal entry | `REPLICA` | IPC socket | QUIC UDP port |
| --- | ---: | --- | ---: |
| `inference-cpu-0` | 0 | `inference-cpu-0.sock` | 7440 |
| `inference-cpu-1` | 1 | `inference-cpu-1.sock` | 7441 |

Both may mount the same read-only model checkout and service credential, but
they must be separate containers/processes with separate writable cache/state
directories and separate CPU/memory/cgroup budgets. They may share the
deployment IPC directory with Policy and Discovery; the replica-specific
socket names prevent bind collisions.

Required per-entry environment:

```text
HYPRSTREAM__INFERENCE__MODEL_PATH=/models/demo
HYPRSTREAM__INFERENCE__MODEL_REF=<authority-facing-model-ref>
HYPRSTREAM__INFERENCE__MODEL_OID=<40-or-64-hex-immutable-object-id>
HYPRSTREAM__INFERENCE__TENANT=<verified-tenant-domain>
HYPRSTREAM__INFERENCE__REPLICA=0
HYPRSTREAM__INFERENCE__STAGE_START=0
HYPRSTREAM__INFERENCE__QUIC_PORT=7440
HYPRSTREAM__INFERENCE__ADVERTISE_ADDR=<replica-0-routable-ip>:7440
HYPRSTREAM__QUIC__ENABLED=true
HYPRSTREAM__QUIC__BIND_ADDR=0.0.0.0:7440
HYPRSTREAM__RUNTIME__USE_GPU=false
```

For replica 1, set `REPLICA=1`, bind/advertise port `7441`, and advertise the
replica-1 routable address. `QUIC_PORT` must be non-zero and equal to the port
in `ADVERTISE_ADDR`; startup rejects ephemeral, loopback, unspecified, and
mismatched advertised reaches. The service itself clears `use_gpu`,
`gpu_device_id`, `devices`, and `gpu_layers`, even if inherited configuration
tries to enable a GPU.

Mounts and secrets:

- `/models/demo`: read-only Git worktree. Startup discovers its repository and
  requires `HEAD == MODEL_OID` before loading weights.
- deployment IPC directory: read-write, shared with Policy and Discovery.
- inference service credential and CA/public trust material: secret/credential
  mounts, never Terraform/OpenTofu state values.
- per-replica cache/state directory: read-write and not shared with the sibling.
- TLS key material: secret mount. The certificate/public pin may be deployment
  output; the private key must not be.

Metal should validate at plan time that entry names, replica ordinals, IPC
socket names, QUIC ports, and writable state paths are pairwise distinct.
Enforce CPU and memory limits with the container/cgroup controls. The existing
`runtime.cpu_threads`, `runtime.kv_cache_size_mb`, and
`runtime.max_concurrent_generations` fields are not consumed by the inference
engine and must not be treated as capacity enforcement.

## Network and browser contract

The authenticated inference RPC returns a signed `StreamInfo` containing:

- `broadcastPath`: the service-scoped MoQ broadcast;
- `announcedAt`: the explicitly configured, pinned QUIC `/moq` reach for that
  replica;
- the server DH public key and the declared stream QoS.

This backend branch supplies the authenticated `generateStream` RPC, signed
reach, and service-scoped `/moq` publisher. `hyprstream-rpc-std` already has a
raw WASM WebTransport/MoQ worker, but does not yet have the production
high-level inference subscriber/verifier. Www #13 must select the advertised
reach and broadcast path, then enforce sequence/QoS, chained-MAC, and AEAD
verification before exposing token bytes. No green browser E2E is claimed by
this backend handoff.

`moq_event` remains the lifecycle/event fan-out plane. Per-request generation
tokens use `moq_stream`; metal does not need to deploy a separate legacy
StreamService for them.

The `/moq` transport currently accepts anonymous subscribers. Generated token
payloads remain AEAD-sealed, but connection admission, tenant authorization,
and subscriber capacity protection are not enforced at the MoQ endpoint. Do
not expose the two replica ports directly to an untrusted network; place them
behind the deployment ingress policy until authenticated MoQ CONNECT lands.

## Replica routing

The service deliberately does not announce both processes under the singleton
`inference` Discovery key: current Discovery replaces duplicate
`(service-name, socket-kind)` endpoints and cannot represent a two-member pool.
Metal #1309 must place both named backends behind one canonical
network-addressable inference route and remove an unhealthy/draining member
there. Discovery multi-endpoint routing remains follow-up work; deploying two
processes alone does not create an in-process load-balancing pool.

The initial authenticated RPC may be load-balanced. Its returned
`StreamInfo.announcedAt` is a second-hop, replica-pinned address and must remain
sticky to the exact replica that created `broadcastPath`; never round-robin
that address to the sibling, whose isolated origin does not contain the
broadcast.

## Readiness, health, and termination

- Readiness is late: the factory verifies the immutable OID before launch; the
  service signals ready only after CPU model/weights load, RPC bind, and QUIC
  `/moq` bind.
- The authoritative health probe is authenticated inference `isReady` followed
  by `healthCheck`; success requires `modelLoaded=true` and `status="ok"`.
  Socket existence or a running container is only liveness, not readiness.
- Before SIGTERM, stop routing new RPCs to the replica and allow a best-effort
  completion window. Transport shutdown drains RPC/QUIC connections, but
  detached generation continuations are not counted, cancelled, or awaited;
  active generation completion is not guaranteed. Process exit is the unload
  boundary and drops the engine and KV cache. Configure `Restart=on-failure`.
- New stream admission during the external drain window is a metal/router
  concern until an explicit capacity/drain RPC lands.

## Fail-closed limitations

Partial transformer ranges `[a,b)` are rejected at startup. The real subset
loader and range-aware forward path are tracked by #314; a whole-model process
must not be labeled as a partial pipeline stage.

The backend publisher currently supports the classical ephemeral-DH
`generateStream` setup. A production browser subscriber, hybrid-KEM browser
support, authenticated MoQ admission, remote cancellation, enforced
per-process admission/KV budgets, Discovery multi-endpoint routing, and an
explicit generation-aware drain/capacity RPC are follow-up acceptance work.
