# hyprstream Helm chart

Deploys hyprstream to Kubernetes as **one Deployment + one ClusterIP Service per
service** ("service-per-Deployment"). Services address each other by Kubernetes
DNS Service name over the networked transport (QUIC over UDP / iroh) — never over
shared-volume Unix-domain sockets. This is the K1 deployment substrate (epic
#778, issue #779).

## Quick start

```bash
helm install hyprstream charts/hyprstream --namespace hyprstream --create-namespace
kubectl -n hyprstream port-forward svc/hyprstream-oai 6789:6789
curl http://localhost:6789/v1/models
```

## Service set

The chart models the real inventory-registered factories
(`crates/hyprstream/src/services/factories.rs`). Map keys are the exact service
names passed to `hyprstream service start <name> --foreground`.

| Service | Kind | Default | Notes |
|---------|------|---------|-------|
| policy | rpc (QUIC/UDP) | on | Casbin policy + CA |
| discovery | rpc | on | endpoint resolution |
| registry | rpc | on | git2db model registry |
| model | rpc | on | inference; `variant: cpu\|cuda\|rocm` |
| event | moq | on | group-keyed event bus |
| streams | moq | on | token stream origin |
| oai | http (TCP) | on | OpenAI-compat API, `/health` probe |
| oauth | http+quic | off | OAuth 2.1 |
| mcp | http+quic | off | Model Context Protocol |
| xet | http | off | HF-XET CAS face |
| flight | grpc | off | Arrow Flight SQL |
| metrics | rpc | off | DuckDB metrics |
| tui | rpc | off | TUI display server |
| worker | rpc | off | sandbox backends — **needs elevated privilege** |

Default-enabled set matches the #779 acceptance path plus transitive
dependencies (discovery).

## Configuration

Values are documented inline in `values.yaml`. Highlights:

- `image.repository` / `image.tag` — container image (see assumption below).
- `services.<name>.enabled` — per-service toggle.
- `services.model.variant` — `cpu` (default) / `cuda` / `rocm`; appends the image
  suffix and is where you add `resources.limits.nvidia.com/gpu` +
  `nodeSelector` for GPU nodes.
- `persistence.*` — the shared `/var/lib/hyprstream` data volume (RWO by default;
  set `accessModes: [ReadWriteMany]` for multi-node).
- `config.*` — shared `HYPRSTREAM__*` env (rendered into a ConfigMap).
- `config.tls.enabled` — false by default so in-cluster HTTP probes and the
  port-forward quick start use plain HTTP on the oai Service port.

## Key/trust bootstrap

Set `keyBootstrap.enabled=true` to mount the flat credential directory expected
by `HYPRSTREAM__SECRETS__PATH`. The chart projects externally generated Secrets
instead of minting identity material in Helm templates.

Required Secret layout:

- Shared trust Secret: `ca-pubkey`, `bootstrap-pubkeys`.
- Non-policy service credential Secrets: `signing-key`, `service-jwt`.
- Policy service credential Secret: `signing-key`.
- Policy CA Secret: `ca-key`.

Omitted per-service Secret names default to
`<release>-hyprstream-<service>-credentials`. The shared trust Secret defaults to
`<release>-hyprstream-trust`, and the policy CA Secret defaults to
`<release>-hyprstream-policy-ca`. Use `keyBootstrap.trust.existingSecret`,
`keyBootstrap.serviceSecrets`, and `keyBootstrap.policyCaSecret` to bind
Secrets created by the bootstrap/wizard flow or an external-secrets controller.

`keyBootstrap.issuerUrl`, when set, is exposed to every service as
`HYPRSTREAM__OAUTH__ISSUER_URL` and must match the issuer stamped into the
pre-generated service JWTs.

## Observability (K1b, #784)

`observability.enabled=true` (default off) deploys a standalone **OpenTelemetry
Collector** (Deployment + ConfigMap + Service) and wires every service to push
OTLP traces to it (`HYPRSTREAM_OTEL_ENABLE=true`, `OTEL_EXPORTER_OTLP_ENDPOINT`,
and per-Deployment `OTEL_SERVICE_NAME`).

hyprstream's `otel` feature is OTLP-**push**, **traces only** — it has no native
`/metrics` surface (an explicit non-goal of #784). The collector closes that gap:
its `spanmetrics` connector derives RED metrics (request rate / errors / latency)
from the incoming spans and its `prometheus` exporter exposes them for scraping.

```bash
helm install hyprstream charts/hyprstream --set observability.enabled=true \
  --namespace hyprstream --create-namespace
kubectl -n hyprstream port-forward svc/hyprstream-otel-collector 8889:8889
curl http://localhost:8889/metrics    # hyprstream_traces_span_metrics_calls_total, ..._duration_milliseconds_*
```

The `spanmetrics` connector emits `traces.span.metrics.calls` (counter) and
`..duration` (histogram); the prometheus exporter's `namespace` prefixes them,
so the scrapeable series are `hyprstream_traces_span_metrics_calls_total` and
`hyprstream_traces_span_metrics_duration_milliseconds_{bucket,count,sum}`,
labelled by `service_name` plus the configured `dimensions` (route/method/status).

| Value | Default | Purpose |
|-------|---------|---------|
| `observability.enabled` | `false` | Master toggle (env wiring + collector). |
| `observability.otelExporterOtlpEndpoint` | `""` | Override; empty ⇒ the in-cluster collector Service. |
| `observability.collector.enabled` | `true` | Deploy the bundled collector (false ⇒ env-only, point at an external one). |
| `observability.collector.image.*` | contrib `0.128.0` | Contrib distro (needed for `spanmetrics`). |
| `observability.collector.ports.prometheus` | `8889` | Prometheus exposition port (`/metrics`). |
| `observability.collector.prometheusNamespace` | `hyprstream` | Metric name prefix. |
| `observability.collector.dimensions` | route/method/status | Span attrs promoted to metric labels. |
| `observability.collector.serviceMonitor.enabled` | `false` | Emit a kube-prometheus-stack `ServiceMonitor` (needs the Prometheus Operator CRD). |

**Autoscaling signal (K6a, #792).** The `/metrics` endpoint is the scaling signal
source: point `prometheus-adapter` or the KEDA prometheus scaler at it to drive an
HPA on e.g. `hyprstream_traces_span_metrics_calls_total` request rate. Which spans hyprstream emits
today (and which scaling signals — stream backpressure/queue depth, tokens/s —
still need instrumentation) are tracked as follow-ups under #792; this chart only
delivers the exposition surface.

## CSI node plugin

Set `csi.enabled=true` to install the `csi.hyprstream.io` node plugin for K3b
(#790). The driver supports the operator-selected dual mounter contract:

- `mounter=kernel` uses kernel v9fs. Mount tickets are presented through the
  normal `uname=` attach field and a node-local stream bridge terminates the
  selected carrier before handing the kernel a connected `trans=fd` socket.
- `mounter=fuse` uses the hypr9p FUSE->9P client and dials the selected carrier
  directly.

The transport is a dial-time carrier, not a storage contract. Phase 1 requires
operators to set `csi.transport.endpoint` to a node-local 9P listener/bridge
that the FUSE client can dial, such as `vsock://<cid>:564`, `unix:///path`, or
`tcp://host:port`. `webtransport` / iroh-QUIC is the normal cross-node target
once the node bridge/dialer lands; UDS is only a co-located DaemonSet corner
case.

By default the chart renders only the FUSE StorageClass
`hyprstream-9p-fuse`. The kernel v9fs StorageClass is opt-in
(`csi.storageClasses.kernel.enabled=true`) until the node-local `trans=fd`
bridge binary ships.

## Image assumption

No published image reference exists in-repo yet — only a local multi-variant
`Dockerfile` (distroless CPU/CUDA/ROCm stages) that tags nothing. `image.repository`
defaults to a **placeholder** (`ghcr.io/hyprstream/hyprstream`) and `image.tag`
defaults to `Chart.appVersion`. Override for your registry, or wire CI to publish
there.

## Known gap (Phase 0)

Cross-pod QUIC service-to-service **peer resolution** — feeding K8s DNS names into
the RPC resolver / discovery lookup so `for_service("policy")` dials
`hyprstream-policy` — is not yet wired in hyprstream itself: the current
multi-process model resolves peers to UDS paths in a shared runtime dir (same-host
only), and QUIC is a WebTransport/announce plane with no static peer-address
config. This chart delivers the deployment substrate (Phase 0); a working
end-to-end install additionally needs that app-side wiring. The chart itself
contains **no shared-volume IPC**, per the #778 non-goal.
