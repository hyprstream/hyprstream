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

## Extension hooks (separate issues — not implemented here)

- `observability.*` → **#784 (K1b)**: OTel Collector → Prometheus exposition.

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
