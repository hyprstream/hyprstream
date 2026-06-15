# hyprstream-workers

Isolated workload execution using Kata Container VMs with CRI-aligned APIs.

## What it does

Two services:

**`WorkerService`** — CRI-aligned container runtime (mirrors the Kubernetes CRI interface):
- `PodSandbox` = Kata VM (hardware-isolated via KVM)
- `Container` = OCI container inside the VM
- `ImageClient` backed by Nydus RAFS for chunk-level image deduplication
- Exposes `RuntimeClient` and `ImageClient` traits over Cap'n Proto RPC

**`WorkflowService`** — GitHub Actions-compatible CI/CD orchestration:
- Discovers `.github/workflows/*.yml` from `RegistryService` repos
- Subscribes to the event bus (`git2db.{repo_id}.push`, `training.{model_id}.completed`, `metrics.{model_id}.breach`)
- Spawns pods/containers via `WorkerService` on trigger

## Architecture position

```
hyprstream-rpc           (transport + envelope)
hyprstream-vfs           (VFS for container I/O)
    ↑
hyprstream-workers       ← you are here
    ↑
hyprstream               (WorkerService factory, WorkflowService factory)
```

## Key types

| Type | Description |
|------|-------------|
| `WorkerService` | Combined `RuntimeClient` + `ImageClient` implementation |
| `WorkflowService` | Workflow discovery + trigger + execution orchestration |
| `PodSandboxConfig` | CRI pod sandbox configuration |
| `ContainerConfig` | OCI container spec |

## Feature flags

- `dbus` — D-Bus integration for host notification / systemd interaction
