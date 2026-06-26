# Spike: CRI-client model — can the VFS/DAX plane go *through* a CRI runtime?

**Date:** 2026-06-26
**Status:** spike / decision input
**Related:** #495 (vfsd + DAX), PR #496, #346 (podman/OCI backend), #341 (Worker GA epic), #391 (9P↔RPC), #409 (Wanix mount)

## The question

hyprstream's worker plane could act as a **CRI *client*** (the kubelet/Kubernetes
role), driving an existing CRI runtime service (containerd's CRI plugin, or CRI-O)
instead of building/embedding its own sandbox runtime. The decisive unknown:

> **Can hyprstream's differentiated VFS/DAX model-plane be delivered to a guest
> *through* a generic CRI runtime — or does that injection force the embedded
> runtime path?**

The "VFS/DAX model-plane" = the live, RPC-backed synthetic filesystem hyprstream
mounts into the sandbox: the model-service synthetic tree, registry worktrees, the
per-`Subject`-isolated namespace, and crucially **DAX zero-copy weight loading**
(mmap host weights straight into the guest's DAX window — the reason `hyprstream-vfsd`
+ DAX exist, see #495/#496).

## 1. CRI mount-model limits (evidence)

Our capnp mirror of CRI v1 (`crates/hyprstream-workers/schema/worker.capnp`) shows
the mount surface exactly as the upstream proto defines it:

```capnp
struct Mount {
  containerPath @0 :Text;
  hostPath      @1 :Text;     # <-- host path ONLY
  readonly      @2 :Bool;
  selinuxRelabel@3 :Bool;
  propagation   @4 :MountPropagation;   # private | hostToContainer | bidirectional
}
struct ContainerConfig {
  ...
  mounts       @6 :List(Mount);
  devices      @7 :List(Device);          # also hostPath-based
  annotations  @9 :List(KeyValue);        # <-- the only open extension channel
  ...
}
```

`PodSandboxConfig` likewise carries `annotations @6`. There is **no** in-spec field
for "mount a live virtiofs/9P FD-backed namespace" — CRI `Mount` is strictly
`hostPath → containerPath`. The newer CRI *image-volume* / OCI `VolumeSource` (KEP
for mounting OCI artifacts as volumes) is also image-addressed, not a live RPC
namespace, and isn't in our mirrored schema.

**Conclusion (1):** The only way to get our VFS into a guest via CRI is to first
**materialize it as a host path** (a host mountpoint) and then express that host
path as a CRI/OCI `Mount`. CRI cannot carry the synthetic namespace itself.

**This is the pivotal finding:** even in the CRI-client model, the VFS must be
materialized on the host by **vfsd** (FUSE/virtio-fs, #495/#496). CRI-client does
**not** eliminate vfsd — it changes *what drives the container lifecycle*, while
vfsd remains the host-side provider that produces the mount CRI then injects.
So #496 is **not at risk** from this decision; it is load-bearing in both models.

## 2. Injection options assessed

| Option | Mechanism | Can inject our mount? | Verdict |
|---|---|---|---|
| (a) **In-spec CRI `Mount`** | `ContainerConfig.mounts[]` host-path bind | Yes, *if* vfsd pre-mounts the VFS at a host path and we bind it in | Works for **contents**, host-path only; no FD/namespace passthrough; no DAX control |
| (b) **runtimeClass + handler options** | select `kata`/custom handler per pod | Selects isolation, not mounts | Necessary for Kata-via-CRI, insufficient alone |
| (c) **custom containerd shim** (fork/wrap `io.containerd.kata.v2`) | shim attaches our virtiofs/DAX device to the VM | Yes, fully (incl. DAX) | Powerful but = maintaining a kata shim fork (high ongoing cost) |
| (d) **NRI plugin** (Node Resource Interface) | `CreateContainer` hook returns `ContainerAdjustment` that **adds mounts** | Yes, for host-path mounts | **The modern seam**; confirmed NRI can add arbitrary bind-mounts pre-run |

NRI is the right answer for option (d): the `ContainerAdjustment` returned from the
`CreateContainer` hook supports adding mounts (bind-mounts, and proc/sys/tmpfs
changes) before the runtime runs — this is exactly "modify the OCI spec / mounts
before the container is created." Both containerd and CRI-O expose NRI. But NRI
adjusts the **OCI spec**, so what it injects is still a **host-path mount** — it
consumes a vfsd host mount, it does not transport a live namespace.

**Conclusion (2):** NRI (d) is the clean, non-fork way to get the VFS *contents*
into a guest under a generic CRI runtime — feeding it a per-sandbox host mount that
vfsd provides. A custom shim (c) is the only way to control the **DAX window**, but
that re-introduces a kata-tracking maintenance burden comparable to embedding.

## 3. The Kata virtio-fs / DAX lead

containerd's kata shim already runs virtiofsd to share the rootfs, and **Kata uses
DAX** to map the guest image into the VM (avoiding host/guest cache duplication).
Kata config exposes `enable_virtio_fs`, `virtio_fs_cache`, `virtio_fs_cache_size`,
and `virtio_fs_extra_args`.

The hopeful path: register vfsd as an **additional** virtio-fs source for a kata
sandbox. The boundaries found:

- Kata's documented **DAX mapping is for the rootfs image**; the *shared* dir
  (`kataShared`) is a separate virtio-fs share whose **per-mount DAX behaviour is
  nuanced** and tied to the virtiofs cache mode (open kata-dev discussion on
  virtiofs cache vs DAX). I **could not confirm from docs** that an arbitrary
  *extra* virtio-fs share gets a guest DAX window the way the rootfs does.
- `virtio_fs_extra_args` warns **not** to set `source=` directly (kata injects it),
  so wiring an extra source through plain config is fragile; doing it cleanly points
  back at a custom shim or a kata feature.

**Conclusion (3):** Getting the VFS *contents* into a kata guest over CRI is
feasible (NRI bind / kataShared). Getting **DAX zero-copy for weights on that extra
share** is the hard boundary — it depends on VMM/kata DAX support for additional
shares, which is not a plain-CRI capability today. **DAX is the differentiator that
generic CRI cannot guarantee.**

## 4. What's preserved vs lost via CRI + NRI

| Differentiator | Through CRI+NRI (host-mount of vfsd) | Notes |
|---|---|---|
| RPC-backed dynamic VFS tree | **Preserved** | vfsd serves it at the host path; guest sees live contents |
| Per-`Subject` namespace isolation | **Preserved** | one vfsd mount per sandbox, keyed by Subject (vfsd already runs ops under a `Subject`) |
| Auth binding | **Preserved** | hyprstream drives the CRI client; auth/policy stays at our capnp ingress; containerd socket is local-trusted |
| **DAX zero-copy weight load** | **Lost / unproven** | needs guest DAX window on the extra share → VMM/kata-config or custom shim, not plain CRI |
| Kata guest-agent/exec/stats (#342/#344/#345) | **Gained for free** | containerd kata shim provides them → sheds the GA epic's hardest items *for this mode* |
| Image pull / CNI / cgroups / logs | **Gained for free** | containerd does it |

## 5. Effort comparison

- **CRI gRPC client (tonic):** generate from `k8s.io/cri-api runtime/v1`
  (`RuntimeService` + `ImageService`) with `tonic-build` — straightforward; the
  capnp↔gRPC mapping is near-isomorphic since our schema *is* CRI v1. Low.
- **NRI plugin (Rust):** NRI is a ttRPC/protobuf protocol; the mature stubs
  (`containerd/nri/pkg/stub`) are **Go**. A Rust NRI plugin must be generated from
  the NRI proto + a ttRPC runtime — doable but **more friction than Go** and a
  smaller ecosystem. Medium. *(Open question: accept a small Go NRI sidecar vs.
  build Rust NRI.)*
- **Custom kata shim fork (for DAX):** **High, ongoing** — tracks kata releases,
  re-does what the embedded path already does. For the DAX case this is **not
  cheaper** than embedding.
- **Embedded Kata + vfsd (status quo, #496):** the existing investment; the only
  path that already controls the DAX window directly.

## Verdict

**Partial yes — and it sharpens the architecture rather than replacing it.**

1. **The VFS *contents* plane CAN go through a generic CRI runtime**, via **NRI
   mount-injection** of a **per-sandbox host mount that vfsd provides** (option d,
   with a host-path bind). This is the clean, non-fork mechanism.
2. **The DAX zero-copy weight path does NOT cleanly go through generic CRI.** The
   guest DAX window for an *extra* virtio-fs share is a VMM/kata concern, reachable
   only via a custom shim (high, ongoing cost) — so **DAX stays an embedded-path
   (vfsd + in-process runtime) differentiator.** That is the boundary.
3. **vfsd (#495/#496) is required in BOTH models** — it is the host-side provider of
   the mount, whether the lifecycle is driven by our embedded runtime or by
   containerd. The CRI-client decision does **not** waste #496.

### Recommendation: `CriBackend` as a standard-host backend *option*

Add `CriBackend` (CRI client over containerd/CRI-O) as **one more `SandboxBackend`**
behind the same capnp-CRI ingress (the trait already abstracts this:
`SandboxBackend::{start,stop,destroy,exec_sync,...}` in
`crates/hyprstream-workers/src/runtime/backend.rs`):

- **Use `CriBackend`** on standard/production hosts that already run containerd:
  lowest-impedance (CRI→CRI), Kata-via-runtimeClass sheds #342/#344/#345 for that
  mode, image/CNI/cgroups for free. VFS contents via NRI + vfsd host mount.
- **Keep embedded Kata + vfsd** for the **DAX weight-loading** differentiation —
  the only path that controls the guest DAX window.
- **Keep crun/youki (oci-spec)** for the **casual/rootless dev** path (no daemon).

`BackendType` selects among the three; `runtimeClass` is the in-CRI selector for the
standard path. The boundary line is crisp: **everything except DAX-zero-copy can ride
CRI; DAX stays embedded.**

### Open questions (not verifiable from docs in this spike)
- Does kata's `kataShared` (or an extra virtio-fs share) support a **per-mount guest
  DAX window**, or is DAX effectively rootfs-only? (kata-dev cache-vs-DAX thread is
  inconclusive.) If per-mount DAX is supported, the boundary moves and CRI-client
  could cover DAX too.
- **Rust NRI** maturity — accept a Go NRI sidecar, or invest in Rust NRI from proto?
- Can an **extra virtio-fs source** be attached to a kata sandbox via annotations
  alone (no shim fork)?

These three gate how *far* `CriBackend` can go; none of them block adding it as the
standard-host option with the DAX boundary as stated.
