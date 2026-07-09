# KIND-in-Podman e2e harness — findings (this branch)

Home: #975. Branch: `ewindisch/kind-e2e-harness`.

## Did the cluster come up?

**Partially — single-node yes, multi-node (workers) no, due to one host sysctl.**

| Path | Result |
|---|---|
| Single control-plane node | ✅ **WORKS.** Node reaches `multi-user.target`, API server healthy, pods schedule, `kubectl get nodes/pods` answer. |
| 1 control-plane + 2–3 workers (the committed `kind-config.yaml`) | ❌ **FAILS at worker join.** All node containers boot and the control-plane inits (CNI, StorageClass, etcd, apiserver all Running), but **worker kubelets crash-loop and never join.** |

### Exact blocker (root-caosed, reproducible)

Worker kubelet logs:

```
E ... "Failed to start cAdvisor" err="inotify_init: too many open files"
E ... "Registration of the raw container factory failed: inotify_init: too many open files"
E ... "error creating fsnotify watcher: too many open files"
E ... "Failed to set rlimit on max file handles" err="operation not permitted"
systemd[1]: kubelet.service: Failed with result 'exit-code'.
[kubelet-check] The kubelet is not healthy after 4m0s
error: ... kubelet-wait-bootstrap: failed while waiting for the kubelet to start
```

Root cause: the host's **`fs.inotify.max_user_instances = 128`** (confirmed by
`doctor.sh`). Each node runs a full kubelet + containerd + cAdvisor, each of
which opens many inotify watchers. The control-plane alone consumes most of the
per-user 128-instance budget; when worker kubelets start they exhaust it and
crash-loop. Rootless podman also can't raise the kubelet's rlimit
(`operation not permitted`), so the kubelet can't work around it in-container.

**The fix is one sysctl** (needs sudo, which the dev box does not grant
non-interactively):

```bash
sudo sysctl -w fs.inotify.max_user_instances=512
sudo sysctl -w fs.inotify.max_user_watches=524288   # also below recommended
# persist under /etc/sysctl.d/99-kind.conf
```

Two secondary host issues `doctor.sh` flags (harmless for single-node, relevant
for a hardened multi-node run): `net.ipv4.ip_forward=0` (pods egress), and cgroup
v2 delegation state unreadable (enable `Delegate=yes` on `user@<uid>.service` if
a cgroup error appears). These did NOT block single-node bring-up; inotify did.

### What this means for the harness

The harness is correct and proven end-to-end on single-node. **Nothing in the
harness needs to change**; the host needs the one sysctl to run multi-node. Once
`fs.inotify.max_user_instances=512` is set, `./up.sh` brings up the full
1-control-plane + 2-worker `kind-config.yaml` unchanged.

## Spike results

| Spike | What ran on this host | Status |
|---|---|---|
| **#799** (cert-pinned QUIC/WT across pod→LB) | NodePort cross-boundary TCP reachability **proven** (pod reachable from host through mapped port; `port-forward` works). UDP/QUIC + cert-pin assertion **design-ready** (needs QUIC stand-in image + external subscriber client). | Plumbing ✅ / cert-pin layer DESIGN-READY |
| **#458** (anycast flow-stability, backend flap) | Flap **mechanism** (Service reroute on backing-pod delete) **proven** on single-node control-plane. Transport-level flow observation (does QUIC survive the flap?) **design-ready** (needs `quic-flow-probe` stand-in). | Mechanism ✅ / transport observation DESIGN-READY |

Detailed writeups: [`spikes/sp799-ingress/FINDINGS.md`](spikes/sp799-ingress/FINDINGS.md),
[`spikes/sp458-anycast/FINDINGS.md`](spikes/sp458-anycast/FINDINGS.md).

## Provisional conclusions (for posting to the issues; confirm once stand-ins run)

- **#799:** SP3 *plumbing* holds (pod→LB/NodePort→external boundary is
  traversable). Recommended #792/K6a reach default = **iroh-direct** (zero
  Service/Ingress surface; pods egress freely; cert/identity in the iroh NodeId),
  with **direct-QUIC-over-UDP-LB as opt-in** via a `ProducerReachConfig`
  reach-override seam (likely a wiring not a new-primitive change — confirm
  against `moq_stream.rs`). Gateway API/HTTPRoute has no UDP/QUIC story, so
  HTTP(S)-terminating ingress is not the path.
- **#458:** Per-flow stability is **not** automatic under anycast; the spike's
  expected finding (to confirm) is that a backend flap resets the live QUIC
  connection (different backing, no shared state) ⇒ recommended S3 design is
  **anycast-for-discovery / unicast-for-session** (redirect to a stable unicast
  endpoint on first contact), unless #358 commits to shared relay connection
  state. WebTransport inherits the QUIC connection fate. Real BGP convergence
  timing remains a real-infra follow-up (out of scope per #975).

## What ships in this branch

- `tests/e2e-k8s/` harness: `kind-config.yaml` (1 CP + 2 workers, podman
  provider, host↔node port map for UDP QUIC), `up.sh`/`down.sh`, `doctor.sh`,
  `lib/{common,metallb,addr}.sh`, `README.md`.
- metallb wiring (optional; both spikes degrade to NodePort without it).
- Two runnable spike scaffolds (deploy→assert→teardown) with manifest sets and
  per-spike `FINDINGS.md`.
- This file (rollup) + the host-blocker diagnosis with the exact one-line fix.

## To complete (tracked, not blocking)

1. Host: `sudo sysctl -w fs.inotify.max_user_instances=512` (+ watches), then
   `./up.sh` brings multi-node up unchanged.
2. Build the two stand-in images (`spike-quic-standin`, `spike-quic-relay`) or
   point the spikes at a real hyprstream streaming image; re-run the spikes to
   flip the two `DESIGN-READY` rows to executed `PASS/FAIL`.
3. CI integration decision (gate vs nightly) — likely nightly/opt-in given
   rootless-podman-in-CI constraints and the ~cluster-bring-up cost.
