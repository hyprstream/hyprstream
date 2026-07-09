# hyprstream e2e Kubernetes harness (KIND-in-Podman)

A committed, scriptable **local Kubernetes** harness for validating the K8s-facing
surface of hyprstream without a cloud cluster. Home issue: **#975** (foundational
for the #778 K8s epic). Built on **rootless podman + kind** because docker is
deliberately absent from this environment.

```
KIND_EXPERIMENTAL_PROVIDER=podman  kind  v0.31.0
podman 5.8.3 (rootless, cgroups v2, netavark/pasta)   ← no docker
kubectl (client only)
```

## Layout

```
tests/e2e-k8s/
  kind-config.yaml        1 control-plane + 2 workers, podman provider, 30443 host<->node
  up.sh                   bring cluster up (reuses healthy cluster; --recreate to force)
  down.sh                 tear cluster down
  doctor.sh               pre-flight diagnostics (cgroup delegation, sysctls, linger)
  lib/
    common.sh             shared helpers (kubeconfig, preflight, kind/kubectl wrappers)
    metallb.sh            metallb address-pool discovery + apply (optional; NodePort fallback)
  spikes/
    sp799-ingress/        #799  cert-pinned QUIC/WebTransport across pod→LB boundary
    sp458-anycast/        #458 anycast flow-stability (forced backend flap)
```

## Quick start

```bash
cd tests/e2e-k8s
./doctor.sh                 # show host readiness (does not modify host)
./up.sh                     # create + reuse; writes ~/.kube/kind-hyprstream
kubectl --kubeconfig ~/.kube/kind-hyprstream get nodes
./down.sh                   # tear down
```

`KUBECONFIG` defaults to `~/.kube/kind-hyprstream`; override per-shell if needed.

## Rootless-podman prerequisites (the finicky part)

KIND-in-podman is **not** turnkey on a stock host. The known blockers, in order
of how often they bite, and the exact fix for each:

### 1. cgroup v2 delegation (most common)

Rootless podman with `cgroupManager=systemd` needs the cgroup v2 controllers
delegated to your user slice. Without it, kind's node containers fail to start
their kubelet/kube-proxy with opaque `runc` errors.

Check: `cat /sys/fs/cgroup/user.slice/$(id -u).user.slice/cgroup.controllers`
should list `memory` (and cpu/io/pids). If empty/absent:

```bash
sudo mkdir -p /etc/systemd/system/user@$(id -u).service.d
printf '[Service]\nDelegate=yes\n' | \
  sudo tee /etc/systemd/system/user@$(id -u).service.d/delegate.conf
sudo systemctl daemon-reload
# log out + back in (or: systemctl restart user@$(id -u).service) for it to take
```

### 2. User linger (rootless session lifetime)

Without linger, your `systemd --user` instance (and the rootless podman runtime
+ the rootless network namespace it creates) dies when your SSH session ends,
killing the cluster.

```bash
sudo loginctl enable-linger "$USER"
```

### 3. `net.ipv4.ip_forward` (pods egress / pod→service)

Rootless networking (netavark/pasta) needs the host to forward for pod egress,
and KIND nodes rely on it for pod↔service traffic in some configs:

```bash
sudo sysctl -w net.ipv4.ip_forward=1
# persist in /etc/sysctl.d/
```

### 4. inotify limits (kubelet runs a full kubelet per node)

```bash
sudo sysctl -w fs.inotify.max_user_watches=524288
sudo sysctl -w fs.inotify.max_user_instances=512
```

`./doctor.sh` checks all four and prints the exact `sudo` line for any that fail.
**None of these can be done without sudo** — that is the load-bearing host
preparation a human must perform before the harness can bring a cluster up on a
locked-down dev/CI box.

## LoadBalancer support (metallb, optional)

`up.sh` installs metallb and carves an `IPAddressPool` out of the podman `kind`
bridge subnet. **Both spikes degrade gracefully without it:**

- **#799** is fully runnable over the stable NodePort `30443` (TCP+UDP, mapped
  host↔node in `kind-config.yaml`) — the cert-pin verification is identical, the
  only difference is the advertised reach is a node IP:NodePort rather than a
  LB VIP.
- **#458** needs no LB at all — it flaps a `ClusterIP` service's endpoints and
  observes from a client pod inside the cluster.

If `install_metallb` fails, `up.sh` warns and continues; spike scripts detect
metallb's absence and use NodePort.

## What this harness deliberately cannot validate

Per #975's scope boundary:

- **Real BGP anycast routing / convergence timing** (#458's remainder) — needs
  actual multi-POP anycast infra. The harness emulates the *failure mode* (a
  mid-flow backend reroute) via endpoint/pod deletion, which is what tests
  whether QUIC/moq/WT survives a flap; the BGP-level convergence-time question
  is a real-infra follow-up.
- **Cloud-LB-specific behavior** (source-IP preservation, cloud TLS termination)
  — approximated by metallb, not identical.

## Adding a new e2e case

The reusable scaffold pattern is **deploy → assert → teardown**, one directory
under `spikes/`. See `spikes/sp458-anycast/run.sh` for the canonical shape:

1. `kubectl apply` your manifests (Deployment + Service).
2. `kubectl wait --for=condition=Available deployment/...`.
3. Run the measurement (`kubectl run` a client pod, or port-forward).
4. `trap teardown EXIT` so the cluster is left clean.

Each spike records its result in its own `FINDINGS.md`; a consolidated
`spikes/FINDINGS.md` rolls them up.
