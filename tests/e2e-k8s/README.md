# hyprstream e2e Kubernetes harness (KIND-in-Podman)

A committed, scriptable **local Kubernetes** harness for validating the K8s-facing
surface of hyprstream without a cloud cluster. Built on **rootless podman + kind**
(docker is deliberately absent from this environment). Home issue: **#975**
(foundational for the #778 K8s epic).

```
KIND_EXPERIMENTAL_PROVIDER=podman   kind v0.31.0
podman 5.8.3 (rootless, cgroups v2, netavark/pasta)   # no docker
kubectl (client only)
```

## Layout

```
tests/e2e-k8s/
  kind-config.yaml   1 control-plane + 2 workers, podman provider, 30443 host<->node
  up.sh              bring cluster up (reuses a healthy cluster; --recreate to force)
  down.sh            tear cluster down
  doctor.sh          pre-flight host diagnostics (cgroup delegation, sysctls, linger)
  lib/
    common.sh        shared helpers (kubeconfig, preflight, kind/kubectl wrappers)
    metallb.sh       metallb address-pool discovery + apply (optional; NodePort fallback)
    addr.sh          address/subnet helpers
```

## Quick start

```bash
cd tests/e2e-k8s
./doctor.sh     # show host readiness (does not modify the host)
./up.sh         # create + reuse; writes ~/.kube/kind-hyprstream
kubectl --kubeconfig ~/.kube/kind-hyprstream get nodes
./down.sh       # tear down
```

`KUBECONFIG` defaults to `~/.kube/kind-hyprstream`; override per-shell if needed.

## Rootless-podman prerequisites (the finicky part)

KIND-in-podman is **not** turnkey on a stock host. `./doctor.sh` checks all four
below and prints the exact `sudo` line for any that fail. **None can be done
without sudo** — this is the load-bearing host prep a human must perform before
the harness can bring a cluster up on a locked-down dev/CI box.

1. **cgroup v2 delegation** (most common). Rootless podman with
   `cgroupManager=systemd` needs the cgroup v2 controllers delegated to your user
   slice, or kind's node containers fail to start kubelet/kube-proxy with opaque
   `runc` errors.
   ```bash
   sudo mkdir -p /etc/systemd/system/user@$(id -u).service.d
   printf '[Service]\nDelegate=yes\n' | \
     sudo tee /etc/systemd/system/user@$(id -u).service.d/delegate.conf
   sudo systemctl daemon-reload   # then re-login or: systemctl restart user@$(id -u).service
   ```
2. **User linger** — without it, `systemd --user` (and rootless podman + its netns)
   dies when your session ends, killing the cluster.
   ```bash
   sudo loginctl enable-linger "$USER"
   ```
3. **`net.ipv4.ip_forward`** — rootless networking (netavark/pasta) needs the host
   to forward for pod egress.
   ```bash
   sudo sysctl -w net.ipv4.ip_forward=1   # persist in /etc/sysctl.d/
   ```
4. **inotify limits** — each node runs a full kubelet; the default
   `fs.inotify.max_user_instances=128` is exhausted by a multi-node cluster
   (workers crash-loop with `inotify_init: too many open files`).
   ```bash
   sudo sysctl -w fs.inotify.max_user_watches=524288
   sudo sysctl -w fs.inotify.max_user_instances=512
   ```

## LoadBalancer support (metallb, optional)

`up.sh` installs metallb and carves an `IPAddressPool` from the podman `kind`
bridge subnet. If `install_metallb` fails, `up.sh` warns and continues; cases
that need external reach can fall back to the stable NodePort `30443` (TCP+UDP,
mapped host<->node in `kind-config.yaml`).

## Adding a new e2e case

The scaffold pattern is **deploy -> assert -> teardown**:

1. `kubectl apply` your manifests (Deployment + Service).
2. `kubectl wait --for=condition=Available deployment/...`.
3. Run the measurement (`kubectl run` a client pod, or `kubectl port-forward`).
4. `trap teardown EXIT` so the cluster is left clean.
