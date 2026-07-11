#!/usr/bin/env bash
# Shared helpers for the hyprstream KIND-in-Podman e2e harness.
# Sourced (never executed directly).

# Emit to stderr with a tag; stdout stays clean for callers that capture output.
log()  { printf '[e2e-k8s] %s\n' "$*" >&2; }
warn() { printf '[e2e-k8s][WARN] %s\n' "$*" >&2; }
die()  { printf '[e2e-k8s][FATAL] %s\n' "$*" >&2; exit 1; }

require_cmd() {
  local missing=()
  for c in "$@"; do
    command -v "$c" >/dev/null 2>&1 || missing+=("$c")
  done
  [[ ${#missing[@]} -eq 0 ]] || die "missing required commands on PATH: ${missing[*]}"
}

# Where the kubeconfig lives. Deterministic so spike scripts and an interactive
# shell agree. Respects $KUBECONFIG if the caller already set one.
e2e_kubeconfig() {
  printf '%s\n' "${KUBECONFIG:-${HOME}/.kube/kind-hyprstream}"
}
KUBECONFIG="$(e2e_kubeconfig)"
export KUBECONFIG

# Pre-flight checks for the rootless-podman + kind combo. These are the known
# failure modes; we surface them up front with a clear message instead of letting
# kind emit an opaque 200-line traceback. This does NOT modify the host — it only
# reports. `up.sh` calls it; `doctor.sh` re-uses it.
preflight() {
  local problems=0

  # 1. podman must be rootless-capable and the user session must be active.
  #    `loginctl enable-linger <user>` is what makes a rootless podman survive
  #    SSH logout; without linger, systemd --user (and thus the rootless runtime)
  #    can vanish mid-test.
  if ! loginctl show-user "$(id -un)" 2>/dev/null | grep -q '^Linger=yes'; then
    warn "user linger is OFF — run: sudo loginctl enable-linger \$USER"
    warn "  (rootless podman containers will die when your login session ends)"
    problems=$((problems+1))
  fi

  # 2. cgroup v2 delegation. Rootless podman with cgroupManager=systemd needs the
  #    user delegated @set cpuset/cpu/io/memory/pids. This is the single most
  #    common kind-in-podman breakage. We can only READ this from the user slice.
  local uid
  uid="$(id -u)"
  local ucgroup="/sys/fs/cgroup/user.slice/user-${uid}.slice/user@${uid}.service"
  if [[ -r "${ucgroup}/cgroup.controllers" ]]; then
    if ! grep -q memory "${ucgroup}/cgroup.controllers"; then
      warn "memory controller not delegated to your user slice"
      warn "  fix: sudo systemctl edit user@${uid}.service, add:"
      warn "       [Service]"
      warn "       Delegate=yes"
      problems=$((problems+1))
    fi
  else
    # Unreadable often means delegation hasn't been set up at all; warn but
    # don't hard-fail (some hosts expose it differently).
    warn "cannot read cgroup delegation state at ${ucgroup} —"
    warn "  if cluster bring-up fails with a cgroup/runc error, enable delegation:"
    warn "  sudo mkdir -p /etc/systemd/system/user@${uid}.service.d"
    warn "  echo -e '[Service]\\nDelegate=yes' | sudo tee .../override.conf; systemctl daemon-reload"
    problems=$((problems+1))
  fi

  # 3. net.ipv4.ip_forward. With pasta/netavark rootless networking, the host
  #    needs forwarding for pod egress; KIND nodes also rely on it for
  #    pod-to-service traffic in some configurations.
  local ipfwd
  ipfwd="$(sysctl -n net.ipv4.ip_forward 2>/dev/null || echo unknown)"
  if [[ "$ipfwd" != "1" ]]; then
    warn "net.ipv4.ip_forward=${ipfwd} (want 1)"
    warn "  fix: sudo sysctl -w net.ipv4.ip_forward=1"
    problems=$((problems+1))
  fi

  # 4. inotify limits. kubelet + many pods can blow past the default instance
  #    count; kind node containers run a full kubelet.
  local watches instances
  watches="$(sysctl -n fs.inotify.max_user_watches 2>/dev/null || echo 0)"
  instances="$(sysctl -n fs.inotify.max_user_instances 2>/dev/null || echo 0)"
  if (( watches < 524288 )); then
    warn "fs.inotify.max_user_watches=${watches} (want >= 524288)"
    warn "  fix: sudo sysctl -w fs.inotify.max_user_watches=524288"
    problems=$((problems+1))
  fi
  if (( instances < 512 )); then
    warn "fs.inotify.max_user_instances=${instances} (want >= 512)"
    warn "  fix: sudo sysctl -w fs.inotify.max_user_instances=512"
    problems=$((problems+1))
  fi

  if (( problems > 0 )); then
    warn "${problems} pre-flight issue(s) found — bring-up may fail. See doctor.sh."
    return "$problems"
  fi
  log "pre-flight OK"
  return 0
}

# --- kind / kubectl wrappers (all force the podman provider) ---

kind_get_clusters() {
  KIND_EXPERIMENTAL_PROVIDER=podman kind get clusters 2>/dev/null
}

kind_delete_cluster() {
  local name="$1"
  KIND_EXPERIMENTAL_PROVIDER=podman kind delete cluster --name "$name"
}

write_kubeconfig() {
  local name="$1"
  mkdir -p "$(dirname "$KUBECONFIG")"
  KIND_EXPERIMENTAL_PROVIDER=podman kind get kubeconfig --name "$name" > "$KUBECONFIG"
  # The kubeconfig kind writes points at 127.0.0.1:<random-host-port>. Inside a
  # rootless podman setup that port is forwarded from the node container's 6443,
  # so 127.0.0.1 is correct on the host. We leave it untouched.
  log "kubeconfig for '${name}' -> ${KUBECONFIG}"
}

# Is the API server reachable AND answering? Used to decide reuse-vs-recreate.
kube_ready() {
  local name="$1"
  local workers
  workers="$(kube_ns_workers "$name")" || return 1
  (( workers > 0 ))
}

kube_ns_workers() {
  local name="${1:-}"
  if [[ -n "$name" ]]; then
    mkdir -p "$(dirname "$KUBECONFIG")"
    KIND_EXPERIMENTAL_PROVIDER=podman kind get kubeconfig --name "$name" > "$KUBECONFIG"
  fi

  local nodes
  nodes="$(kubectl --kubeconfig "$KUBECONFIG" get nodes --no-headers 2>/dev/null)" || return 1
  awk '$2 ~ /(^|,)Ready(,|$)/ { ready++ } END { print ready + 0; exit ready > 0 ? 0 : 1 }' <<<"$nodes"
}

# Locate a free TCP port on the host (best-effort, for ad-hoc port-forwards).
free_port() {
  python3 - <<'PY' 2>/dev/null || awk 'BEGIN{srand();print 30000+int(rand()*2000)}'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
PY
}
