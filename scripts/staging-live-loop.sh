#!/usr/bin/env bash
# Bounded, on-instance staging iteration. Run as rocky from a Hyprstream checkout.
# This does not replace the pinned-image CI deployment path.
set -euo pipefail

usage() {
  echo "usage: $0 probe | build | activate SERVICE IMAGE | rollback SERVICE" >&2
  exit 2
}

command_name="${1:-}"
case "$command_name" in probe|build|activate|rollback) shift ;; *) usage ;; esac

if [ "$(id -un)" != rocky ]; then
  echo 'Run as rocky: staging Quadlets and Podman storage belong to rocky.' >&2
  exit 2
fi

export XDG_RUNTIME_DIR="/run/user/$(id -u)"
quadlet_dir="${HYPRSTREAM_DEV_QUADLET_DIR:-/home/rocky/.config/containers/systemd}"
state_dir="${HYPRSTREAM_DEV_STATE_DIR:-/home/rocky/.local/state/hyprstream-live-loop}"
cache_dir="${HYPRSTREAM_DEV_CACHE_DIR:-/home/rocky/.cache/hyprstream-dev-build}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

service_file() {
  case "$1" in
    event|registry|policy|discovery|streams|model|oai|oauth) ;;
    *) echo "Unknown staging service: $1" >&2; exit 2 ;;
  esac
  if [ "$1" = event ]; then
    local current="$quadlet_dir/hyprstream-event.container"
    local legacy="$quadlet_dir/hyprstream.container"
    if { [ -e "$current" ] || [ -L "$current" ]; } &&
       { [ -e "$legacy" ] || [ -L "$legacy" ]; }; then
      echo 'Ambiguous Event Quadlets: both hyprstream-event.container and hyprstream.container exist.' >&2
      return 2
    fi
    if [ -e "$current" ] || [ -L "$current" ]; then
      printf '%s\n' "$current"
    elif [ -e "$legacy" ] || [ -L "$legacy" ]; then
      printf '%s\n' "$legacy"
    else
      echo 'Missing Event Quadlet: expected hyprstream-event.container or hyprstream.container.' >&2
      return 2
    fi
    return
  fi
  printf '%s/hyprstream-%s.container\n' "$quadlet_dir" "$1"
}

service_unit() {
  local filename="${1##*/}"
  printf '%s.service\n' "${filename%.container}"
}

restore_quadlet() {
  local service="$1" file="$2" backup="$3" stage
  stage="$(mktemp "$quadlet_dir/.hyprstream-$service.restore.XXXXXX")"
  if ! cp -p -- "$backup" "$stage"; then
    rm -f -- "$stage"
    return 1
  fi
  mv -f -- "$stage" "$file"
}

require_guest() {
  [ -d "$quadlet_dir" ] || { echo "Missing staging Quadlets: $quadlet_dir" >&2; exit 2; }
  command -v podman >/dev/null || { echo 'Podman is missing.' >&2; exit 2; }
  command -v systemctl >/dev/null || { echo 'systemctl is missing.' >&2; exit 2; }
}

probe() {
  echo "host=$(hostname) user=$(id -un) arch=$(uname -m)"
  echo "source=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unavailable)"
  if [ -n "$(git -C "$repo_root" status --porcelain 2>/dev/null || true)" ]; then
    echo 'source_worktree=dirty'
  else
    echo 'source_worktree=clean'
  fi
  df -h -- "$repo_root"
  [ ! -d "$quadlet_dir" ] || df -h -- "$quadlet_dir"
  if command -v podman >/dev/null; then
    graph_root="$(podman info --format '{{.Store.GraphRoot}}' 2>/dev/null || true)"
    [ -z "$graph_root" ] || df -h -- "$graph_root"
  fi
  for tool in podman systemctl git; do
    command -v "$tool" || true
  done
  if [ -d "$quadlet_dir" ]; then
    for service in policy registry model; do
      file="$(service_file "$service")"
      [ -f "$file" ] || { echo "missing_quadlet=$file"; continue; }
      echo "quadlet=$file"
      systemctl --user is-active "hyprstream-$service.service" || true
    done
  fi
}

build() {
  [ "$#" -eq 0 ] || usage
  require_guest
  [ -f "$repo_root/Dockerfile" ] || { echo 'Dockerfile is missing.' >&2; exit 2; }
  case "$(uname -m)" in
    x86_64) variant=cpu ;;
    aarch64) variant=cpu-arm64 ;;
    *) echo 'Unsupported staging build architecture.' >&2; exit 2 ;;
  esac
  case "$cache_dir" in /*) ;; *) echo 'Build cache must be an absolute path.' >&2; exit 2 ;; esac
  jobs="${HYPRSTREAM_DEV_BUILD_JOBS:-2}"
  case "$jobs" in 1|2|3|4) ;; *) echo 'Build jobs must be between 1 and 4.' >&2; exit 2 ;; esac
  mkdir -p -m 0700 "$cache_dir"
  [ ! -L "$cache_dir" ] && [ -O "$cache_dir" ] || { echo 'Build cache must be an owned directory, not a symlink.' >&2; exit 2; }
  # The production Dockerfile builds the exact staging feature set and its
  # matching runtime libraries. A bare host binary would not exercise either.
  graph_root="$(podman info --format '{{.Store.GraphRoot}}')"
  [ -d "$graph_root" ] || { echo "Missing Podman storage: $graph_root" >&2; exit 2; }
  source_free_kib="$(df -Pk -- "$cache_dir" | awk 'NR == 2 { print $4 }')"
  image_free_kib="$(df -Pk -- "$graph_root" | awk 'NR == 2 { print $4 }')"
  minimum_kib=$((30 * 1024 * 1024))
  if [ "$(stat -f -c %i -- "$cache_dir")" = "$(stat -f -c %i -- "$graph_root")" ]; then
    # These two consumers share one pool; do not count the same free blocks twice.
    minimum_kib=$((2 * minimum_kib))
  fi
  if [ "$source_free_kib" -lt "$minimum_kib" ] ||
     [ "$image_free_kib" -lt "$minimum_kib" ]; then
    echo 'Refusing a cold image build: require 30 GiB per filesystem, or 60 GiB when build cache and image storage share one.' >&2
    exit 75
  fi
  mkdir -p -m 0700 "$state_dir"
  exec 9>"$state_dir/build.lock"
  flock -n 9 || { echo 'A staging build is already running.' >&2; exit 75; }
  revision="$(git -C "$repo_root" rev-parse --short=12 HEAD)"
  image="localhost/hyprstream-staging-dev:${revision}-$(date -u +%Y%m%d%H%M%S)"
  echo "Building $image from $repo_root (arch=$variant); no service will change."
  podman build --target runtime --build-arg "VARIANT=$variant" \
    --build-arg "HYPRSTREAM_BUILD_JOBS=$jobs" \
    --volume "$cache_dir:$cache_dir:Z" \
    --build-arg "CARGO_TARGET_DIR=$cache_dir/target" \
    -t "$image" -f "$repo_root/Dockerfile" "$repo_root"
  podman image exists "$image"
  echo "built_image=$image"
}

activate() {
  [ "$#" -eq 2 ] || usage
  require_guest
  service="$1"
  image="$2"
  file="$(service_file "$service")"
  unit="$(service_unit "$file")"
  case "$image" in
    localhost/hyprstream-staging-dev:*) ;;
    *) echo 'Only a locally built staging-dev image may be activated.' >&2; exit 2 ;;
  esac
  [ -f "$file" ] && [ ! -L "$file" ] || { echo "Invalid Quadlet: $file" >&2; exit 2; }
  podman image exists "$image" || { echo "Image does not exist: $image" >&2; exit 2; }
  mkdir -p -m 0700 "$state_dir"
  exec 8>"$state_dir/update.lock"
  flock -n 8 || { echo 'A staging service update is already running.' >&2; exit 75; }
  backup="$state_dir/hyprstream-$service.container.original"
  [ "$(grep -c '^Image=' "$file")" -eq 1 ] || { echo 'Expected exactly one Image= line.' >&2; exit 2; }
  if [ -e "$backup" ]; then
    [ -f "$backup" ] && [ ! -L "$backup" ] || { echo 'Invalid rollback configuration.' >&2; exit 2; }
  fi
  # Preserve the release during a hot-update sequence, but refresh it after
  # normal deployment has replaced the local override with a newer release.
  if [ ! -e "$backup" ] || ! grep -q '^Image=localhost/hyprstream-staging-dev:' "$file"; then
    backup_stage="$(mktemp "$state_dir/.hyprstream-$service.backup.XXXXXX")"
    if ! cp -p -- "$file" "$backup_stage"; then
      rm -f -- "$backup_stage"
      exit 1
    fi
    mv -f -- "$backup_stage" "$backup"
  fi
  [ -f "$backup" ] && [ ! -L "$backup" ] || { echo 'Invalid rollback configuration.' >&2; exit 2; }
  stage="$(mktemp "$quadlet_dir/.hyprstream-$service.XXXXXX")"
  trap 'rm -f -- "$stage"' EXIT
  awk -v image="$image" '
    /^Image=/ { print "Image=" image; next }
    /^AutoUpdate=registry$/ { next }
    { print }
  ' "$file" >"$stage"
  chmod 0644 "$stage"
  mv -f -- "$stage" "$file"
  trap - EXIT
  if ! systemctl --user daemon-reload ||
     ! systemctl --user restart "$unit" ||
     ! systemctl --user is-active --quiet "$unit"; then
    echo 'Service failed; restoring original Quadlet.' >&2
    if ! restore_quadlet "$service" "$file" "$backup"; then
      echo "Could not restore the original Quadlet; retained rollback configuration: $backup" >&2
      exit 1
    fi
    systemctl --user daemon-reload
    if systemctl --user restart "$unit" &&
       systemctl --user is-active --quiet "$unit"; then
      rm -f -- "$backup"
    else
      echo "Original service is still unhealthy; retained rollback configuration: $backup" >&2
    fi
    exit 1
  fi
  echo "active_image=$image service=$unit"
  echo "rollback_command=$0 rollback $service"
}

rollback() {
  [ "$#" -eq 1 ] || usage
  require_guest
  service="$1"
  file="$(service_file "$service")"
  unit="$(service_unit "$file")"
  exec 8>"$state_dir/update.lock"
  flock -n 8 || { echo 'A staging service update is already running.' >&2; exit 75; }
  backup="$state_dir/hyprstream-$service.container.original"
  [ -f "$backup" ] && [ ! -L "$backup" ] || { echo "No saved Quadlet: $backup" >&2; exit 2; }
  [ -f "$file" ] && [ ! -L "$file" ] || { echo "Invalid Quadlet: $file" >&2; exit 2; }
  if ! grep -q '^Image=localhost/hyprstream-staging-dev:' "$file" && ! cmp -s -- "$file" "$backup"; then
    echo 'Normal deployment superseded this checkpoint; refusing to overwrite its configuration.' >&2
    exit 2
  fi
  restore_quadlet "$service" "$file" "$backup"
  systemctl --user daemon-reload
  systemctl --user restart "$unit"
  systemctl --user is-active --quiet "$unit"
  rm -f -- "$backup"
  echo "restored=$unit"
}

case "$command_name" in
  probe) [ "$#" -eq 0 ] || usage; probe ;;
  build) build "$@" ;;
  activate) activate "$@" ;;
  rollback) rollback "$@" ;;
esac
