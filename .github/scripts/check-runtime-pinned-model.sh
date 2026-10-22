#!/bin/sh
# Local fixture overrides are explicit arguments; production invokes no args.
set -eu
git_bin=${1:-/usr/bin/git}
cache=${2:-/var/cache/hyprstream-pinned}
test "$#" -le 2
test "$(id -u)" != 0
test -x "$git_bin"
test -d "$cache" && test ! -L "$cache"
test "$(stat -c '%a:%u:%g' "$cache")" = "700:$(id -u):$(id -g)"
case "$(stat -f -c '%T' "$cache")" in tmpfs|ramfs) exit 1 ;; esac
test -w "$cache"
work=$(mktemp -d "$cache/runtime-check.XXXXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM
mkdir "$work/templates"
env -i "$git_bin" init --bare --template="$work/templates" "$work/repo" >/dev/null 2>&1
oid=$(printf 'pinned-runtime-fixture\n' | env -i GIT_DIR="$work/repo" "$git_bin" hash-object -w --stdin)
actual=$(printf '%s\n' "$oid" | env -i GIT_DIR="$work/repo" \
    GIT_NO_REPLACE_OBJECTS=1 GIT_CONFIG_NOSYSTEM=1 GIT_OPTIONAL_LOCKS=0 \
    "$git_bin" cat-file --batch)
expected=$(printf '%s blob 23\npinned-runtime-fixture\n\n' "$oid")
test "$actual" = "$expected"
printf 'runtime-pinned-model: PASS (nonroot, private disk cache, env-clear Git batch)\n'
