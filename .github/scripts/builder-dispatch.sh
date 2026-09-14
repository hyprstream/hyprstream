#!/usr/bin/env bash
# Validate dispatch identity before any build, registry login, or publication.
set -euo pipefail
actual=$(git rev-parse HEAD)
[[ "$actual" == "${GITHUB_SHA:?}" ]] || { echo 'checkout differs from event SHA' >&2; exit 1; }
mode=release
if [[ "${GITHUB_EVENT_NAME:?}" == workflow_dispatch ]]; then
    [[ "${EXPECTED_SOURCE_SHA:-}" =~ ^[0-9a-f]{40}$ && "$actual" == "$EXPECTED_SOURCE_SHA" ]] || {
        echo 'expected_source_sha must be the exact reviewed 40-hex checkout SHA' >&2; exit 1;
    }
    case "${CANDIDATE:-}" in
        true) mode=candidate ;;
        false) [[ "${GITHUB_REF:?}" == refs/heads/main ]] || {
            echo 'non-main dispatch must use candidate mode' >&2; exit 1;
        } ;;
        *) echo 'candidate must be true or false' >&2; exit 1 ;;
    esac
else
    case "${GITHUB_EVENT_NAME}:${GITHUB_REF:?}" in
        push:refs/heads/main|push:refs/heads/ewindisch/runner-smoke-test|schedule:refs/heads/main) ;;
        *) echo 'unsupported builder event/ref' >&2; exit 1 ;;
    esac
fi
printf 'mode=%s\nsource_sha=%s\n' "$mode" "$actual" >> "${GITHUB_OUTPUT:?}"
