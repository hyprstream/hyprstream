#!/usr/bin/env bash
# HyprStream ontology/dialect atlas sweep.
#
# Quarterly re-sweep owner: NEEDS OPERATOR ASSIGNMENT (#1091 section 7 / #1100).
# Do not replace this marker with a guessed person. The operator who accepts the
# ownership must update it in a reviewed PR.
#
# Usage: tools/ontology-sweep.sh <rev>
#
# Every source measurement is made from the resolved commit with `git grep` or
# `git show`; the current index and working tree are never read. This is
# intentional: an earlier atlas was invalidated by checkout skew.
#
# R9 definitions:
#   total LOC              every tracked textual line; implementation/schema
#                          LOC is printed separately for code-focused comparison
#   workspace crate count parsed members of the pinned root Cargo.toml
#   dead API candidates   pinned `allow(dead_code)` suppressions plus direct
#                         dependencies with no crate-name reference in that
#                         member's Rust sources (a cargo-machete-class scan)
#   generated/consumed    generated Cap'n Proto module declarations divided by
#                         downstream `_capnp::` reference lines; schema LOC is
#                         printed beside the ratio for scale
#
# If ONTOLOGY_DEAD_API_REPORT points to a report produced by cargo-machete or
# cargo-udeps, its SHA-256 is printed as supplemental evidence. The report must
# itself name this sweep's pinned SHA; the script deliberately does not run an
# analyzer against the live checkout.

set -euo pipefail

usage() {
    printf 'usage: %s <rev>\n' "${0##*/}" >&2
    exit 64
}

[[ $# -eq 1 ]] || usage

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
    printf 'error: ontology-sweep.sh must run inside a Git repository\n' >&2
    exit 69
}
cd "$repo_root"

rev=$1
pinned_sha=$(git rev-parse --verify --end-of-options "${rev}^{commit}" 2>/dev/null) || {
    printf 'error: revision is not a commit: %s\n' "$rev" >&2
    exit 65
}

# Count matching lines/occurrences/files at the pinned object. A no-match result
# is a valid zero, hence the guarded git-grep status in each pipeline.
grep_line_count() {
    local pattern=$1
    shift
    { git grep -I -E "$pattern" "$pinned_sha" -- "$@" 2>/dev/null || true; } |
        awk 'END { print NR + 0 }'
}

grep_occurrence_count() {
    local pattern=$1
    shift
    { git grep -I -h -o -E "$pattern" "$pinned_sha" -- "$@" 2>/dev/null || true; } |
        awk 'END { print NR + 0 }'
}

ratio() {
    awk -v numerator="$1" -v denominator="$2" 'BEGIN {
        if (denominator == 0) {
            print "n/a"
        } else {
            printf "%.3f", numerator / denominator
        }
    }'
}

printf 'pinned_sha=%s\n' "$pinned_sha"
printf 'requested_rev=%s\n' "$rev"
printf 'quarterly_owner=NEEDS_OPERATOR_ASSIGNMENT\n'

printf '\n[dialect-atlas]\n'

# WHO: raw DID strings at the placement boundary, subject-family type dialects,
# and model-identity type dialects.
placement_did=$(
    grep_line_count \
        '(^|[^[:alnum:]_])[[:alnum:]_]*did[[:space:]]*:[[:space:]]*(&str|String)' \
        crates/hyprstream-discovery/src/placement_index.rs
)
subject_types=$(
    grep_line_count \
        '^[[:space:]]*(pub([[:space:]]*\([^)]*\))?[[:space:]]+)?(struct|enum|trait|type)[[:space:]]+([[:alnum:]_]*Subject|Subject[[:alnum:]_]*)' \
        'crates/**/*.rs'
)
model_identity_types=$(
    grep_line_count \
        '^[[:space:]]*(pub([[:space:]]*\([^)]*\))?[[:space:]]+)?(struct|enum|type)[[:space:]]+(ModelId|ModelRef|ModelInstanceId|ModelRevision)' \
        'crates/**/*.rs'
)
printf 'WHO placement_bare_did=%s subject_type_declarations=%s model_identity_type_declarations=%s\n' \
    "$placement_did" "$subject_types" "$model_identity_types"

# MAY-DO: parse the two hand-maintained Operation lists from the pinned file.
operation_source=$(git show "${pinned_sha}:crates/hyprstream/src/auth/mod.rs")
operation_variants=$(
    awk '
        /^pub enum Operation \{/ { in_enum = 1; next }
        in_enum && /^}/ { in_enum = 0 }
        in_enum && /^[[:space:]]+[A-Z][A-Za-z0-9_]*,/ { count++ }
        END { print count + 0 }
    ' <<<"$operation_source"
)
operation_all=$(
    awk '
        /pub fn all\(\)/ { in_all = 1 }
        in_all && /Operation::[A-Za-z0-9_]+,/ { count++ }
        in_all && /^    }/ { in_all = 0 }
        END { print count + 0 }
    ' <<<"$operation_source"
)
operation_omitted=$((operation_variants - operation_all))
printf 'MAY-DO operation_variants=%s operation_all=%s operation_omitted=%s\n' \
    "$operation_variants" "$operation_all" "$operation_omitted"

# RESOLVE and DIAL report literal declarations; reviewers can distinguish
# legitimate axes from same-job contracts without relying on a hand count.
resolver_traits=$(
    grep_line_count \
        '^[[:space:]]*(pub([[:space:]]*\([^)]*\))?[[:space:]]+)?trait[[:space:]]+[A-Za-z0-9_]*Resolver([[:space:]:<{]|$)' \
        'crates/**/*.rs'
)
resolver_suffix_contracts=$(
    grep_line_count \
        '^[[:space:]]*(pub([[:space:]]*\([^)]*\))?[[:space:]]+)?trait[[:space:]]+(DidDocumentProvider|DidDocResolve|DidDocFetcher)([[:space:]:<{]|$)' \
        'crates/**/*.rs'
)
native_dials=$(
    grep_line_count \
        '^[[:space:]]*pub[[:space:]]+(async[[:space:]]+)?fn[[:space:]]+dial(_with_kem_store|_with_crypto_stores|_stream)?[<(]' \
        crates/hyprstream-rpc/src/dial.rs
)
wasm_dials=$(
    grep_line_count \
        '^[[:space:]]*pub[[:space:]]+(async[[:space:]]+)?fn[[:space:]]+dial(_with_kem_store|_with_crypto_stores|_with_js_signer)?[<(]' \
        crates/hyprstream-rpc/src/dial_wasm.rs
)
mount_dials=$(
    grep_line_count \
        '^[[:space:]]*pub[[:space:]]+fn[[:space:]]+dial[<(]' \
        crates/hyprstream-rpc-std/src/vfs_mount.rs
)
printf 'RESOLVE resolver_named_traits=%s resolver_suffix_contracts=%s total_resolver_contracts=%s\n' \
    "$resolver_traits" "$resolver_suffix_contracts" "$((resolver_traits + resolver_suffix_contracts))"
printf 'DIAL native_entry_points=%s wasm_entry_points=%s mount_entry_points=%s total_literal_entry_points=%s\n' \
    "$native_dials" "$wasm_dials" "$mount_dials" "$((native_dials + wasm_dials + mount_dials))"

# DIGEST: count active token families, while retaining per-family occurrences.
digest_families=0
digest_summary=()
for digest_spec in \
    'cid:(Cid|CID)' \
    'merkle:MerkleHash' \
    'sha256:(Sha256|SHA256|sha256:)' \
    'sha512:(Sha512|SHA512)' \
    'blake3:(Blake3|BLAKE3)'
do
    digest_name=${digest_spec%%:*}
    digest_pattern=${digest_spec#*:}
    digest_count=$(grep_occurrence_count "$digest_pattern" 'crates/**/*.rs')
    if ((digest_count > 0)); then
        digest_families=$((digest_families + 1))
    fi
    digest_summary+=("${digest_name}=${digest_count}")
done
printf 'DIGEST active_token_families=%s %s\n' "$digest_families" "${digest_summary[*]}"

# SURFACE and RUNNING-WORK: repeated nouns that are mechanically detectable.
tool_call_formats=$(grep_line_count '(struct|enum|type)[[:space:]]+ToolCallFormat' 'crates/**/*.rs')
model_configs=$(grep_line_count '(struct|enum|type)[[:space:]]+ModelConfig' 'crates/**/*.rs')
service_mode_kinds=$(grep_line_count '(struct|enum|type)[[:space:]]+Service(Mode|Kind)' 'crates/**/*.rs')
process_mode_kinds=$(grep_line_count '(struct|enum|type)[[:space:]]+Process(Backend|Kind)' 'crates/**/*.rs')
session_types=$(grep_line_count '(struct|enum|type)[[:space:]]+[A-Za-z0-9_]*Session[A-Za-z0-9_]*' 'crates/**/*.rs')
sandbox_types=$(grep_line_count '(struct|enum|trait|type)[[:space:]]+[A-Za-z0-9_]*Sandbox[A-Za-z0-9_]*' 'crates/**/*.rs')
printf 'SURFACE tool_call_format=%s model_config=%s service_mode_kind=%s process_backend_kind=%s\n' \
    "$tool_call_formats" "$model_configs" "$service_mode_kinds" "$process_mode_kinds"
printf 'RUNNING-WORK session_type_declarations=%s sandbox_type_declarations=%s\n' \
    "$session_types" "$sandbox_types"

# OBSERVABILITY: the previously missing metrics / Arrow Flight / OpenTelemetry
# family. Public items show the two crate surfaces; repeated config nouns and
# service fronts show parallel vocabulary; OTel counts cover consumers outside
# those two crates.
public_item_pattern='^[[:space:]]*pub([[:space:]]*\([^)]*\))?[[:space:]]+(struct|enum|trait|type)[[:space:]]+[A-Za-z0-9_]+'
metrics_public=$(grep_line_count "$public_item_pattern" 'crates/hyprstream-metrics/src/**/*.rs')
flight_public=$(grep_line_count "$public_item_pattern" 'crates/hyprstream-flight/src/**/*.rs')
metrics_configs=$(grep_line_count '(struct|enum|type)[[:space:]]+MetricsConfig' 'crates/**/*.rs')
flight_configs=$(grep_line_count '(struct|enum|type)[[:space:]]+FlightConfig' 'crates/**/*.rs')
observability_fronts=$(
    grep_line_count \
        '(struct|enum|trait|type)[[:space:]]+(MetricsService|MetricsStorage|FlightService|FlightSqlServer|TelemetryProvider)' \
        'crates/**/*.rs'
)
otel_pattern='(opentelemetry|tracing_opentelemetry|feature[[:space:]]*=[[:space:]]*"otel"|cfg\(feature[[:space:]]*=[[:space:]]*"otel"\))'
otel_consumer_files=$(
    { git grep -I -l -E "$otel_pattern" "$pinned_sha" -- 'crates/**/*.rs' 'crates/*/Cargo.toml' 2>/dev/null || true; } |
        sed 's/^[^:]*://' |
        awk '$0 !~ /^crates\/hyprstream-metrics\// && $0 !~ /^crates\/hyprstream-flight\//' |
        sort -u |
        awk 'END { print NR + 0 }'
)
otel_consumer_crates=$(
    { git grep -I -l -E "$otel_pattern" "$pinned_sha" -- 'crates/**/*.rs' 'crates/*/Cargo.toml' 2>/dev/null || true; } |
        sed 's/^[^:]*://' |
        awk -F/ '$1 == "crates" && $2 != "hyprstream-metrics" && $2 != "hyprstream-flight" { print $2 }' |
        sort -u |
        awk 'END { print NR + 0 }'
)
printf 'OBSERVABILITY metrics_public_items=%s flight_public_items=%s metrics_config_declarations=%s flight_config_declarations=%s service_fronts=%s otel_consumer_files=%s otel_consumer_crates=%s\n' \
    "$metrics_public" "$flight_public" "$metrics_configs" "$flight_configs" \
    "$observability_fronts" "$otel_consumer_files" "$otel_consumer_crates"

printf '\n[R9-shrinkage]\n'

total_loc=$(
    { git grep -I -h -e '^' "$pinned_sha" -- . 2>/dev/null || true; } |
        awk 'END { print NR + 0 }'
)

source_loc=$(
    { git grep -I -h -e '^' "$pinned_sha" -- \
        '*.rs' '*.py' '*.sh' '*.ts' '*.tsx' '*.js' '*.jsx' \
        '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' '*.proto' '*.capnp' '*.wit' \
        2>/dev/null || true; } |
        awk 'END { print NR + 0 }'
)

workspace_members=$(
    git show "${pinned_sha}:Cargo.toml" |
        awk '
            /^members[[:space:]]*=[[:space:]]*\[/ { members = 1 }
            members {
                line = $0
                while (match(line, /"[^"]+"/)) {
                    print substr(line, RSTART + 1, RLENGTH - 2)
                    line = substr(line, RSTART + RLENGTH)
                }
                if (line ~ /\]/) members = 0
            }
        '
)
workspace_crates=$(awk 'NF { count++ } END { print count + 0 }' <<<"$workspace_members")

dead_public_api_candidates=$(grep_line_count '#!?\[allow\(dead_code\)\]' 'crates/**/*.rs')

# A source-only dependency scan gives every checkout the same baseline even when
# cargo-machete/cargo-udeps is unavailable. It is deliberately conservative and
# is reported as "candidates": renamed crates are normalized from `foo-bar` to
# `foo_bar`, but build-system-only/native-link dependencies may need triage.
unused_dependency_candidates=()
while IFS= read -r member; do
    [[ -n $member ]] || continue
    manifest="${member}/Cargo.toml"
    while IFS= read -r dependency; do
        [[ -n $dependency ]] || continue
        dependency_ident=${dependency//-/_}
        if ! git grep -I -q -E \
            "(^|[^A-Za-z0-9_])${dependency_ident}([^A-Za-z0-9_]|$)" \
            "$pinned_sha" -- "${member}/**/*.rs" 2>/dev/null
        then
            unused_dependency_candidates+=("${member#crates/}:${dependency}")
        fi
    done < <(
        git show "${pinned_sha}:${manifest}" |
            awk '
                /^\[/ {
                    section = $0
                    gsub(/[[:space:]]/, "", section)
                    sub(/^\[/, "", section)
                    sub(/\]$/, "", section)
                    flat = (section ~ /(^|\.)(dev-|build-)?dependencies$/)
                    if (section ~ /(^|\.)(dev-|build-)?dependencies\.[A-Za-z0-9_-]+$/) {
                        dependency = section
                        sub(/^.*(dev-|build-)?dependencies\./, "", dependency)
                        print dependency
                    }
                    next
                }
                flat {
                    line = $0
                    sub(/[[:space:]]*#.*/, "", line)
                    if (line ~ /^[[:space:]]*[A-Za-z0-9_-]+[[:space:]]*=/) {
                        sub(/^[[:space:]]*/, "", line)
                        sub(/[[:space:]]*=.*$/, "", line)
                        print line
                    }
                }
            ' |
            sort -u
    )
done <<<"$workspace_members"
unused_dependency_count=${#unused_dependency_candidates[@]}

generated_modules=$(
    grep_occurrence_count '[A-Za-z0-9_]+_capnp\.rs' 'crates/**/*.rs'
)
generated_consumers=$(
    grep_line_count '[A-Za-z0-9_]+_capnp::' 'crates/**/*.rs'
)
schema_loc=$(
    { git grep -I -h -e '^' "$pinned_sha" -- '*.capnp' 2>/dev/null || true; } |
        awk 'END { print NR + 0 }'
)
generated_ratio=$(ratio "$generated_modules" "$generated_consumers")

printf 'total_loc=%s\n' "$total_loc"
printf 'total_source_loc=%s\n' "$source_loc"
printf 'workspace_crate_count=%s\n' "$workspace_crates"
printf 'dead_public_api_candidates=%s method=allow_dead_code_suppression_proxy\n' \
    "$dead_public_api_candidates"
printf 'unused_dependency_candidates=%s method=pinned_static_cargo_machete_class\n' "$unused_dependency_count"
if ((unused_dependency_count > 0)); then
    printf 'unused_dependency_candidate_list=%s\n' "$(IFS=,; printf '%s' "${unused_dependency_candidates[*]}")"
fi
printf 'generated_schema_loc=%s generated_module_declarations=%s generated_consumer_reference_lines=%s generated_to_consumed_ratio=%s\n' \
    "$schema_loc" "$generated_modules" "$generated_consumers" "$generated_ratio"

if [[ -n ${ONTOLOGY_DEAD_API_REPORT:-} ]]; then
    report=$ONTOLOGY_DEAD_API_REPORT
    [[ -f $report ]] || {
        printf 'error: ONTOLOGY_DEAD_API_REPORT is not a file: %s\n' "$report" >&2
        exit 66
    }
    if ! grep -Fq "$pinned_sha" "$report"; then
        printf 'error: dead-API report does not name pinned SHA %s\n' "$pinned_sha" >&2
        exit 67
    fi
    if command -v sha256sum >/dev/null 2>&1; then
        report_sha256=$(sha256sum "$report" | awk '{ print $1 }')
    else
        report_sha256=$(shasum -a 256 "$report" | awk '{ print $1 }')
    fi
    printf 'dead_api_analyzer_report=%s report_sha256=%s\n' "$report" "$report_sha256"
else
    printf 'dead_api_analyzer_report=not_supplied (set ONTOLOGY_DEAD_API_REPORT to a pinned cargo-machete/cargo-udeps report)\n'
fi
