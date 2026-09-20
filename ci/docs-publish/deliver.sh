#!/usr/bin/env bash
# Hand the packed docs CAR to the cyberdione/ingest promotion pipeline.
#
# Required env: GITLAB_PACKAGE_TOKEN (GitLab PAT, api scope — used for the
# generic-package upload and for polling pipeline status) and
# GITLAB_TRIGGER_TOKEN (pipeline trigger token for cyberdione/ingest).
# The workflow only invokes this script when both are non-empty.
#
# Steps:
#   1. PUT ipfs/{site.car,manifest.json} to the GitLab generic package
#      registry of cyberdione/ingest, package `hyprstream-docs`, version =
#      the root CID (the package version IS the content address).
#   2. Trigger the ingest promotion pipeline, passing HYPRSTREAM_DOCS_CID
#      and HYPRSTREAM_DOCS_PACKAGE_VERSION.
#   3. Poll the triggered pipeline until it concludes (15-minute budget);
#      fail this job if promotion fails or times out.
#
# Tokens never appear on a command line: curl reads the auth header from a
# 0600 config file and the trigger token from a 0600 JSON payload file, both
# removed on exit. All curl calls are --fail --show-error --silent.
set -euo pipefail

GITLAB_API="https://gitlab.com/api/v4"
PROJECT="cyberdione%2Fingest"
PACKAGE_NAME="hyprstream-docs"
POLL_INTERVAL_SECONDS=30
POLL_DEADLINE_SECONDS=900

: "${GITLAB_PACKAGE_TOKEN:?must be set (workflow guards on it)}"
: "${GITLAB_TRIGGER_TOKEN:?must be set (workflow guards on it)}"

CID="$(tr -d '[:space:]' < ipfs/cid.txt)"
case "$CID" in
  baf*) ;;
  *) echo "ERROR: ipfs/cid.txt does not contain a CIDv1 root" >&2; exit 1 ;;
esac
VERSION="$CID"

workdir="$(mktemp -d)"
chmod 700 "$workdir"
trap 'rm -rf "$workdir"' EXIT

auth_config="$workdir/curl-auth.conf"
printf 'header = "PRIVATE-TOKEN: %s"\n' "$GITLAB_PACKAGE_TOKEN" > "$auth_config"
chmod 600 "$auth_config"

echo "Uploading CAR + manifest as generic package $PACKAGE_NAME@$VERSION ..."
for file in site.car manifest.json; do
  curl --config "$auth_config" \
    --fail --show-error --silent \
    --upload-file "ipfs/$file" \
    "$GITLAB_API/projects/$PROJECT/packages/generic/$PACKAGE_NAME/$VERSION/$file" \
    > /dev/null
  echo "  uploaded $file"
done

payload="$workdir/trigger-payload.json"
chmod 600 "$payload"
node -e '
  const [token, cid, version] = process.argv.slice(1);
  process.stdout.write(JSON.stringify({
    token,
    ref: "main",
    variables: {
      HYPRSTREAM_DOCS_CID: cid,
      HYPRSTREAM_DOCS_PACKAGE_VERSION: version,
    },
  }));
' "$GITLAB_TRIGGER_TOKEN" "$CID" "$VERSION" > "$payload"

echo "Triggering cyberdione/ingest promotion pipeline ..."
response="$workdir/trigger-response.json"
curl --fail --show-error --silent \
  -X POST -H "Content-Type: application/json" --data @"$payload" \
  "$GITLAB_API/projects/$PROJECT/trigger/pipeline" \
  > "$response"

PIPELINE_ID="$(node -e '
  let d = "";
  process.stdin.on("data", (c) => (d += c)).on("end", () => {
    const j = JSON.parse(d);
    if (!j.id) process.exit(1);
    process.stdout.write(String(j.id));
  });
' < "$response")"
echo "Triggered pipeline $PIPELINE_ID; polling (deadline ${POLL_DEADLINE_SECONDS}s) ..."

elapsed=0
while :; do
  status="$(curl --config "$auth_config" \
    --fail --show-error --silent \
    "$GITLAB_API/projects/$PROJECT/pipelines/$PIPELINE_ID" \
    | node -e '
      let d = "";
      process.stdin.on("data", (c) => (d += c)).on("end", () => {
        const j = JSON.parse(d);
        if (!j.status) process.exit(1);
        process.stdout.write(j.status);
      });
    ')"
  echo "  pipeline $PIPELINE_ID: $status (${elapsed}s elapsed)"
  case "$status" in
    success)
      echo "Promotion pipeline succeeded for CID $CID"
      exit 0
      ;;
    failed|canceled|skipped)
      echo "ERROR: promotion pipeline concluded with status '$status'" >&2
      exit 1
      ;;
  esac
  if (( elapsed >= POLL_DEADLINE_SECONDS )); then
    echo "ERROR: promotion pipeline did not conclude within ${POLL_DEADLINE_SECONDS}s (last status '$status')" >&2
    exit 1
  fi
  sleep "$POLL_INTERVAL_SECONDS"
  elapsed=$((elapsed + POLL_INTERVAL_SECONDS))
done
