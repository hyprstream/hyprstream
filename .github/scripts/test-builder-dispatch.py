#!/usr/bin/env python3
"""Exercise dispatch identity and the real publication shell with fake podman."""
import os
from pathlib import Path
import subprocess
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[2]
workflow = yaml.load((ROOT / '.github/workflows/build-image.yml').read_text(), Loader=yaml.BaseLoader)
steps = workflow['jobs']['publish-arm64-builder']['steps']
by_name = {step.get('name'): step for step in steps}
assert workflow['jobs']['publish-arm64-builder']['runs-on'][0] == 'self-hosted'
assert steps[0]['with']['persist-credentials'] == 'false'
assert steps[1]['id'] == 'dispatch'
assert workflow['on']['workflow_dispatch']['inputs']['candidate']['default'] == 'true'
for name in ['Rewrite the canonical builder-image pin if the digest changed', 'Check for the bump PAT', 'Open digest-bump PR']:
    assert "steps.dispatch.outputs.mode == 'release'" in by_name[name]['if']
sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
count = 0
with tempfile.TemporaryDirectory() as tmp:
    temp = Path(tmp)
    base = dict(os.environ, GITHUB_SHA=sha, GITHUB_REF='refs/heads/codex/pr1523-candidate',
                GITHUB_EVENT_NAME='workflow_dispatch', EXPECTED_SOURCE_SHA=sha, CANDIDATE='true',
                GITHUB_OUTPUT=str(temp / 'output'))
    def dispatch(passes, mode='candidate', **updates):
        global count
        output = temp / 'output'
        output.unlink(missing_ok=True)
        result = subprocess.run(['bash', '.github/scripts/builder-dispatch.sh'], cwd=ROOT,
                                env=dict(base, **updates), capture_output=True, text=True)
        assert (result.returncode == 0) == passes, result.stderr
        if passes:
            assert f'mode={mode}\nsource_sha={sha}\n' == output.read_text()
        else:
            assert not output.exists()
        count += 1
    dispatch(True)
    dispatch(False, EXPECTED_SOURCE_SHA='')
    dispatch(False, EXPECTED_SOURCE_SHA='abcd')
    dispatch(False, EXPECTED_SOURCE_SHA='0' * 40)
    dispatch(False, EXPECTED_SOURCE_SHA=sha + '\n')
    dispatch(False, EXPECTED_SOURCE_SHA='$(touch forbidden)')
    dispatch(False, GITHUB_SHA='0' * 40)
    dispatch(False, CANDIDATE='invalid')
    dispatch(False, CANDIDATE='false')
    dispatch(True, mode='release', CANDIDATE='false', GITHUB_REF='refs/heads/main')
    dispatch(True, mode='release', GITHUB_EVENT_NAME='schedule', GITHUB_REF='refs/heads/main')
    dispatch(False, GITHUB_EVENT_NAME='pull_request')
    # Execute the actual workflow publication body. Only podman is substituted.
    podman = temp / 'podman'
    podman.write_text('''#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$CALLS"
if [[ "${1:-}" == push && "${2:-}" == --digestfile ]]; then
 printf 'sha256:%064d\\n' 0 > "$3"
fi
''')
    podman.chmod(0o755)
    for mode in ['candidate', 'release']:
        calls = temp / 'calls'
        calls.unlink(missing_ok=True)
        env = dict(base, PATH=f'{temp}:{os.environ["PATH"]}', MODE=mode,
                   IMAGE='ghcr.io/hyprstream/rust-builder-arm64',
                   BUILD_IMAGE=f'ghcr.io/hyprstream/rust-builder-arm64:{mode}-fixture',
                   RUNNER_TEMP=tmp, GITHUB_RUN_ID='1', GITHUB_RUN_ATTEMPT='1',
                   GITHUB_STEP_SUMMARY=str(temp / 'summary'), CALLS=str(calls))
        subprocess.run(['bash', '-euo', 'pipefail', '-c', by_name['Push validated image']['run']],
                       env=env, cwd=ROOT, check=True)
        recorded = calls.read_text().splitlines()
        assert len(recorded) == (1 if mode == 'candidate' else 3), recorded
        assert (':latest' in calls.read_text()) == (mode == 'release'), recorded
        count += 1
print(f'builder workflow fixtures: {count} PASS')
