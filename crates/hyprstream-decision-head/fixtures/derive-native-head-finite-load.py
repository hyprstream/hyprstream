#!/usr/bin/env python3
"""Reproduce archived DEVELOPMENT native head finite-load binding; no tensor IO.
Defaults resolve to the bundled public receipts and exact historical source.
This is not a new head scan, current release validation, or private-log reader.
"""
import argparse
import hashlib
import json
from pathlib import Path

EVIDENCE_PINS = {'clef-production-v4-audit.json': 'aa3513c14e8c2f388b536cb02360d8f921251d22ccb2f6e9abc987fb3383e63e', 'clef-production-profile-native-build-manifest.json': '5a43593f1334e51f4c83330a1fae8c11882eee98c72579e9cea7f7b05a5234e9', 'clef-production-profile-root-build-review.json': 'b01b2a0bdd1288fc595229a837ce080d806fc6da3d06e9d772bce5f0a12556b1', 'clef-production-full-native-source-manifest.json': '15174f668985e29a7875df297e5034680c0a336f9698061a7bc537d4c4e8f49f', 'clef-production-v4-receipts/service-receipt.json': 'c0c855ff0229fdd83cfaf4684f909f74d15c6f7a78dda239db2d9819369ee72a', 'clef-production-v4-receipts/run-result.json': 'ca78f4ebfacf10eab640f23c30e5e512ff163fea90c3a332c035c27105a66e2a', 'clef-production-v4-receipts/setup-receipt.json': '4020d118a35f00e59d3da39d29ed8f20807b4712d0f94e985ae0f611409bbcb9'}
SOURCE_PINS = {'crates/hyprstream-decision-head/src/clef/head.rs': 'bd00449629868f0d8de82ca0ec8e76ee00215adc677f68912d009cb2c2158aed', 'crates/hyprstream-inference/src/runtime/clef_decision.rs': '544daf1ff3f50fc1f318e80da3c3576ed65d84b4a4402946ffeb34ae2384c745'}
EXECUTABLE = "e90affaa7a958b9b1528eae99a6a8882ff02314010cc378491c6195865bdadb5"
COMPOSITION = "f2a82911ea8b6344db43ee1587760376a795832d74728edca0341ae3deb42a0b"
HEAD = "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0"
BUNDLE = Path(__file__).resolve().parents[3] / "docs/evidence/clef-head-finite-load-20261018"

def require(ok, message):
    if not ok: raise ValueError(message)

def read_pinned(root, relative, expected):
    path = root / relative
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024*1024,
            "bounded public source/evidence required")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, "public input digest mismatch")
    return data

def derive(evidence_root, source_root):
    evidence = {name:json.loads(read_pinned(evidence_root, name, pin)) for name,pin in EVIDENCE_PINS.items()}
    sources = {name:read_pinned(source_root, name, pin).decode() for name,pin in SOURCE_PINS.items()}
    audit = evidence['clef-production-v4-audit.json']
    build = evidence['clef-production-profile-native-build-manifest.json']
    review = evidence['clef-production-profile-root-build-review.json']
    manifest = evidence['clef-production-full-native-source-manifest.json']
    service = evidence['clef-production-v4-receipts/service-receipt.json']
    end = evidence['clef-production-v4-receipts/run-result.json']
    setup = evidence['clef-production-v4-receipts/setup-receipt.json']
    require(all(x['composition_identity'] == COMPOSITION for x in [audit,build,review,manifest,end,setup]), 'composition binding differs')
    require(all(x['executable_sha256'] == EXECUTABLE for x in [audit,build,review,setup]) and end['binary_sha256'] == EXECUTABLE, 'executable binding differs')
    require(review['root_build_reviewed'] is True and build['build_exit'] == 0 and build['test_list_exit'] == 0, 'build evidence not successful')
    require(review['build_manifest_sha256'] == EVIDENCE_PINS['clef-production-profile-native-build-manifest.json'], 'build review digest differs')
    for name,pin in SOURCE_PINS.items():
        require(manifest['files'][name] == pin and build['source_pre_sha256'][name] == pin
                and build['source_post_sha256'][name] == pin, 'checked constructor source differs')
    require(audit['actual_exit'] == end['actual_exit'] == 0 and not end['timeout_exit'], 'archived execution not clean')
    require(audit['service_receipt_sha256'] == end['service_receipt_sha256'] == EVIDENCE_PINS['clef-production-v4-receipts/service-receipt.json'], 'executed service digest differs')
    require(service['result'] == 'AUTHENTICATED_CPU_SERVICE_CASES_PASS', 'service cases did not pass')
    head_source = sources['crates/hyprstream-decision-head/src/clef/head.rs']
    runtime = sources['crates/hyprstream-inference/src/runtime/clef_decision.rs']
    require('if expected.len() != weights.len()' in head_source and 'for (name, shape) in expected' in head_source
            and '.get(&name)' in head_source and 'if t.isfinite().all().int64_value(&[]) != 1' in head_source
            and 'Ok(Self { config, weights })' in head_source, 'complete checked constructor path differs')
    require('JointSchemaHead::from_named_tensors(config, weights)' in runtime
            and 'joint head validation: {e}' in runtime, 'production checked constructor call differs')
    provenances = []
    def visit(value):
        if isinstance(value, dict):
            if 'headWeightsSha256' in value:
                provenances.append({key:value[key] for key in ['headWeightsSha256','headConfigSha256','headTensorCount']})
            for child in value.values(): visit(child)
        elif isinstance(value,list):
            for child in value: visit(child)
    visit(service)
    require(len(provenances) == 5 and all(p == dict(headWeightsSha256=HEAD,
            headConfigSha256='77efe959a38b5b17b241543e129e695f3c77465ece55a25985279bd8176279a0',
            headTensorCount=122) for p in provenances), 'head identity/count differs')
    return dict(schema='clef-archived-native-head-finite-load-evidence/1',
        scope='ARCHIVED DEVELOPMENT LOAD-TIME EVIDENCE; not standalone head scan or current release qualification',
        model_commit=service['artifact_commit'], model_tree=service['artifact_tree'],
        executable_sha256_recorded=EXECUTABLE, executable_rehashed_by_this_procedure=False,
        source_composition_identity=COMPOSITION, checked_source_sha256=SOURCE_PINS,
        public_evidence_sha256=EVIDENCE_PINS, head_provenance=provenances[0],
        observed_matching_head_provenance_records=len(provenances), actual_archived_exit=end['actual_exit'],
        actual_service_result=service['result'],
        finite_load_basis=['Production Clef loader propagates checked JointSchemaHead constructor failure.',
                          'Constructor rejects cardinality/name/shape/dtype mismatches and checks isfinite().all() on EVERY expected head tensor before returning Ok.',
                          'Successful executed source-bound service receipt reports122 tensors for exact19cd artifact.'],
        derivation='python3 derive-native-head-finite-load.py --evidence-root <retained-public-evidence> --source-root <frozen-f2a82911-source> --output <new-receipt.json>',
        exclusions=['No tensor/model execution performed by receipt derivation.',
                    '200-scale audit covers only scales and cannot prove head finiteness.',
                    'Not whole-backbone finiteness, standalone raw head scan, accuracy/calibration, current-main or new-candidate qualification.',
                    'Executable digest comes from mutually bound archived build/review/setup/run/audit receipts; executable bytes are not freshly hashed here.'])

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-root',type=Path,default=BUNDLE / 'public-receipts')
    p.add_argument('--source-root',type=Path,default=BUNDLE / 'source-snapshot')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=derive(a.evidence_root,a.source_root)
    with a.output.open('x') as stream: stream.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
