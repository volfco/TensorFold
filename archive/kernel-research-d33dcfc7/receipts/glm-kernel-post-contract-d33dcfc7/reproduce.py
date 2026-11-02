#!/usr/bin/env python3
"""Verify completed post-contract results without running a GPU command."""
import argparse
import hashlib
import json
import pathlib
import runpy
import subprocess
import sys
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources(identity, revisions, source_root):
    by_path = {item['path']: item for item in revisions['revisions']}
    if len(by_path) != len(revisions['revisions']):
        raise ValueError('duplicate documented source revision')
    for name, revision in by_path.items():
        if revision['kind'] != 'comment-only' or revision['tested_sha256'] != identity['source_sha256'].get(name):
            raise ValueError('revision does not anchor the tested source: ' + name)
        for key in ['old_comment', 'new_comment']:
            text = revision[key]
            if not text.endswith('\n') or not all(line.startswith('//') for line in text.splitlines()):
                raise ValueError('revision contains non-comment bytes: ' + name)
    tested, revised = 0, 0
    for name, expected in identity['source_sha256'].items():
        data = (source_root / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest == expected:
            tested += 1
            continue
        revision = by_path.get(name)
        if revision is None or digest != revision['current_sha256']:
            raise ValueError('observed source changed: ' + name)
        old, new = revision['old_comment'].encode(), revision['new_comment'].encode()
        if data.count(new) != 1 or old in data:
            raise ValueError('documented comment replacement is not unique: ' + name)
        reconstructed = data.replace(new, old, 1)
        if hashlib.sha256(reconstructed).hexdigest() != expected:
            raise ValueError('comment reversal does not reconstruct tested source: ' + name)
        revised += 1
    return tested, revised


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw', type=pathlib.Path, help='also rederive from the unchanged raw archive')
    ap.add_argument('--source-root', type=pathlib.Path, help='also compare the observed source hashes')
    args = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parent
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if sha(root / name) != expected:
            raise ValueError('receipt checksum mismatch: ' + name)
    provenance = json.loads((root / 'provenance.json').read_text())
    if sha(root / 'derive.py') != provenance['derive_code_sha256']:
        raise ValueError('derivation code differs')
    for name, expected in provenance['derived_files_sha256'].items():
        if sha(root / name) != expected:
            raise ValueError('derived data differs: ' + name)
    module = runpy.run_path(str(root / 'derive.py'))
    actual = module['classify']((root / 'kernel-contract-live.log').read_text().splitlines(),
                                (root / 'kernel-contract.log').read_text().splitlines())
    stored = json.loads((root / 'classification.json').read_text())
    if actual != stored:
        raise ValueError('post classification does not reproduce')
    if len(actual['research_controls']) != 5 or not actual['all_live_mismatches_fit_hardware']:
        raise ValueError('research controls or hardware-fit classification differs')
    prior = json.loads((root / 'inherited-resource-evidence.json').read_text())
    if actual['missing_compiled_usage_by_image'] != prior['missing_compiled_usage']:
        raise ValueError('missing-usage classification differs from independent preflight')
    if actual['live_resource_mismatch_count'] != sum(prior['live_mismatches'].values()):
        raise ValueError('live mismatch count differs from independent preflight')
    identity = json.loads((root / 'source-identities.json').read_text())
    runner = json.loads((root / 'runner.json').read_text())
    if runner['sha256'] != identity['runner_sha256'] or runner['sha256'] != identity['paired_model_identities']['contract_runner']:
        raise ValueError('runner identities disagree')
    if sha(root / 'inherited-resource-evidence.json') != identity['auxiliary_resource_evidence']['sha256']:
        raise ValueError('independent inheritance evidence differs')
    if not all(x['match'] for x in identity['prior_source_crosschecks']):
        raise ValueError('prior source cross-check differs')
    revisions = json.loads((root / 'source-revisions.json').read_text())
    if sha(root / 'source-identities.json') != revisions['tested_source_manifest_sha256']:
        raise ValueError('tested source identities changed after qualification')
    if revisions['tested_runner_sha256'] != runner['sha256']:
        raise ValueError('comment revision runner anchor differs')
    if args.source_root:
        tested, revised = check_sources(identity, revisions, args.source_root)
        print(f'PASS source: {tested} exact tested hashes; {revised} documented comment-only revisions reconstruct tested bytes')
    if args.raw:
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run([sys.executable, str(root / 'derive.py'), str(args.raw), '--out', folder], check=True)
            rebuilt = pathlib.Path(folder)
            for name in [*provenance['derived_files_sha256'], 'provenance.json']:
                if (rebuilt / name).read_bytes() != (root / name).read_bytes():
                    raise ValueError('raw derivation differs: ' + name)
        print('PASS raw archive: 14 source entries verified; derivation byte-identical')
    print(actual['summary'])
    print('PASS receipt verification; actual GPU commands both exited 1 and overall contract remains red')


if __name__ == '__main__':
    main()
