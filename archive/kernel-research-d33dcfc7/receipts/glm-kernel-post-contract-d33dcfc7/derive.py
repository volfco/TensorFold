#!/usr/bin/env python3
"""Extract reviewed logs and classify the completed post-contract raw snapshot."""
import argparse
import collections
import hashlib
import json
import pathlib
import posixpath
import re
import tarfile

RAW_SHA = 'c6f75fe174cb042aaf8c51f9e46bd18ef8f5297eb7b1d431ab3b5012080be9d4'
PREFIX = 'paired-20261030-r2-post-contract/'
RESEARCH_SYMBOLS = {
    *('tf_exl3_grouped_' + str(i) for i in range(3)),
    'tf_exl3_gateup_epilogue', 'tf_exl3_down_combine',
    'tf_exl3_grouped_256_1', 'tf_exl3_grouped_256_small_1', 'tf_exl3_group_epi',
    *('tf_exl3_fused_' + part + '_' + str(i) for part in ['gu', 'down'] for i in range(3)),
    'tf_glm_kda_step', 'tf_glm_kda_step_vector8', 'tf_glm_kda_step_vector16',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def classify(live, full):
    missing = collections.Counter(line.split()[0] for line in live if 'no compiled usage recorded' in line)
    resource_reds = [x for x in live if x.startswith('FAIL limits live ')]
    known = [x for x in full if x.startswith('KNOWN ')]
    accepted = [x.split(': ', 1)[0].replace('KNOWN', 'FAIL', 1) + ': ' for x in known]
    controls = []
    for line in full:
        if line.startswith('CONTROL ') and ': red at ' in line:
            name, tail = line[len('CONTROL '):].rsplit(': red at ', 1)
            gate = tail.split(',', 1)[0]
            accepted.append('FAIL ' + gate + ' ' + name + ': ')
            controls.append({'name': name, 'gate': gate})
    unknown = [x for x in full if x.startswith('FAIL ') and not x.startswith(('FAIL limits', 'FAIL kernel-contract'))
               and not any(x.startswith(prefix) for prefix in accepted)]
    research = {}
    for line in full:
        if not line.startswith(('PASS ', 'N/A ')):
            continue
        status, gate, rest = line.split(maxsplit=2)
        name = rest.split(': ', 1)[0]
        if name.startswith(('cuda EXL3 N256', 'cuda EXL3 fused', 'cuda glm KDA vector')):
            research.setdefault(name, {})[gate] = status
    resource_passes = []
    resource_failures = []
    for line in live:
        match = re.match(r'^(PASS|FAIL) limits live (\S+) (\S+): ', line)
        if match and match[3] in RESEARCH_SYMBOLS:
            (resource_passes if match[1] == 'PASS' else resource_failures).append(line)
    shared = re.search(r'limits live sm_121 most shared bytes a block \(opt-in\): the driver says (\d+)', '\n'.join(live))
    if not shared:
        raise ValueError('actual opt-in shared-memory limit missing')
    max_shared = int(shared[1])
    resource_shape = re.compile(r'^FAIL limits live ([^:]+): (\d+) threads launched, the driver allows (\d+); (\d+) registers, (\d+) \+ (\d+) shared bytes \(recorded (\d+), (\d+)\)$')
    resource_rows = []
    for line in resource_reds:
        match = resource_shape.fullmatch(line)
        if not match:
            raise ValueError('resource mismatch line could not be parsed')
        values = list(map(int, match.groups()[1:]))
        keys = ['launch_threads', 'driver_max_threads', 'actual_registers', 'actual_static_shared', 'dynamic_shared', 'recorded_registers', 'recorded_static_shared']
        row = {'name': match[1], **dict(zip(keys, values))}
        row['fits_hardware'] = row['launch_threads'] <= row['driver_max_threads'] and row['actual_static_shared'] + row['dynamic_shared'] <= max_shared
        resource_rows.append(row)
    summary = [x for x in full if x.startswith('SUMMARY ')]
    pattern = (r'SUMMARY (\d+) kernels: (\d+) pass gates 1-4, (\d+) fail, (\d+) known findings; '
               r'(\d+) controls: (\d+) red as planted, (\d+) missed; (\d+) limit problems; '
               r'gates (\d+) ms, limits (\d+) ms')
    if len(summary) != 1 or not (match := re.fullmatch(pattern, summary[0])):
        raise ValueError('complete numerical SUMMARY missing')
    names = ['kernels', 'passed', 'failed', 'known_findings', 'controls', 'caught', 'missed',
             'limit_problems', 'gates_ms', 'post_gate_components_and_limits_ms']
    totals = dict(zip(names, map(int, match.groups())))
    if totals != dict(zip(names, [446, 446, 0, 6, 321, 321, 0, 183, 2199248, 2178184])):
        raise ValueError('unexpected terminal totals')
    missed = [x for x in full if x.startswith(('CONTROL MISSED', 'CONTROL ERROR'))]
    if sum(missing.values()) != 168 or len(resource_reds) != 15 or len(known) != 6 or len(controls) != 321:
        raise ValueError('classification counts differ')
    if unknown or missed or len(research) != 13 or len(resource_passes) != 17 or resource_failures:
        raise ValueError('research qualification or complete failure classification differs')
    for gates in research.values():
        if any(gates.get(g) != 'PASS' for g in ['nonfinite', 'repeat', 'shapes']):
            raise ValueError('missing research arithmetic gate')
        if gates.get('sums') not in ['PASS', 'N/A']:
            raise ValueError('research sum gate not qualified')
    return {
        'summary': summary[0], 'totals': totals,
        'overall_contract_passed': False, 'known_numerical_findings': known,
        'missing_compiled_usage_by_image': dict(sorted(missing.items())),
        'missing_compiled_usage_total': 168, 'live_resource_mismatch_count': 15,
        'live_resource_mismatches': resource_reds, 'live_resource_details': resource_rows,
        'driver_opt_in_shared_bytes': max_shared,
        'all_live_mismatches_fit_hardware': all(x['fits_hardware'] for x in resource_rows), 'caught_controls': 321,
        'missed_or_error_controls': missed, 'unclassified_numerical_failures': unknown,
        'research_numerical_gates': research,
        'research_controls': [x for x in controls if any(s in x['name'] for s in ['EXL3 N256', 'fused EXL3', 'KDA vector'])],
        'research_resource_passes': resource_passes,
        'research_resource_set': 'Eleven experimental symbols plus six comparison-baseline symbols.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive', type=pathlib.Path)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    if sha(args.archive.read_bytes()) != RAW_SHA:
        raise ValueError('archive is not the reviewed post-contract snapshot')
    args.out.mkdir(parents=True, exist_ok=True)
    inputs, generated = {}, []
    with tarfile.open(args.archive) as tar:
        def get(name):
            data = tar.extractfile(PREFIX + name).read()
            inputs[name] = sha(data)
            return data

        def put(name, value, raw=False):
            data = value if raw else (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
            (args.out / name).write_bytes(data)
            generated.append(name)

        manifest = get('SHA256SUMS')
        count = 0
        for line in manifest.decode().splitlines():
            expected, name = line.split('  ', 1)
            member = posixpath.normpath(PREFIX + name)
            if sha(tar.extractfile(member).read()) != expected:
                raise ValueError('raw manifest mismatch: ' + name)
            count += 1
        if count != 14:
            raise ValueError('unexpected post receipt size')
        live = get('kernel-contract-live.log')
        full = get('kernel-contract.log')
        results = json.loads(get('results.json'))
        runner = json.loads(get('runner.json'))
        if [x['exit'] for x in results] != [1, 1] or get('post-contract.exit').strip() != b'1':
            raise ValueError('terminal red results were changed')
        put('kernel-contract-live.log', live, True)
        put('kernel-contract.log', full, True)
        put('results.json', results)
        put('runner.json', runner)
        put('classification.json', classify(live.decode().splitlines(), full.decode().splitlines()))
        events = [json.loads(x) for x in get('events.jsonl').splitlines()]
        timeline = [{k: x[k] for k in ['time', 'state', 'command', 'exit', 'complete_success'] if k in x}
                    for x in events if x.get('state') in ['running', 'command-finished', 'finished']]
        put('timeline.json', timeline)
        provenance = {
            'raw_archive_sha256': RAW_SHA, 'raw_manifest_sha256': sha(manifest),
            'raw_manifest_files_verified': count, 'raw_inputs_sha256': inputs,
            'derive_code_sha256': sha(pathlib.Path(__file__).read_bytes()),
            'derived_files_sha256': {name: sha((args.out / name).read_bytes()) for name in generated},
            'privacy': 'Reviewed synthetic numerical logs copied verbatim; node/process/interface metadata, PIDs and operational state are omitted.'}
        (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
        print(json.dumps({'raw_files_verified': count, 'kernels_pass': 446, 'known_findings': 6,
                          'controls_caught': 321, 'limit_problems': 183, 'overall_passed': False}))


if __name__ == '__main__':
    main()
