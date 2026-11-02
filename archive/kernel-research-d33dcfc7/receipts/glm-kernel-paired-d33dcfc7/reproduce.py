#!/usr/bin/env python3
"""Verify data hashes, independent token parity and all three timing brackets."""
import argparse
import collections
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile

MODES = ['original', 'off1', 'vector', 'off2', 'n256', 'off3', 'fusion', 'off4']
BRACKETS = [('kda', 'off1', 'vector', 'off2'), ('n256', 'off2', 'n256', 'off3'), ('fusion', 'off3', 'fusion', 'off4')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def cells(rows, before, candidate, after):
    modes = [before, candidate, after]
    groups = {}
    for mode in modes:
        by = {}
        for row in rows:
            if row['mode'] != mode or row.get('warmup') or row['phase'] not in ['cold', 'decode']:
                continue
            cell = by.setdefault(row['request'], {
                'phase': row['phase'], 'prompt_tokens': row['usage']['prompt_tokens'],
                'completion_tokens': row['usage']['completion_tokens'], 'draft': row['draft'], 'metrics': {}})
            metrics = ['wall', 'ttft'] if row['phase'] == 'cold' else ['decode_tps', 'wall']
            for metric in metrics:
                value = row.get(metric)
                if isinstance(value, (int, float)):
                    cell['metrics'].setdefault(metric, []).append(value)
        groups[mode] = by
    result = []
    for body, cell in groups[candidate].items():
        for metric in cell['metrics']:
            samples = {mode: groups[mode][body]['metrics'][metric] for mode in modes}
            medians = {mode: statistics.median(values) for mode, values in samples.items()}
            control = (medians[before] + medians[after]) / 2
            controls = samples[before] + samples[after]
            values = samples[candidate]
            ratio = medians[candidate] / control
            stats = {mode: {
                'n': len(v), 'median': medians[mode], 'min': min(v), 'max': max(v),
                'mean': statistics.mean(v), 'stdev': statistics.stdev(v) if len(v) > 1 else 0,
                'samples': v} for mode, v in samples.items()}
            result.append({
                'request': body, **{k: cell[k] for k in ['phase', 'prompt_tokens', 'completion_tokens', 'draft']},
                'metric': metric, 'candidate_over_control': ratio, 'candidate_change_percent': 100 * (ratio - 1),
                'control_drift_percent': 100 * (medians[after] / medians[before] - 1),
                'candidate_relative_range_percent': 100 * (max(values) - min(values)) / medians[candidate],
                'sample_ranges_overlap': max(min(values), min(controls)) <= min(max(values), max(controls)),
                'stats': stats})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw', type=pathlib.Path, help='also derive again from the retained raw archive')
    args = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parent
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if sha(root / name) != expected:
            raise ValueError('receipt checksum mismatch: ' + name)
    provenance = json.loads((root / 'provenance.json').read_text())
    if sha(root / 'derive.py') != provenance['derive_code_sha256']:
        raise ValueError('derivation source differs')
    for name, expected in provenance['derived_files_sha256'].items():
        if sha(root / name) != expected:
            raise ValueError('derived input differs: ' + name)
    rows = jsonl(root / 'samples.jsonl')
    audit = jsonl(root / 'audited-responses.jsonl')
    mode_status = json.loads((root / 'modes.json').read_text())
    reference = None
    for mode in MODES:
        mine = [x for x in rows if x['mode'] == mode]
        records = [x for x in audit if x['mode'] == mode and x.get('complete')]
        pairs = {(x['request'], x['sha']) for x in records}
        if len(mine) != 46 or len(records) != 46 or any((x['request'], x['sha']) not in pairs for x in mine):
            raise ValueError('saved/independent response coverage differs: ' + mode)
        hashes = collections.defaultdict(set)
        for row in records:
            hashes[row['request']].add(row['sha'])
        hashes = dict(hashes)
        if len(hashes) != 15 or any(len(v) != 1 for v in hashes.values()):
            raise ValueError('within-body mismatch: ' + mode)
        if reference is None:
            reference = hashes
        elif hashes != reference:
            raise ValueError('between-mode token mismatch: ' + mode)
        if any(row.get('cached') != 0 for row in mine):
            raise ValueError('fresh cache count is not zero: ' + mode)
        status = mode_status[mode]
        if status['matrix_exit'] != 0 or not status['cleanup_cleared'] or status['audit']['failures']:
            raise ValueError('mode status failed: ' + mode)
    for name, before, candidate, after in BRACKETS:
        stored = json.loads((root / 'brackets' / (name + '.json')).read_text())
        calculated = cells(rows, before, candidate, after)
        if calculated != stored['cells']:
            raise ValueError('bracket does not reproduce: ' + name)
        deltas = [x['candidate_change_percent'] for x in calculated if x['metric'] == 'decode_tps']
        print(name + ' decode TPS changes (%): ' + ', '.join(f'{x:+.3f}' for x in deltas))
    if args.raw:
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run([sys.executable, str(root / 'derive.py'), str(args.raw), '--out', folder], check=True)
            rebuilt = pathlib.Path(folder)
            for name in [*provenance['derived_files_sha256'], 'provenance.json']:
                if (rebuilt / name).read_bytes() != (root / name).read_bytes():
                    raise ValueError('raw derivation does not reproduce: ' + name)
        print('PASS raw archive: 1558 source entries verified; derivation byte-identical')
    print('PASS paired model receipt: 8 modes, 368 saved/independent responses, 15 matching bodies, 3 exact brackets')


if __name__ == '__main__':
    main()
