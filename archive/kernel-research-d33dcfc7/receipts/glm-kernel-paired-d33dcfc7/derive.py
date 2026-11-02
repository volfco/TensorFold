#!/usr/bin/env python3
"""Build the reviewed, data-only receipt from its exact retained raw archive."""
import argparse
import hashlib
import json
import pathlib
import posixpath
import tarfile

RAW_SHA = 'fff265110c75e063a200fe54ab78b64f00eab9b8aff6b061b1b04ef9a4a2928e'
CAMPAIGN = 'paired-20261030-r2'
MODES = ['original', 'off1', 'vector', 'off2', 'n256', 'off3', 'fusion', 'off4']
TEXT_HASHES = {
    '7816ec236a6680ce49c784a65cb6635e9adba8c76575a26e4a537f75d8a29315': 'gpu_explanation',
    'd95ff51b06239e367f792e84596640426351be0272c27440b721b4142bb9d4f2': 'lru_cache',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive', type=pathlib.Path)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    if sha(args.archive.read_bytes()) != RAW_SHA:
        raise ValueError('archive is not the privacy-reviewed raw snapshot')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    generated = []
    inputs = {}
    with tarfile.open(args.archive) as tar:
        def get(name):
            data = tar.extractfile(name).read()
            inputs[name] = sha(data)
            return data

        def obj(name):
            return json.loads(get(name))

        def lines(name):
            return [json.loads(x) for x in get(name).splitlines()]

        def put(name, value, jsonl=False):
            path = out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            text = ''.join(json.dumps(x, sort_keys=True, allow_nan=False) + '\n' for x in value) if jsonl else json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
            path.write_text(text)
            generated.append(name)

        manifest = get(CAMPAIGN + '/SHA256SUMS')
        checked = 0
        for line in manifest.decode().splitlines():
            expected, rel = line.split('  ', 1)
            name = posixpath.normpath(CAMPAIGN + '/' + rel)
            if sha(tar.extractfile(name).read()) != expected:
                raise ValueError('raw manifest mismatch: ' + name)
            checked += 1
        if checked != 1558:
            raise ValueError('unexpected raw manifest size')
        samples, audits, statuses = [], [], {}
        request_sets = []
        for mode in MODES:
            prefix = CAMPAIGN + '-' + mode
            complete = obj(prefix + '/complete.json')
            if complete.get('matrix_exit') != 0 or not complete.get('cleanup_cleared') or complete['audit']['failures']:
                raise ValueError('mode was not qualified: ' + mode)
            statuses[mode] = {k: complete[k] for k in ['variant', 'flags', 'started', 'finished', 'matrix_exit', 'cleanup_cleared', 'audit']}
            for row in lines(prefix + '/runs.jsonl'):
                keep = ['request', 'sha', 'draft', 'temperature', 'cached', 'wall', 'ttft', 'decode_tps', 'phase', 'length', 'rep', 'warmup', 'usage']
                samples.append({'mode': mode, **{k: row[k] for k in keep if k in row}})
            for row in lines(prefix + '/audit/responses.jsonl'):
                keep = ['request', 'sha', 'complete', 'status']
                audits.append({'mode': mode, **{k: row[k] for k in keep if k in row}})
            posted = lines(prefix + '/audit/requests.jsonl')
            hashes = set()
            for row in posted:
                if row['method'] != 'POST':
                    raise ValueError('unexpected request method')
                body = get(prefix + '/audit/' + row['sha256'] + '.body')
                if sha(body) != row['sha256'] or len(body) != row['bytes']:
                    raise ValueError('request body audit mismatch')
                hashes.add(row['sha256'])
            request_sets.append(hashes)
        if any(x != request_sets[0] for x in request_sets) or len(request_sets[0]) != 16:
            raise ValueError('mode request bodies differ')
        catalog = []
        for digest in sorted(request_sets[0]):
            data = get(CAMPAIGN + '-original/audit/' + digest + '.body')
            body = json.loads(data)
            prompt = body['prompt']
            if isinstance(prompt, str):
                fixture = TEXT_HASHES[sha(prompt.encode())]
                kind = 'text'
            else:
                if not isinstance(prompt, list) or not all(type(x) is int for x in prompt):
                    raise ValueError('unreviewed prompt representation')
                fixture, kind = 'synthetic_repeated_paragraph', 'token_ids'
            params = {k: body[k] for k in ['max_tokens', 'temperature', 'seed', 'draft', 'ignore_eos', 'stream', 'stream_options', 'top_k', 'top_p'] if k in body}
            catalog.append({'request': digest, 'bytes': len(data), 'fixture': fixture, 'prompt_kind': kind, 'prompt_length': len(prompt), 'parameters': params})
        put('samples.jsonl', samples, True)
        put('audited-responses.jsonl', audits, True)
        put('requests.json', catalog)
        put('modes.json', statuses)
        put('identities.json', obj(CAMPAIGN + '/identities.json'))
        put('dispatch-evidence.json', obj(CAMPAIGN + '/dispatch-evidence.json'))
        for experiment in ['kda', 'n256', 'fusion']:
            put('brackets/' + experiment + '.json', obj(CAMPAIGN + '/preliminary-' + experiment + '/comparison.json'))
        provenance = {
            'raw_archive_sha256': RAW_SHA,
            'raw_manifest_sha256': sha(manifest),
            'raw_manifest_files_verified': checked,
            'derive_code_sha256': sha(pathlib.Path(__file__).read_bytes()),
            'raw_inputs_sha256': inputs,
            'derived_files_sha256': {name: sha((out / name).read_bytes()) for name in generated},
            'transform': 'Allowlist numeric timing/usage/cache metadata and complete request/token hashes; omit prompts, generated text, process/network/path metadata and operational authorizations. Bracket reports retain original sample arrays and input hashes.'
        }
        (out / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
        print(json.dumps({'raw_files_verified': checked, 'sample_rows': len(samples), 'audit_rows': len(audits), 'request_bodies': len(catalog), 'derived_files': len(generated) + 1}))


if __name__ == '__main__':
    main()
