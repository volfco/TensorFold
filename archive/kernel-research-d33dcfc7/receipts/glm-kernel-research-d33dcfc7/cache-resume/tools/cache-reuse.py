#!/usr/bin/env python3
"""Four frozen chat requests: genuine warm-prefix reuse versus fresh cache-off output, with strict evidence."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request


def encoded(value):
    return json.dumps(value).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def put(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def jsonl(path):
    return [json.loads(x) for x in pathlib.Path(path).read_text().splitlines() if x.strip()]


def index(rows):
    result = {}
    for row in rows:
        if type(row.get('i')) is not int or row['i'] in result:
            raise ValueError('missing or duplicate row identity')
        result[row['i']] = row
    return result


def valid_reply(got):
    return (isinstance(got, dict) and not got.get('error') and
            isinstance(got.get('sha'), str) and re.fullmatch('[0-9a-f]{64}', got['sha']) is not None and
            type(got.get('cached')) is int and type(got.get('prompt')) is int and got['prompt'] > 0 and
            type(got.get('tokens')) is int and 0 < got['tokens'] <= 32)


def fits(value, allowed):
    if isinstance(allowed, list):
        return value in allowed
    return isinstance(allowed, dict) and allowed.get('min', 1) <= value <= allowed.get('max', -1)


def audit(cases_path, warm_path, fresh_path):
    cases, warm, fresh = index(jsonl(cases_path)), index(jsonl(warm_path)), index(jsonl(fresh_path))
    problems, pairs = [], {}
    if set(cases) != {0, 1, 2, 3} or set(warm) != set(cases) or set(fresh) != set(cases):
        problems.append('incomplete four-request matrix')
    for i, case in sorted(cases.items()):
        if i not in warm or i not in fresh:
            continue
        w, f, original = warm[i].get('got'), fresh[i].get('got'), case.get('got')
        for name, got in [('warm', w), ('fresh', f), ('canonical', original)]:
            if not valid_reply(got):
                problems.append(f'{i}: {name} has invalid counters, missing full digest, empty reply or error')
        if not all(valid_reply(x) for x in [w, f, original]):
            continue
        ids = case.get('prompt_ids')
        if (not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids) or
                not 2049 <= len(ids) <= 8192 or len(ids) != w['prompt'] or len(ids) != f['prompt'] or len(ids) != original['prompt']):
            problems.append(f'{i}: rendered prompt IDs or prompt counters differ')
        body = case['body']
        body_sha = digest(body)
        if case.get('body_sha256') != body_sha:
            problems.append(f'{i}: canonical request-body hash mismatch')
        for name, row in [('warm', warm[i]), ('fresh', fresh[i])]:
            if row.get('body_sha256') != body_sha or row.get('prompt_ids') != ids:
                problems.append(f'{i}: {name} did not use the frozen body and rendered IDs')
        if f['cached'] != 0:
            problems.append(f'{i}: fresh reply reused a prefix')
        if i == 0 and w['cached'] != 0:
            problems.append(f'{i}: initial warm-server request was not cold')
        if i > 0 and not (0 < w['cached'] <= case['history'] and fits(w['cached'], case['allowed'])):
            problems.append(f'{i}: expected resumed prefix was absent or outside its allowed history')
        if not (w['sha'] == f['sha'] == original['sha']):
            problems.append(f'{i}: warm/fresh/canonical full token hashes differ')
        if not (w['tokens'] == f['tokens'] == original['tokens']):
            problems.append(f'{i}: completion-token counts differ')
        if (body.get('temperature') != 0 or body.get('seed') != 1729 or body.get('max_tokens') != 32 or
                body.get('chat_template_kwargs', {}).get('enable_thinking') is not False):
            problems.append(f'{i}: request sampling settings changed')
        key = digest({k: v for k, v in body.items() if k != 'draft'})
        pairs.setdefault(key, []).append((case['arm'], w['sha'], f['sha']))
    if len(pairs) != 2 or any(sorted(x[0] for x in group) != ['drafted', 'plain'] or
                              len({x[1] for x in group} | {x[2] for x in group}) != 1 for group in pairs.values()):
        problems.append('drafted/plain request pairs or full token equality are missing')
    return {'passed': not problems, 'requests': len(cases), 'positive_warm_reuses_required': 3,
            'fresh_zero_counters_required': 4, 'problems': problems}


def load_reuse(path):
    spec = importlib.util.spec_from_file_location('tensorfold_reuse', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def client(module, base, model, out):
    class Audited(module.Client):
        counter = 0

        def post(self, path, body):
            number = self.counter
            self.counter += 1
            payload = encoded(body)
            stem = out / f'wire-{number:04d}'
            stem.with_suffix('.request').write_bytes(payload)
            started = time.time()
            status, response, error = None, b'', None
            try:
                request = urllib.request.Request(self.base + path, data=payload,
                                                 headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=600) as r:
                    status, response = r.status, r.read()
                if status != 200:
                    raise RuntimeError('non-200 response')
                return json.loads(response)
            except urllib.error.HTTPError as e:
                status, response, error = e.code, e.read(), repr(e)
                raise
            except Exception as e:
                error = repr(e)
                raise
            finally:
                stem.with_suffix('.response').write_bytes(response)
                put(stem.with_suffix('.json'), {'path': path, 'status': status, 'started': started,
                    'finished': time.time(), 'request_sha256': hashlib.sha256(payload).hexdigest(),
                    'response_sha256': hashlib.sha256(response).hexdigest(), 'error': error})
    return Audited(base, model)


def record(args):
    module = load_reuse(args.source)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    c = client(module, args.base, args.model, out)

    class Seeded(module.Conv):
        def body(self, model, max_tokens, plain):
            result = super().body(model, max_tokens, plain)
            result['seed'] = 1729
            count = len(c.tokens(result['messages'], result.get('tools'), False, True))
            if not 2049 <= count <= 8192:
                raise RuntimeError(f'prompt outside bounded envelope before serving: {count}')
            return result

    guide = 'Public deterministic cache validation. Read the supplied fixture and answer briefly.\n' + ''.join(
        f'Public record {i}: alpha beta gamma delta; preserve order and report a short summary.\n' for i in range(180))
    conv = Seeded('R', guide, 'Explain the public records in one sentence.', module.TOOLS, False, False)
    first = c.tokens(conv.messages, conv.tools, False, True)
    if not 2049 <= len(first) <= 8192:
        raise RuntimeError(f'fixture outside registered bounded envelope: {len(first)} tokens')
    rec = module.Recorder(c, str(out / 'record.raw.jsonl'), 32, planned=True, min_prompt=0)
    prompts = {}
    try:
        for turn in range(2):
            tool = ('public_fixture.py', '\n'.join(f'value_{i} = {i} + 1' for i in range(120)))
            result = rec.turn(conv, [0] if turn == 0 else rec.prev(conv), tool)
            prompts[turn + 1] = result['prompt']
    finally:
        rec.out.close()
    rows = jsonl(out / 'record.raw.jsonl')
    if len(rows) != 4 or rec.failures:
        raise RuntimeError('cache record failed its allowed-prefix assertions')
    for row in rows:
        row['prompt_ids'] = prompts[row['turn']]
        row['body_sha256'] = digest(row['body'])
    with pathlib.Path(args.cases).open('x') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    with (out / 'responses.jsonl').open('x') as f:
        for row in rows:
            f.write(json.dumps({k: row[k] for k in ['i', 'got', 'prompt_ids', 'body_sha256']}) + '\n')
    return 0


def replay(args):
    module = load_reuse(args.source)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    c = client(module, args.base, args.model, out)
    with (out / 'responses.jsonl').open('x') as f:
        for row in jsonl(args.cases):
            body = row['body']
            if digest(body) != row['body_sha256']:
                raise RuntimeError('frozen request hash mismatch')
            ids = c.tokens(body['messages'], body.get('tools'), False, True)
            if ids != row['prompt_ids']:
                raise RuntimeError('rendered prompt IDs changed')
            got = c.chat(body)
            f.write(json.dumps({'i': row['i'], 'got': got, 'prompt_ids': ids,
                                'body_sha256': digest(body)}) + '\n')
            f.flush()
            if not valid_reply(got):
                raise RuntimeError('invalid served reply')
    return 0


def selftest():
    import copy
    with tempfile.TemporaryDirectory(prefix='cache-resume-selftest-') as root:
        p = pathlib.Path(root)
        cases, warm, fresh = [], [], []
        for i in range(4):
            body = {'messages': [{'role': 'user', 'content': str(i // 2)}], 'temperature': 0,
                    'seed': 1729, 'max_tokens': 32, 'chat_template_kwargs': {'enable_thinking': False}}
            if i % 2:
                body['draft'] = False
            got = {'sha': str(i // 2 + 1) * 64, 'prompt': 2050, 'cached': 0 if i == 0 else 2, 'tokens': 32}
            row = {'i': i, 'arm': 'plain' if i % 2 else 'drafted', 'body': body, 'got': got,
                   'history': 2049, 'allowed': [0] if i == 0 else {'min': 1, 'max': 2049},
                   'prompt_ids': list(range(2050)), 'body_sha256': digest(body)}
            cases.append(row)
            warm.append(copy.deepcopy(row))
            fresh.append(copy.deepcopy(row))
            fresh[-1]['got']['cached'] = 0

        def check(c, w, f):
            for name, rows in [('c', c), ('w', w), ('f', f)]:
                (p / name).write_text(''.join(json.dumps(x) + '\n' for x in rows))
            return audit(p / 'c', p / 'w', p / 'f')['passed']

        assert check(cases, warm, fresh)
        controls = 0
        for value in [None, True, 1]:
            bad = copy.deepcopy(fresh)
            bad[0]['got']['cached'] = value
            assert not check(cases, warm, bad)
            controls += 1
        for field, value in [('cached', 0), ('sha', None), ('tokens', 0), ('tokens', 33), ('prompt', 5)]:
            bad = copy.deepcopy(warm)
            bad[1]['got'][field] = value
            assert not check(cases, bad, fresh)
            controls += 1
        bad = copy.deepcopy(warm)
        bad[1]['body_sha256'] = '0' * 64
        assert not check(cases, bad, fresh)
        controls += 1
        try:
            check(cases, warm + warm[:1], fresh)
            raise AssertionError('duplicate identity was accepted')
        except ValueError:
            controls += 1
    print(json.dumps({'offline_selftest': 'pass', 'positive': 1, 'false_pass_controls_rejected': controls}))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['record', 'replay', 'audit', 'selftest'])
    p.add_argument('--source', default=str(pathlib.Path.home() / 'tensorfold-kernel-night/candidate/tools/zig/reuse_served.py'))
    p.add_argument('--base', default='http://127.0.0.1:18080')
    p.add_argument('--model', default='kernel-research')
    p.add_argument('--out')
    p.add_argument('--cases')
    p.add_argument('--warm')
    p.add_argument('--fresh')
    a = p.parse_args()
    if a.mode == 'selftest':
        return selftest()
    if a.mode == 'audit':
        report = audit(a.cases, a.warm, a.fresh)
        if a.out:
            put(a.out, report)
        print(json.dumps(report))
        return 0 if report['passed'] else 1
    if not a.out or not a.cases:
        p.error('--out and --cases are required')
    return record(a) if a.mode == 'record' else replay(a)


if __name__ == '__main__':
    raise SystemExit(main())
