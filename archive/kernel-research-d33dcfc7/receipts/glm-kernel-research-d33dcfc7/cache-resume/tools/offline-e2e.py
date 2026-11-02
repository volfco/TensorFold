#!/usr/bin/env python3
"""CPU-only mocked HTTP exercise of the actual recorder, replayer and audit; no sockets or GPU calls."""
import argparse, hashlib, importlib.util, json, pathlib, tempfile, urllib.parse
spec = importlib.util.spec_from_file_location('cache_reuse', pathlib.Path(__file__).with_name('cache-reuse.py'))
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
p = argparse.ArgumentParser(); p.add_argument('--source', default=str(pathlib.Path.home() / 'tensorfold-kernel-night/candidate/tools/zig/reuse_served.py')); a = p.parse_args()
phase, chats = 'warm', 0
class Reply:
    status = 200
    def __init__(self, obj): self.data = json.dumps(obj).encode()
    def __enter__(self): return self
    def __exit__(self, *unused): pass
    def read(self): return self.data

def urlopen(request, timeout):
    global chats
    assert timeout == 600
    body = json.loads(request.data)
    if urllib.parse.urlparse(request.full_url).path == '/v1/tokenize':
        return Reply({'tokens': list(range(2050 if body['add_generation_prompt'] else 2049))})
    assert urllib.parse.urlparse(request.full_url).path == '/v1/chat/completions'
    assert body['temperature'] == 0 and body['seed'] == 1729 and body['max_tokens'] == 32
    cached = 0 if phase == 'fresh' or chats == 0 else 2048
    chats += 1
    return Reply({'tensorfold': {'token_sha256': hashlib.sha256(json.dumps(body['messages']).encode()).hexdigest()},
                  'usage': {'prompt_tokens': 2050, 'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': cached}},
                  'choices': [{'finish_reason': 'stop', 'message': {'content': 'Fixture accepted.'}}]})
h.urllib.request.urlopen = urlopen
with tempfile.TemporaryDirectory(prefix='cache-offline-e2e-') as name:
    root = pathlib.Path(name)
    args = argparse.Namespace(source=a.source,
                              out=str(root / 'record'), cases=str(root / 'cases.jsonl'), base='http://offline.invalid', model='kernel-research')
    assert h.record(args) == 0
    for mode in ['vector', 'n256', 'fusion']:
        phase, chats = 'warm', 0; args.out = str(root / (mode + '-warm'))
        assert h.replay(args) == 0
        warm = pathlib.Path(args.out) / 'responses.jsonl'
        phase, chats = 'fresh', 0; args.out = str(root / (mode + '-fresh'))
        assert h.replay(args) == 0
        report = h.audit(args.cases, warm, pathlib.Path(args.out) / 'responses.jsonl')
        assert report['passed'], report
    wire = list(root.rglob('wire-*.json'))
    assert wire and all(json.loads(x.read_text())['status'] == 200 for x in wire)
    print(json.dumps({'offline_mocked_recorder_replay_audit': 'pass', 'three_modes': True,
                      'raw_wire_responses': len(wire), 'gpu_work_started': False}))
