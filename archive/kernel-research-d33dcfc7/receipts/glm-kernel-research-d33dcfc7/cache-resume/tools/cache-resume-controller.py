#!/usr/bin/env python3
"""Plan by default; run three owned cache-on/cache-off pairs only after explicit post-contract release."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

MODES = {'vector': (1, 0, 0), 'n256': (0, 1, 0), 'fusion': (0, 0, 1)}
TIMING_MODES = ['original', 'off1', 'vector', 'off2', 'n256', 'off3', 'fusion', 'off4']
STOP = False


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def prerequisites(root, campaign, run_id):
    base = root / 'receipts' / campaign
    post = root / 'receipts' / (campaign + '-post-contract')
    try:
        if (base / 'campaign.exit').read_text().strip() != '0':
            return 'model campaign did not exit zero'
        for mode in TIMING_MODES:
            record = load(root / 'receipts' / (campaign + '-' + mode) / 'complete.json')
            if (record.get('cleanup_cleared') is not True or type(record.get('matrix_exit')) is not int or record.get('matrix_exit') != 0 or
                    record.get('error') or record.get('matrix_abort') or record.get('audit', {}).get('failures') != [] or
                    type(record.get('audit', {}).get('rows')) is not int or record['audit']['rows'] <= 0 or
                    type(record['audit'].get('complete_response_records')) is not int or
                    record['audit']['complete_response_records'] < record['audit']['rows']):
                return 'a model phase failed or has incomplete request/cleanup evidence'
        release_path = base / 'cache-resume.release.json'
        release = load(release_path)
        if (release.get('authorized') is not True or release.get('campaign') != campaign or
                release.get('run') != run_id or not re.fullmatch('[0-9a-f]{64}', release.get('candidate_sha256', ''))):
            return 'explicit release is missing or invalid'
        if release.get('contract_policy') == 'cache-independent':
            policy = load_policy()
            facts = policy.authorization(root, campaign, run_id)
            post_facts = policy.post_context(root, campaign)
            if release.get('independent_authorization') != facts or release.get('post_contract') != post_facts:
                return 'independent authorization or preserved post-contract evidence changed'
        else:
            if (post / 'post-contract.exit').read_text().strip() != '0':
                return 'post-contract did not exit zero'
            results = load(post / 'results.json')
            if (len(results) != 2 or {x['command'] for x in results} != {'kernel-contract-live', 'kernel-contract'} or
                    any(x.get('complete_success') is not True or x.get('exit') != 0 or not x.get('summaries') for x in results)):
                return 'post-contract evidence is incomplete'
        if release_path.stat().st_mtime_ns <= max((post / 'post-contract.exit').stat().st_mtime_ns,
                                                (base / 'campaign.exit').stat().st_mtime_ns):
            return 'release predates completion and cleanup'
        return None
    except (OSError, ValueError, KeyError, TypeError):
        return 'waiting for complete campaign, post-contract and explicit release evidence'


def load_policy():
    import importlib.util
    path = pathlib.Path(__file__).with_name('cache-handoff-policy.py')
    spec = importlib.util.spec_from_file_location('cache_handoff_policy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R2_FATAL_MARKERS = ('ncclCommInitRank', 'RoCE all-gather failed', 'NCCL carries every gather',
                    'ibv_reg_mr', 'Cannot allocate memory', 'memlock validation failed', 'Address already in use')
ROCE_READY = 'RoCE all-gather: 2 HCA(s), shards up to 512 KiB by RDMA writes'


def validate_transport(logs):
    for text in logs:
        for marker in R2_FATAL_MARKERS:
            if marker in text:
                raise RuntimeError('common r2 transport/startup failure: ' + marker)
    return len(logs) == 2 and all(ROCE_READY in text for text in logs)


def validate_native(native, uid, rank, flags, phase):
    # These are the actual native process's /proc facts, not the wrapper's claim.
    lines = [line for line in native.get('limits', '').splitlines() if line.startswith('Max locked memory')]
    if len(lines) != 1 or lines[0].split()[-3:-1] != ['unlimited', 'unlimited']:
        raise RuntimeError('native memlock validation failed')
    if native.get('status', {}).get('Uid', '').split() != [str(uid)] * 4:
        raise RuntimeError('native user identity changed')
    expected = {'TF_GLM_RANK': str(rank), 'TF_GLM_WORLD': '2', 'TF_GLM_PROF': '0',
                'TF_ROCE': '1', 'TF_ROCE_MAX_KB': '512', 'NCCL_IB_DISABLE': '0',
                'TF_GLM_KDA_VECTOR': str(flags[0]), 'TF_EXL3_N256': str(flags[1]),
                'TF_EXL3_FUSED_EPILOGUES': str(flags[2]), 'NIGHT_CAMPAIGN_ID': phase}
    if any(native.get('env', {}).get(k) != v for k, v in expected.items()):
        raise RuntimeError('unexpected effective rank environment or transport settings')


class Controller:
    def __init__(self, args):
        self.a = args
        self.root = pathlib.Path(args.root).resolve()
        self.out = self.root / 'receipts' / args.run
        self.ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15',
                    '-o', 'ServerAliveCountMax=4', args.worker]
        self.digest = None

    def event(self, event, **data):
        row = {'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': event, **data}
        print(json.dumps(row), flush=True)
        with (self.out / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        put(self.out / 'state.json', row)

    def run(self, rank, argv, timeout=30, checked=True):
        command = list(map(str, argv))
        if rank:
            command = self.ssh + [shlex.join(command)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if checked and result.returncode:
            raise RuntimeError(f'node {rank} command failed: {result.returncode}: {result.stderr[-600:]}')
        return result

    def probe(self, rank, pgid=0, cleanup=None, directory=None):
        argv = ['python3', self.root / 'paired-node.py', '--pgid', str(pgid)]
        if cleanup:
            if pgid <= 0:
                raise RuntimeError('refusing a nonpositive process group')
            argv += ['--cleanup', cleanup, '--mode-dir', directory]
        return json.loads(self.run(rank, argv).stdout)

    def get(self, rank, path):
        return self.run(rank, ['cat', path], checked=False).stdout

    def identity(self):
        files = ['candidate/zig-out/native/bin/tensorfold-native', 'launch-native.sh', 'paired-node.py',
                 'paired-rank-cache.sh', 'cache-reuse.py', 'cache-resume-controller.py',
                 'candidate/tools/zig/reuse_served.py', 'candidate/zig-out/bin/tf-cuda-test']
        if (self.root / 'cache-handoff-policy.py').exists():
            files += ['cache-handoff-policy.py', 'cache-handoff.py']
        identities = {}
        for name in files:
            values = [self.run(rank, ['sha256sum', self.root / name]).stdout.split()[0] for rank in [0, 1]]
            if len(set(values)) != 1:
                raise RuntimeError('rank identity differs for ' + name)
            identities[name] = values[0]
        release = load(self.root / 'receipts' / self.a.campaign / 'cache-resume.release.json')
        campaign = load(self.root / 'receipts' / self.a.campaign / 'identities.json')
        post = load(self.root / 'receipts' / (self.a.campaign + '-post-contract') / 'runner.json')
        if (identities[files[0]] != campaign.get('candidate') or identities['launch-native.sh'] != campaign.get('launcher') or
                identities['candidate/zig-out/bin/tf-cuda-test'] != post.get('sha256')):
            raise RuntimeError('current candidate, launcher or contract runner differs from completed qualification')
        if identities[files[0]] != release['candidate_sha256']:
            raise RuntimeError('candidate differs from explicit release identity')
        if self.digest is not None and identities != self.digest:
            raise RuntimeError('a binary, launcher or helper changed during the cache campaign')
        self.digest = identities
        return identities

    def wait(self):
        deadline = time.monotonic() + self.a.wait_seconds
        while not STOP and time.monotonic() < deadline:
            problem = prerequisites(self.root, self.a.campaign, self.a.run)
            if problem is None:
                infos = [self.probe(rank) for rank in [0, 1]]
                if any(any(p['exe'].startswith('tensorfold-native') or p['exe'] == 'tf-cuda-test'
                           for p in info['processes']) for info in infos):
                    problem = 'a model or GPU test process is still active'
                elif any(self.run(rank, ['flock', '-n', '/tmp/atlas-bench.lock', 'true'], checked=False).returncode
                         for rank in [0, 1]):
                    problem = 'a GPU lock is still held'
                else:
                    put(self.out / 'released-nodes.json', infos)
                    return
            self.event('waiting', reason=problem)
            time.sleep(10)
        raise RuntimeError('explicit release wait timed out or controller was stopped')

    def start(self, rank, directory, flags, cache):
        self.run(rank, ['mkdir', '-p', directory])
        argv = ['env', f'NIGHT_REUSE_CACHE_GIB={cache}', 'bash', self.root / 'paired-rank-cache.sh',
                str(rank), 'candidate', directory, *map(str, flags)]
        command = list(map(str, argv)) if not rank else self.ssh + [shlex.join(['setsid', '--wait', *map(str, argv)])]
        with (directory / f'rank{rank}.log').open('wb') as f:
            return subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)

    def group(self, rank, directory):
        try:
            return int(self.get(rank, directory / f'rank{rank}.pgid').strip())
        except ValueError:
            return 0

    def ready(self, directory, procs, groups, flags, cache):
        deadline = time.monotonic() + 900
        while not STOP and time.monotonic() < deadline:
            for rank in [0, 1]:
                if procs[rank].poll() is not None:
                    raise RuntimeError(f'rank {rank} ended before readiness')
                if not groups[rank]:
                    groups[rank] = self.group(rank, directory)
            logs = [(directory / f'rank{i}.log').read_text(errors='replace') for i in [0, 1]]
            transport_ready = validate_transport(logs)
            try:
                with urllib.request.urlopen('http://127.0.0.1:18080/health', timeout=3) as r:
                    health = json.load(r)
                with urllib.request.urlopen('http://127.0.0.1:18080/v1/models', timeout=3) as r:
                    models = json.load(r)
                if (health.get('status') == 'ok' and any(m.get('id') == 'kernel-research' for m in models.get('data', []))
                        and all(groups) and transport_ready):
                    infos = [self.probe(rank, groups[rank]) for rank in [0, 1]]
                    for rank, info in enumerate(infos):
                        native = [x for x in info['processes'] if x['exe'] == 'tensorfold-native']
                        if len(native) != 1:
                            raise RuntimeError('ambiguous native process ownership')
                        validate_native(native[0], os.getuid(), rank, flags, directory.name)
                        code = "import json,pathlib,sys; a=(pathlib.Path('/proc')/sys.argv[1]/'cmdline').read_bytes().split(b'\\0'); s=[x.decode() for x in a if x]; k=['--prompt-cache-gib','--context','--parallel','--host','--port']; print(json.dumps({v:s[s.index(v)+1] for v in k}))"
                        actual = json.loads(self.run(rank, ['python3', '-c', code, str(native[0]['pid'])]).stdout)
                        if actual != {'--prompt-cache-gib': str(cache), '--context': '131072', '--parallel': '4',
                                      '--host': '127.0.0.1', '--port': '18080'}:
                            raise RuntimeError('effective serve settings differ from the registered cache phase')
                        info['effective_serve_arguments'] = actual
                    put(directory / 'ready.json', {'health': health, 'models': models, 'ranks': infos, 'cache_gib': cache,
                                                       'native_uid_memlock_verified': True, 'roce_verified': True})
                    return
            except RuntimeError:
                raise
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError('startup timeout or stop')

    def cleanup(self, directory, procs, groups):
        result, cleared = [], True
        for rank in [0, 1]:
            item = {'rank': rank, 'pgid': groups[rank]}
            try:
                if not groups[rank]:
                    groups[rank] = self.group(rank, directory)
                    item['pgid'] = groups[rank]
                if groups[rank] > 0:
                    item['term'] = self.probe(rank, groups[rank], 'TERM', directory)
            except Exception as e:
                item['term_error'] = repr(e)
            result.append(item)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and any(p is not None and p.poll() is None for p in procs):
            time.sleep(1)
        for rank, item in enumerate(result):
            try:
                if groups[rank] <= 0:
                    if procs[rank] is not None:
                        cleared = False
                    continue
                remaining = self.probe(rank, groups[rank])
                if remaining['processes']:
                    item['kill'] = self.probe(rank, groups[rank], 'KILL', directory)
                    time.sleep(2)
                    remaining = self.probe(rank, groups[rank])
                item['remaining'] = remaining
                if remaining['processes']:
                    cleared = False
            except Exception as e:
                item['cleanup_error'] = repr(e)
                cleared = False
            for suffix in ['pgid', 'child.pid', 'child.exit', 'wrapper.exit', 'stopped.txt', 'memlock.json']:
                try:
                    text = self.get(rank, directory / f'rank{rank}.{suffix}')
                    if text and rank:
                        (directory / f'rank{rank}.{suffix}').write_text(text)
                except Exception as e:
                    item['evidence_error'] = repr(e)
                    cleared = False
            p = procs[rank]
            if p is not None:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                item['transport_exit'] = p.returncode
        report = {'cleared': cleared, 'ranks': result}
        put(directory / 'cleanup.json', report)
        return cleared

    def phase(self, mode, role, flags, cases):
        self.identity()
        directory = self.out / (mode + '-' + role)
        directory.mkdir()
        procs, groups = [None, None], [0, 0]
        child, status, error, ready = None, None, None, False
        try:
            procs[1] = self.start(1, directory, flags, self.a.cache_gib if role == 'warm' else 0)
            procs[0] = self.start(0, directory, flags, self.a.cache_gib if role == 'warm' else 0)
            self.ready(directory, procs, groups, flags, self.a.cache_gib if role == 'warm' else 0)
            ready = True
            operation = 'record' if not cases.exists() else 'replay'
            command = ['python3', self.root / 'cache-reuse.py', operation, '--out', directory / 'client', '--cases', cases]
            self.event('client-running', mode=mode, role=role, operation=operation)
            with (directory / 'client.log').open('wb') as f:
                child = subprocess.Popen(list(map(str, command)), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + 1800
                while child.poll() is None:
                    if STOP or time.monotonic() >= deadline:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        raise RuntimeError('client deadline or controller stop')
                    if any(p.poll() is not None for p in procs):
                        raise RuntimeError('a model rank ended while the client was running')
                    sample = {'time': time.time(), 'nodes': [self.probe(rank, groups[rank]) for rank in [0, 1]]}
                    with (directory / 'monitor.jsonl').open('a') as log:
                        log.write(json.dumps(sample) + '\n')
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
            status = child.returncode
            if status != 0:
                raise RuntimeError('cache client failed')
        except Exception as e:
            error = repr(e)
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            cleared = self.cleanup(directory, procs, groups)
            put(directory / 'complete.json', {'client_exit': status, 'ready_verified': ready,
                                             'cleanup_cleared': cleared, 'error': error})
        self.event('phase-finished', mode=mode, role=role, client_exit=status, error=error, cleanup_cleared=cleared)
        if error or not cleared or status != 0:
            raise RuntimeError('cache phase failed; no later phase will start')
        return directory / 'client/responses.jsonl'

    def execute(self):
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            self.wait()
            put(self.out / 'identity.json', self.identity())
            release = load(self.root / 'receipts' / self.a.campaign / 'cache-resume.release.json')
            put(self.out / 'qualification-context.json', release)
            cases = self.out / 'cases.jsonl'
            reports = []
            for mode in self.a.modes.split(','):
                if STOP:
                    raise RuntimeError('controller stopped')
                warm = self.phase(mode, 'warm', MODES[mode], cases)
                fresh = self.phase(mode, 'fresh', MODES[mode], cases)
                audit = self.run(0, ['python3', self.root / 'cache-reuse.py', 'audit', '--cases', cases,
                                     '--warm', warm, '--fresh', fresh, '--out', self.out / (mode + '-comparison.json')])
                reports.append(json.loads(audit.stdout))
                self.event('comparison', mode=mode, result=reports[-1])
            put(self.out / 'results.json', {'passed': all(x['passed'] for x in reports), 'scope': 'cache-correctness-only',
                 'post_contract': release.get('post_contract'), 'contract_policy': release.get('contract_policy', 'require-pass'),
                 'modes': reports})
            (self.out / 'cache-resume.exit').write_text('0\n')
            return 0
        except Exception as e:
            self.event('failed', error=repr(e))
            (self.out / 'cache-resume.exit').write_text('1\n')
            return 1


def selftest():
    with tempfile.TemporaryDirectory(prefix='cache-controller-selftest-') as name:
        root, campaign, run_id = pathlib.Path(name), 'campaign', 'reuse'
        base, post = root / 'receipts' / campaign, root / 'receipts' / (campaign + '-post-contract')
        base.mkdir(parents=True)
        post.mkdir()
        assert prerequisites(root, campaign, run_id)
        (base / 'campaign.exit').write_text('0\n')
        for mode in TIMING_MODES:
            put(root / 'receipts' / (campaign + '-' + mode) / 'complete.json', {'cleanup_cleared': True, 'matrix_exit': 0, 'audit': {'failures': [], 'rows': 1, 'complete_response_records': 1}})
        (post / 'post-contract.exit').write_text('0\n')
        put(post / 'results.json', [{'command': x, 'exit': 0, 'complete_success': True, 'summaries': ['SUMMARY pass']}
                                   for x in ['kernel-contract-live', 'kernel-contract']])
        assert prerequisites(root, campaign, run_id)
        release = base / 'cache-resume.release.json'
        put(release, {'authorized': True, 'campaign': campaign, 'run': run_id, 'candidate_sha256': 'a' * 64})
        latest = max((base / 'campaign.exit').stat().st_mtime_ns, (post / 'post-contract.exit').stat().st_mtime_ns)
        os.utime(release, ns=(latest + 1000000, latest + 1000000))
        assert prerequisites(root, campaign, run_id) is None
        phase = root / 'receipts' / (campaign + '-vector') / 'complete.json'
        put(phase, {'cleanup_cleared': True})
        assert prerequisites(root, campaign, run_id)
        put(phase, {'cleanup_cleared': True, 'matrix_exit': 0, 'audit': {'failures': [], 'rows': 1, 'complete_response_records': 1}})
        assert prerequisites(root, campaign, run_id) is None
        os.utime(release, ns=(latest - 1000000, latest - 1000000))
        assert prerequisites(root, campaign, run_id)
        os.utime(release, ns=(latest + 1000000, latest + 1000000))
        (post / 'post-contract.exit').write_text('1\n')
        assert prerequisites(root, campaign, run_id)
        (post / 'post-contract.exit').write_text('0\n')
        put(root / 'receipts' / (campaign + '-fusion') / 'complete.json', {'cleanup_cleared': False})
        assert prerequisites(root, campaign, run_id)
    print(json.dumps({'offline_controller_selftest': 'pass', 'positive': 1, 'release_and_cleanup_controls_rejected': 6}))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=str(pathlib.Path.home() / 'tensorfold-kernel-night'))
    p.add_argument('--campaign')
    p.add_argument('--run')
    p.add_argument('--worker', default='worker')
    p.add_argument('--modes', default='vector,n256,fusion')
    p.add_argument('--cache-gib', type=int, default=1)
    p.add_argument('--wait-seconds', type=int, default=1800)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if not a.campaign or not a.run or any(not re.fullmatch('[A-Za-z0-9_.-]+', x) or x in {'.', '..'} for x in [a.campaign, a.run]):
        p.error('distinct safe --campaign and --run names are required')
    if (a.campaign == a.run or not a.modes or len(set(a.modes.split(','))) != len(a.modes.split(',')) or
            any(x not in MODES for x in a.modes.split(',')) or a.cache_gib != 1 or a.wait_seconds <= 0):
        p.error('use distinct run name, registered modes and the reviewed 1 GiB budget')
    if not a.execute:
        print(json.dumps({'plan_only': True, 'campaign': a.campaign, 'run': a.run, 'modes': a.modes.split(','),
                          'starts_gpu_work': False, 'requires_execute_and_fresh_post_contract_release': True,
                          'pairs': 2 * len(a.modes.split(',')), 'completion_requests_per_mode': 8}, indent=2))
        return 0
    def stop(sig, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    return Controller(a).execute()


if __name__ == '__main__':
    raise SystemExit(main())
