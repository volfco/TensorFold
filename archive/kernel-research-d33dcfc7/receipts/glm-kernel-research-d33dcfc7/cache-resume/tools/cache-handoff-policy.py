#!/usr/bin/env python3
"""Strict, GPU-free validation of independent cache authorization and unchanged red post-contract evidence."""
import hashlib, json, pathlib, re

FOCUSED = {
    'kda': 'SUMMARY KDA diagnostic: 6 kernels, 6 pass, 0 fail; 1 controls, 1 red, 0 missed; limits not run',
    'n256': 'SUMMARY N256 focused: 3 kernels pass, 0 fail; 2 controls red, 0 missed; limits not run',
    'fusion': 'SUMMARY exl3-fused focused kernels=4 passed=4 failed=0 controls=2 caught=2 missed=0; launch limits belong to full runner',
}
REQUIRED_SOURCES = {
    'zig/kernels/cuda/glm_kda.cu', 'zig/kernels/cuda/glm_kda_vector.cuh',
    'zig/kernels/cuda/exl3_experts.cu', 'zig/kernels/cuda/exl3_experts_epi.cuh',
    'zig/src/families/glm/cuda_kda.zig', 'zig/src/cuda/exl3_experts.zig',
    'zig/src/cuda/exl3_experts_epi.zig', 'zig/src/core/contract/kernel_research_records.zig',
}


def load(path): return json.loads(pathlib.Path(path).read_text())
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def member(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('evidence path must be an existing file under the task root')
    return path

def checked(root, item):
    path = member(root, item['path'])
    if not re.fullmatch('[0-9a-f]{64}', item['sha256']) or sha(path) != item['sha256']:
        raise ValueError('evidence hash changed: ' + item['path'])
    return path


def authorization(root, campaign, run_id):
    root = pathlib.Path(root).resolve()
    path = root / 'receipts' / campaign / 'cache-handoff.authorization.json'
    value = load(path)
    if (value.get('schema') != 1 or value.get('enabled') is not True or
            value.get('scope') != 'cache-correctness-independent-of-overall-contract' or
            value.get('campaign') != campaign or value.get('run') != run_id):
        raise ValueError('independent cache authorization is absent or invalid')
    candidate = root / 'candidate/zig-out/native/bin/tensorfold-native'
    runner = root / 'candidate/zig-out/bin/tf-cuda-test'
    if sha(candidate) != value['candidate_sha256'] or sha(runner) != value['runner_sha256']:
        raise ValueError('authorized candidate or runner changed')
    helpers = value['helper_sha256']
    required_helpers = {'cache-resume-controller.py','cache-handoff-policy.py','cache-handoff.py','cache-reuse.py','paired-rank-cache.sh','launch-native.sh','paired-node.py','candidate/tools/zig/reuse_served.py'}
    if set(helpers) != required_helpers: raise ValueError('helper inventory is incomplete')
    for name, digest in helpers.items():
        if sha(member(root,name)) != digest: raise ValueError('authorized helper or launcher changed: '+name)
    sources = value['reviewed_current_research_sources']
    if set(sources) != REQUIRED_SOURCES:
        raise ValueError('reviewed research source inventory is incomplete')
    for name, digest in sources.items():
        if sha(member(root, 'candidate/' + name)) != digest:
            raise ValueError('reviewed research source changed: ' + name)
    focused = value['focused_qualifications']
    if set(focused) != set(FOCUSED):
        raise ValueError('all three focused qualifications are required')
    for name, expected in FOCUSED.items():
        log = checked(root, focused[name]['log']).read_text(errors='strict')
        status = checked(root, focused[name]['exit']).read_text().strip()
        if status != '0' or log.splitlines().count(expected) != 1 or 'MISSED' in log:
            raise ValueError('focused qualification did not pass: ' + name)
    facts = preflight(root, value['research_preflight'], value['runner_sha256'])
    return {'authorization_sha256': sha(path), 'candidate_sha256': value['candidate_sha256'],
            'runner_sha256': value['runner_sha256'], 'research_preflight': facts,
            'focused_summaries': FOCUSED, 'scope': value['scope']}


def preflight(root, resource, expected_runner):
    root = pathlib.Path(root).resolve()
    log_path = checked(root, resource['log'])
    runner_receipt = load(checked(root, resource['runner']))
    result = load(checked(root, resource['result']))
    status = checked(root, resource['exit']).read_text().strip()
    if runner_receipt.get('sha256') != expected_runner or result.get('runner_unchanged') is not True:
        raise ValueError('preflight used a different or changed contract runner')
    registry = (root / 'candidate/zig/src/core/contract/kernel_research_records.zig').read_text()
    expected = {image + ' ' + symbol for image, symbol in re.findall(r'entry\("([^\"]+)\", \"([^\"]+)\"', registry)}
    text = log_path.read_text(errors='strict')
    observed = []
    for line in text.splitlines():
        match = re.match(r'(PASS|FAIL|MISSED) limits live (.*?): ', line)
        if match and match[2] in expected:
            if match[1] != 'PASS': raise ValueError('research resource record is red: ' + match[2])
            observed.append(match[2])
    summaries = [line for line in text.splitlines() if line.startswith('SUMMARY')]
    if (not expected or len(observed) != len(expected) or set(observed) != expected or
            resource.get('passed_record_count') != len(expected)):
        raise ValueError('research preflight is incomplete or has duplicate records')
    if status not in ('0', '1') or result.get('exit') != int(status) or result.get('summaries') != summaries:
        raise ValueError('actual preflight receipt disagrees with its log or exit')
    return {'research_passed_record_count': len(expected), 'research_failed_record_count': 0,
            'research_symbols': sorted(expected), 'overall_preflight_exit': int(status),
            'overall_preflight_summaries': summaries, 'log_sha256': sha(log_path),
            'result_sha256': resource['result']['sha256']}


def post_context(root, campaign):
    post = pathlib.Path(root) / 'receipts' / (campaign + '-post-contract')
    overall = int((post / 'post-contract.exit').read_text().strip())
    if overall not in (0, 1): raise ValueError('post watcher has no normal terminal outcome')
    state = load(post / 'state.json')
    if state.get('state') != 'finished' or state.get('stopped') is not False:
        raise ValueError('post watcher did not finish its owned command sequence')
    results = load(post / 'results.json')
    if len(results) != 2 or {x.get('command') for x in results} != {'kernel-contract-live', 'kernel-contract'}:
        raise ValueError('both post-contract commands must have recorded terminal outcomes')
    facts = []
    for item in results:
        name, status = item['command'], item.get('exit')
        if type(status) is not int or status not in (0, 1, 124, 137):
            raise ValueError('unexpected post-command outcome requires manual review')
        if int((post / (name + '.exit')).read_text().strip()) != status:
            raise ValueError('post-command exit evidence disagrees')
        log = post / (name + '.log')
        text = log.read_text(errors='strict')
        if not text: raise ValueError('post-command raw log is missing')
        summaries = [line for line in text.splitlines() if line.startswith('SUMMARY')]
        if summaries != item.get('summaries'):
            raise ValueError('post-command summaries disagree with preserved raw evidence')
        if status == 0 and (not summaries or item.get('complete_success') is not True):
            raise ValueError('zero post-command exit lacks its successful summary')
        if status != 0 and item.get('complete_success') is not False:
            raise ValueError('red/timeout post-command was incorrectly marked passed')
        facts.append({'command': name, 'exit': status, 'summaries': summaries,
                      'log_sha256': sha(log), 'outcome': 'passed' if status == 0 else
                      'timeout' if status == 124 else 'killed-or-timeout' if status == 137 else 'failed'})
    passed = overall == 0 and all(x['exit'] == 0 for x in facts)
    if overall == 0 and not passed: raise ValueError('post watcher incorrectly calls a red command successful')
    return {'overall_exit': overall, 'overall_contract_passed': passed, 'commands': facts,
            'results_sha256': sha(post / 'results.json'), 'state_sha256': sha(post / 'state.json')}
