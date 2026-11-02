#!/usr/bin/env python3
"""Collect immutable final cache evidence after exit zero; verify both GPUs are released and original red receipts unchanged."""
import hashlib,json,pathlib,shlex,subprocess,time
root=pathlib.Path.home()/'tensorfold-kernel-night';run=root/'receipts/paired-20261030-r2-cache-resume'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
if (run/'cache-resume.exit').read_text().strip()!='0':raise SystemExit('cache run did not exit zero')
results=load(run/'results.json');wire=load(run/'all-wire-audit.json');context=load(run/'qualification-context.json')
if (results.get('passed') is not True or results.get('scope')!='cache-correctness-only' or wire.get('passed') is not True or
    len(results['modes'])!=3 or any(x.get('passed') is not True for x in results['modes']) or len(wire['phases'])!=6):raise SystemExit('cache result incomplete')
post=root/'receipts/paired-20261030-r2-post-contract';facts=context['post_contract']
if sha(post/'results.json')!=facts['results_sha256'] or sha(post/'state.json')!=facts['state_sha256']:raise SystemExit('post context changed')
for item in facts['commands']:
 if sha(post/(item['command']+'.log'))!=item['log_sha256'] or int((post/(item['command']+'.exit')).read_text())!=item['exit']:raise SystemExit('original post log/exit changed')
if int((post/'post-contract.exit').read_text())!=facts['overall_exit']:raise SystemExit('original overall exit changed')
all_rows=[];wire_count=0;positive=0;fresh_zero=0
for phase in wire['phases']:
 name=phase['phase'];mode,role=name.split('-');rows=phase['replies'];all_rows.extend(rows);wire_count+=phase['raw_wire_responses']
 if phase.get('passed') is not True or len(rows)!=4 or phase.get('cleanup_verified') is not True:raise SystemExit('phase incomplete')
 expected={'vector':('1','0','0'),'n256':('0','1','0'),'fusion':('0','0','1')}[mode]
 ready=load(run/name/'ready.json')
 for rank in ready['ranks']:
  native=[x for x in rank['processes'] if x['exe']=='tensorfold-native']
  if len(native)!=1:raise SystemExit('native proof missing')
  env=native[0]['env'];flags=tuple(env[x] for x in ['TF_GLM_KDA_VECTOR','TF_EXL3_N256','TF_EXL3_FUSED_EPILOGUES'])
  if flags!=expected:raise SystemExit('wrong feature mode in actual native proof')
 if role=='warm':positive+=sum(x['cached']>0 for x in rows)
 else:fresh_zero+=sum(type(x['cached']) is int and x['cached']==0 for x in rows)
if (len(all_rows),wire_count,positive,fresh_zero)!=(24,54,9,12):raise SystemExit('request or cache witness counts differ')
infos=[]
for rank in [0,1]:
 command=['python3',str(root/'paired-node.py')]
 if rank:command=['ssh','-o','BatchMode=yes','worker',shlex.join(command)]
 info=json.loads(subprocess.run(command,capture_output=True,text=True,timeout=30,check=True).stdout)
 if info['processes']:raise SystemExit('model or GPU test still active')
 command=['flock','-n','/tmp/atlas-bench.lock','true']
 if rank:command=['ssh','-o','BatchMode=yes','worker',shlex.join(command)]
 if subprocess.run(command,capture_output=True,timeout=30).returncode:raise SystemExit('GPU lock still held')
 infos.append({'rank':rank,'lock_free':True,'probe':info})
put(run/'final-released-nodes.json',infos)
summary={'scope':'cache-correctness-only','cache_passed':True,'cache_exit':0,'modes':['vector','n256','fusion'],
         'completed_replies':24,'raw_wire_responses':wire_count,'positive_warm_reuses':positive,'fresh_strict_zero_counters':fresh_zero,
         'all_frozen_bodies_and_rendered_ids_equal':True,'all_full_token_hashes_equal':True,'all_phases_cleaned':True,
         'both_gpu_locks_free':True,'original_post_receipts_unchanged':True,'post_contract':facts,
         'candidate_sha256':load(run/'identity.json')['candidate/zig-out/native/bin/tensorfold-native'],
         'runner_sha256':load(run/'identity.json')['candidate/zig-out/bin/tf-cuda-test'],
         'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
put(run/'final-summary.json',summary)
files=sorted(p for p in run.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
(run/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(run))+'\n' for p in files))
print(json.dumps(summary,indent=2))
