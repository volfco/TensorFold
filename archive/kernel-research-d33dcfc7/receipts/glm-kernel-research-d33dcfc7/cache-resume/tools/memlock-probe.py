#!/usr/bin/env python3
"""Run only the cache wrapper's process-local memlock guard in a disposable CPU shell; never launch a model."""
import argparse,json,os,pathlib,shlex,subprocess
p=argparse.ArgumentParser();p.add_argument('--rank',choices=['0','1'],required=True);a=p.parse_args()
root=pathlib.Path.home()/'tensorfold-kernel-night';out=root/'receipts/cache-resume-preparation/memlock-probe';out.mkdir(parents=True,exist_ok=True)
text=(root/'paired-rank-cache.sh').read_text();begin=text.index('night_uid=$(id -u)');end=text.index('bash "$HOME/tensorfold-kernel-night/launch-native.sh"',begin);guard=text[begin:end]
assert 'sudo -n prlimit --pid $$ --memlock=unlimited:unlimited' in guard
code='night_out='+shlex.quote(str(out))+'\nnight_rank='+a.rank+'\n'+guard
result=subprocess.run(['bash','-c',code],capture_output=True,text=True,timeout=30)
(out/('rank'+a.rank+'.guard.log')).write_text(result.stdout+result.stderr)
(out/('rank'+a.rank+'.guard.exit')).write_text(str(result.returncode)+'\n')
if result.returncode:raise SystemExit(result.returncode)
facts=json.loads((out/('rank'+a.rank+'.memlock.json')).read_text())
if facts!={'uid':os.getuid(),'euid':os.getuid(),'memlock':[-1,-1]}:raise SystemExit('unexpected isolated guard result')
print(json.dumps({'rank':int(a.rank),'process_local_guard':'pass','facts':facts,'gpu_work_started':False,'parent_or_active_process_limits_changed':False}))
