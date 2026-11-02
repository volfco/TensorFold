#!/usr/bin/env python3
"""Disabled by default: prepare an independent-cache authorization template or wait for an explicit enabled one."""
import argparse, importlib.util, json, pathlib, re, signal, time

def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,pathlib.Path(__file__).with_name(filename))
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value
c=module('cache_controller','cache-resume-controller.py')
p=module('cache_policy','cache-handoff-policy.py')
HELPERS=['cache-resume-controller.py','cache-handoff-policy.py','cache-handoff.py','cache-reuse.py',
         'paired-rank-cache.sh','launch-native.sh','paired-node.py','candidate/tools/zig/reuse_served.py']

def evidence(root,name): return {'path':name,'sha256':p.sha(root/name)}
def template(root,campaign,run):
    location='receipts/cache-resume-preparation/'
    pre='receipts/'+campaign+'-preflight/'
    resource={key:evidence(root,pre+filename) for key,filename in [
        ('log','kernel-contract-live.log'),('exit','kernel-contract-live.exit'),('result','result.json'),('runner','runner.json')]}
    registry=(root/'candidate/zig/src/core/contract/kernel_research_records.zig').read_text()
    resource['passed_record_count']=len(set(re.findall(r'entry\("([^"]+)", "([^"]+)"',registry)))
    value={'schema':1,'enabled':False,'scope':'cache-correctness-independent-of-overall-contract',
           'campaign':campaign,'run':run,'candidate_sha256':p.sha(root/'candidate/zig-out/native/bin/tensorfold-native'),
           'runner_sha256':p.sha(root/'candidate/zig-out/bin/tf-cuda-test'),
           'reviewed_current_research_sources':{name:p.sha(root/('candidate/'+name)) for name in sorted(p.REQUIRED_SOURCES)},
           'helper_sha256':{name:p.sha(root/name) for name in HELPERS},
           'focused_qualifications':{name:{'log':evidence(root,location+'focused-'+name+'.log'),
                                            'exit':evidence(root,location+'focused-'+name+'.exit')} for name in p.FOCUSED},
           'research_preflight':resource}
    for role, expected in p.FOCUSED.items():
        item=value['focused_qualifications'][role]
        text=p.checked(root,item['log']).read_text()
        if (p.checked(root,item['exit']).read_text().strip()!='0' or text.splitlines().count(expected)!=1 or 'MISSED' in text):
            raise ValueError('actual focused qualification is not green: '+role)
    facts=p.preflight(root,resource,value['runner_sha256'])
    return value,facts

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(pathlib.Path.home()/'tensorfold-kernel-night'))
    parser.add_argument('--campaign',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--wait-seconds',type=int,default=21600);a=parser.parse_args()
    if (a.campaign==a.run or any(not re.fullmatch('[A-Za-z0-9_.-]+',x) or x in {'.','..'} for x in [a.campaign,a.run]) or
        a.wait_seconds<=0 or a.prepare and a.execute):parser.error('safe distinct IDs and one operation are required')
    root=pathlib.Path(a.root).resolve();base=root/'receipts'/a.campaign
    if a.prepare:
        value,facts=template(root,a.campaign,a.run)
        target=base/'cache-handoff.authorization.TEMPLATE.json'
        with target.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
        c.put(base/'cache-handoff.preflight-parsed.json',facts)
        print(json.dumps({'template':str(target),'enabled':False,'gpu_work_started':False,'actual_preflight':facts},indent=2));return 0
    if not a.execute:
        print(json.dumps({'plan_only':True,'starts_gpu_work':False,'campaign':a.campaign,'run':a.run,
                          'requires_explicit_enabled_authorization':True,'post_contract_red_or_timeout_is_preserved':True}));return 0
    # This is the only execution path. Root must explicitly enable a reviewed template.
    authorization=p.authorization(root,a.campaign,a.run)
    args=argparse.Namespace(root=str(root),campaign=a.campaign,run=a.run,worker='worker',
                            modes='vector,n256,fusion',cache_gib=1,wait_seconds=120,execute=True)
    controller=c.Controller(args)
    def stop(sig,frame): c.STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    (base/'cache-handoff.pid').write_text(str(__import__('os').getpid())+'\n')
    deadline=time.monotonic()+a.wait_seconds
    while not c.STOP and time.monotonic()<deadline:
        try:
            if p.authorization(root,a.campaign,a.run)!=authorization:raise RuntimeError('authorization facts changed')
            post=p.post_context(root,a.campaign)
            path=base/'cache-resume.release.json'
            release={'authorized':True,'campaign':a.campaign,'run':a.run,'candidate_sha256':authorization['candidate_sha256'],
                     'contract_policy':'cache-independent','independent_authorization':authorization,'post_contract':post}
            # Use the exact controller model gates before creating the release. A temporary release is never installed.
            if (base/'campaign.exit').read_text().strip()!='0':raise ValueError('model campaign did not pass')
            for mode in c.TIMING_MODES:
                row=p.load(root/'receipts'/(a.campaign+'-'+mode)/'complete.json');audit=row.get('audit',{})
                if (row.get('cleanup_cleared') is not True or type(row.get('matrix_exit')) is not int or row.get('matrix_exit')!=0 or row.get('error') or row.get('matrix_abort') or
                    audit.get('failures')!=[] or type(audit.get('rows')) is not int or audit['rows']<=0 or
                    type(audit.get('complete_response_records')) is not int or audit['complete_response_records']<audit['rows']):
                    raise ValueError('model phase failed or audit/cleanup incomplete: '+mode)
            infos=[controller.probe(rank) for rank in [0,1]]
            if any(info['processes'] for info in infos):raise ValueError('a model or test process remains active')
            if any(controller.run(rank,['flock','-n','/tmp/atlas-bench.lock','true'],checked=False).returncode for rank in [0,1]):
                raise ValueError('a GPU lease is still held')
            if c.STOP:raise RuntimeError('handoff stopped before release')
            with path.open('x') as f:json.dump(release,f,indent=2);f.write('\n')
            problem=c.prerequisites(root,a.campaign,a.run)
            if problem:raise RuntimeError('refused after release: '+problem)
            print(json.dumps({'handoff':'starting-cache-only','post_contract':post}),flush=True)
            return controller.execute()
        except FileExistsError:
            raise RuntimeError('release already exists; refusing duplicate handoff')
        except (OSError,ValueError,KeyError,TypeError) as error:
            print(json.dumps({'waiting':str(error),'gpu_work_started':False}),flush=True);time.sleep(10)
    raise RuntimeError('handoff deadline; no cache launch')

if __name__=='__main__':raise SystemExit(main())
