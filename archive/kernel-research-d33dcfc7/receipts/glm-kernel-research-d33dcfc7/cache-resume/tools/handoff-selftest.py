#!/usr/bin/env python3
"""Offline fixtures only; no subprocesses, sockets, GPU queries, releases in a real campaign, or model launches."""
import argparse, copy, importlib.util, json, os, pathlib, tempfile
D=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location('handoff',D/'cache-handoff.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
p,c=h.p,h.c
parser=argparse.ArgumentParser();parser.add_argument('--registry',default=str(pathlib.Path(__file__).resolve().parents[5]/'zig/src/core/contract/kernel_research_records.zig'));a=parser.parse_args()
controls=0
with tempfile.TemporaryDirectory(prefix='cache-handoff-selftest-') as temporary:
 root=pathlib.Path(temporary);campaign,run='campaign','cache';base=root/'receipts'/campaign;base.mkdir(parents=True)
 post=root/'receipts'/(campaign+'-post-contract');post.mkdir()
 def write(name,text):
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);return path
 for name in p.REQUIRED_SOURCES:write('candidate/'+name,'source '+name)
 write('candidate/zig/src/core/contract/kernel_research_records.zig',pathlib.Path(a.registry).read_text())
 for name in h.HELPERS:write(name,'helper '+name)
 write('candidate/zig-out/native/bin/tensorfold-native','native');write('candidate/zig-out/bin/tf-cuda-test','runner')
 for mode in c.TIMING_MODES:c.put(root/'receipts'/(campaign+'-'+mode)/'complete.json',{'matrix_exit':0,'cleanup_cleared':True,'audit':{'failures':[],'rows':1,'complete_response_records':1}})
 write('receipts/'+campaign+'/campaign.exit','0\n')
 for role,summary in p.FOCUSED.items():
  write('receipts/cache-resume-preparation/focused-'+role+'.log',summary+'\n');write('receipts/cache-resume-preparation/focused-'+role+'.exit','0\n')
 pre=root/'receipts'/(campaign+'-preflight');pre.mkdir()
 registry=(root/'candidate/zig/src/core/contract/kernel_research_records.zig').read_text()
 names={image+' '+symbol for image,symbol in h.re.findall(r'entry\("([^"]+)", "([^"]+)"',registry)}
 assert len(names)==17
 (pre/'kernel-contract-live.log').write_text('\n'.join('PASS limits live '+x+': fixture' for x in sorted(names))+'\nFAIL limits live unrelated: fixture\nSUMMARY live CUDA limits: 183 problems; numerical gates not run\n')
 (pre/'kernel-contract-live.exit').write_text('1\n')
 c.put(pre/'result.json',{'exit':1,'runner_unchanged':True,'summaries':['SUMMARY live CUDA limits: 183 problems; numerical gates not run']})
 c.put(pre/'runner.json',{'sha256':p.sha(root/'candidate/zig-out/bin/tf-cuda-test')})
 value,pre_facts=h.template(root,campaign,run);assert value['enabled'] is False;assert pre_facts['research_passed_record_count']==17
 value['enabled']=True;auth=base/'cache-handoff.authorization.json';c.put(auth,value)
 assert p.authorization(root,campaign,run)['research_preflight']['overall_preflight_exit']==1
 def rejected(call):
  global controls
  try:call()
  except (ValueError,OSError,KeyError,TypeError):controls+=1;return
  raise AssertionError('false pass was accepted')
 original=copy.deepcopy(value)
 for field,bad in [('enabled',False),('runner_sha256','a'*64)]:
  value=copy.deepcopy(original);value[field]=bad;c.put(auth,value);rejected(lambda:p.authorization(root,campaign,run))
 value=copy.deepcopy(original);value['research_preflight']['passed_record_count']=16;c.put(auth,value);rejected(lambda:p.authorization(root,campaign,run))
 for role in p.FOCUSED:
  value=copy.deepcopy(original);item=value['focused_qualifications'][role]['exit'];path=root/item['path'];path.write_text('1\n');item['sha256']=p.sha(path);c.put(auth,value);rejected(lambda:p.authorization(root,campaign,run));path.write_text('0\n')
 value=copy.deepcopy(original);resource=value['research_preflight']['log'];path=root/resource['path'];text=path.read_text()
 for bad in [text.replace('PASS limits live '+sorted(names)[0],'FAIL limits live '+sorted(names)[0]),text+'PASS limits live '+sorted(names)[0]+': duplicate\n']:
  path.write_text(bad);resource['sha256']=p.sha(path);c.put(auth,value);rejected(lambda:p.authorization(root,campaign,run))
 path.write_text(text);c.put(auth,original)
 source=root/('candidate/'+sorted(p.REQUIRED_SOURCES)[0]);old=source.read_text();source.write_text(old+' changed');rejected(lambda:p.authorization(root,campaign,run));source.write_text(old)
 results=[]
 for command in ['kernel-contract-live','kernel-contract']:
  (post/(command+'.log')).write_text('FAIL unrelated fixture\nSUMMARY complete red fixture\n');(post/(command+'.exit')).write_text('1\n')
  results.append({'command':command,'exit':1,'summaries':['SUMMARY complete red fixture'],'complete_success':False})
 c.put(post/'results.json',results);c.put(post/'state.json',{'state':'finished','stopped':False});(post/'post-contract.exit').write_text('1\n')
 before={x.name:p.sha(x) for x in post.iterdir()};red=p.post_context(root,campaign);assert red['overall_exit']==1 and red['overall_contract_passed'] is False
 assert before=={x.name:p.sha(x) for x in post.iterdir()}
 results[-1].update(exit=124,summaries=[]);(post/'kernel-contract.log').write_text('Timeout preserved fixture\n');(post/'kernel-contract.exit').write_text('124\n');c.put(post/'results.json',results)
 timeout=p.post_context(root,campaign);assert timeout['commands'][-1]['outcome']=='timeout' and timeout['overall_contract_passed'] is False
 results[-1]['complete_success']=True;c.put(post/'results.json',results);rejected(lambda:p.post_context(root,campaign));results[-1]['complete_success']=False;c.put(post/'results.json',results)
 c.put(post/'state.json',{'state':'finished','stopped':True});rejected(lambda:p.post_context(root,campaign));c.put(post/'state.json',{'state':'finished','stopped':False})
 c.put(post/'results.json',results[:1]);rejected(lambda:p.post_context(root,campaign));c.put(post/'results.json',results)
 release={'authorized':True,'campaign':campaign,'run':run,'candidate_sha256':original['candidate_sha256'],'contract_policy':'cache-independent','independent_authorization':p.authorization(root,campaign,run),'post_contract':p.post_context(root,campaign)}
 release_path=base/'cache-resume.release.json';c.put(release_path,release)
 latest=max((base/'campaign.exit').stat().st_mtime_ns,(post/'post-contract.exit').stat().st_mtime_ns);os.utime(release_path,ns=(latest+1000000,latest+1000000))
 assert c.prerequisites(root,campaign,run) is None
 (post/'kernel-contract.log').write_text('changed raw timeout evidence\n');assert c.prerequisites(root,campaign,run) is not None;controls+=1
print(json.dumps({'offline_handoff_selftest':'pass','research_records':17,'red_and_timeout_retained':True,'false_pass_controls_rejected':controls,'gpu_work_started':False}))
