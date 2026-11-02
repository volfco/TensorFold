#!/usr/bin/env python3
"""Read-only independent audit of saved cache request/response bytes, tokenization, counters, identities and cleanup."""
import argparse,hashlib,json,pathlib,re
if not __debug__: raise RuntimeError("refusing audit with Python assertions disabled")

def sha(data):return hashlib.sha256(data).hexdigest()
def load(path):return json.loads(path.read_text())
def lines(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def render_key(body):
 return json.dumps({'model':body.get('model'),'messages':body['messages'],'tools':body.get('tools'),
                    'thinking':body.get('chat_template_kwargs',{}).get('enable_thinking')},sort_keys=True)

def audit(run,phase,cases):
 folder=run/phase;complete=load(folder/'complete.json');cleanup=load(folder/'cleanup.json');ready=load(folder/'ready.json')
 assert complete=={'client_exit':0,'ready_verified':True,'cleanup_cleared':True,'error':None},'incomplete phase'
 assert cleanup.get('cleared') is True and ready.get('native_uid_memlock_verified') is True and ready.get('roce_verified') is True
 cache=1 if phase.endswith('-warm') else 0
 assert ready['cache_gib']==cache
 normalized=lines(folder/'client/responses.jsonl');assert [x['i'] for x in normalized]==list(range(4))
 tokens={};chats=[];wire_count=0
 for metadata in sorted((folder/'client').glob('wire-*.json')):
  meta=load(metadata);request=metadata.with_suffix('.request').read_bytes();response=metadata.with_suffix('.response').read_bytes()
  assert meta['status']==200 and meta['error'] is None,'failed wire response'
  assert sha(request)==meta['request_sha256'] and sha(response)==meta['response_sha256'],'raw wire hash mismatch'
  body,out=json.loads(request),json.loads(response);wire_count+=1
  if meta['path']=='/v1/tokenize':
   if body.get('add_generation_prompt') is True:tokens[render_key(body)]=out['tokens']
  else:
   assert meta['path']=='/v1/chat/completions','unexpected route'
   i=len(chats);assert i<4
   case,row=cases[i],normalized[i];got=row['got'];usage=out['usage'];full=out['tensorfold']['token_sha256']
   assert re.fullmatch('[0-9a-f]{64}',full),'missing full token digest'
   assert sha(request)==case['body_sha256']==row['body_sha256'],'frozen request bytes changed'
   assert body==case['body'],'canonical body differs'
   ids=tokens[render_key(body)];assert ids==case['prompt_ids']==row['prompt_ids'],'raw rendered IDs differ'
   assert type(usage['prompt_tokens']) is int and usage['prompt_tokens']==len(ids)==got['prompt']
   cached=usage['prompt_tokens_details']['cached_tokens'];assert type(cached) is int
   count=usage['completion_tokens'];assert type(count) is int and 0<count<=32
   assert full==got['sha']==case['got']['sha'] and count==got['tokens']==case['got']['tokens'],'raw output differs'
   assert cached==got['cached']
   assert cached==0 if not cache or i==0 else 0<cached<=case['history'],'cache witness absent'
   chats.append({'i':i,'prompt':len(ids),'cached':cached,'tokens':count,'token_sha256':full,'request_sha256':sha(request)})
 assert len(chats)==4,'incomplete raw completion matrix'
 for rank in ready['ranks']:
  native=[x for x in rank['processes'] if x['exe']=='tensorfold-native'];assert len(native)==1
  n=native[0];assert n['status']['Uid'].split()==['1000']*4
  locked=[x for x in n['limits'].splitlines() if x.startswith('Max locked memory')]
  assert len(locked)==1 and locked[0].split()[-3:-1]==['unlimited','unlimited']
  assert n['env']['TF_ROCE']=='1' and n['env']['NCCL_IB_DISABLE']=='0'
  assert rank['effective_serve_arguments']['--prompt-cache-gib']==str(cache)
 return {'phase':phase,'passed':True,'completed_replies':len(chats),'raw_wire_responses':wire_count,'replies':chats,
         'cleanup_verified':True,'native_uid_memlock_transport_verified':True}

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=pathlib.Path,required=True);p.add_argument('--phase');p.add_argument('--out',type=pathlib.Path);a=p.parse_args()
cases=lines(a.run/'cases.jsonl');assert [x['i'] for x in cases]==list(range(4))
phases=[a.phase] if a.phase else [m+'-'+r for m in ['vector','n256','fusion'] for r in ['warm','fresh']]
report={'scope':'independent-raw-wire-cache-audit','passed':True,'phases':[audit(a.run,x,cases) for x in phases],
        'original_post_contract':load(a.run/'qualification-context.json')['post_contract']}
if a.out:
 with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report,indent=2))
