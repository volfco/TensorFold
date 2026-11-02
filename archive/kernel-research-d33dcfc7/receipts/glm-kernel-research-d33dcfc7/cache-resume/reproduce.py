#!/usr/bin/env python3
"""Verify the reviewed cache receipt; optionally rederive it from the separate original archive. No GPU/network work."""
import argparse,copy,hashlib,json,pathlib,subprocess,sys,tarfile,tempfile
from derive import derive,encoded
ROOT=pathlib.Path(__file__).parent
FILES=['summary.json','qualification.json','audits.json','readiness-cleanup.json','privacy-review.json','provenance.json','comparisons.json']

def load(path):return json.loads(path.read_text())
def require(value,message):
    if not value:raise ValueError(message)

def validate(data):
    s=data['summary.json'];q=data['qualification.json'];audit=data['audits.json'];ready=data['readiness-cleanup.json']
    require(s['scope']=='cache-correctness-only' and s['cache_passed'] is True and s['cache_exit']==0,'cache scope/result changed')
    require((s['completed_replies'],s['raw_wire_responses'],s['positive_warm_reuses'],s['fresh_strict_zero_counters'])==(24,54,9,12),'matrix totals changed')
    require(s['post_contract']==q['post_contract'] and s['post_contract']['overall_exit']==1 and s['post_contract']['overall_contract_passed'] is False,'original red contract status changed')
    research=q['independent_authorization']['research_preflight']
    require(research['research_passed_record_count']==17 and research['research_failed_record_count']==0,'research resource qualification changed')
    require(len(audit['phases'])==6 and len(ready['ranks'])==12 and len(ready['final_release'])==2,'readiness/cleanup coverage missing')
    refs={};positive=fresh=wire=0
    for phase in audit['phases']:
        require(phase['passed'] is True and phase['cleanup_verified'] is True and len(phase['replies'])==4,'phase failed')
        require([x['i'] for x in phase['replies']]==list(range(4)),'request IDs changed')
        for row in phase['replies']:
            require(type(row['cached']) is int and type(row['tokens']) is int and 0<row['tokens']<=32,'counter type/range changed')
            key=(row['request_sha256'],row['token_sha256'],row['prompt'],row['tokens'])
            require(row['i'] not in refs or refs[row['i']]==key,'request or full token hash differs')
            refs[row['i']]=key
            if phase['phase'].endswith('-fresh'):require(row['cached']==0,'fresh reused prefix');fresh+=1
            elif row['i']==0:require(row['cached']==0,'initial warm was not cold')
            else:require(row['cached']>0,'warm reuse absent');positive+=1
        wire+=phase['raw_wire_responses']
    require((positive,fresh,wire)==(9,12,54),'witness totals differ')
    for rank in ready['ranks']:
        require(rank['actual_native_uid_fields']==[1000]*4 and rank['native_memlock_soft']==rank['native_memlock_hard']=='unlimited','native identity/limits changed')
        require(rank['cleanup']['cleared'] is True and rank['cleanup']['remaining_process_count']==0,'phase cleanup incomplete')
        require(rank['transport']['positive_two_hca_witness'] is True and rank['transport']['TF_ROCE']=='1' and rank['transport']['NCCL_IB_DISABLE']=='0','transport qualification changed')
    for rank in ready['final_release']:require(rank['lock_free'] is True and rank['model_or_gpu_test_process_count']==0,'GPU lease not released')
    require(data['provenance.json']['tested_contract_limits_sha256']=='49d3db427b3bb04e1ec23ef0c727a91670cb5634e1496b0b1cd79393b6f700eb','tested source anchor replaced')

def verify_manifest():
    count=0
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        require('..' not in pathlib.PurePosixPath(name).parts and not pathlib.PurePosixPath(name).is_absolute(),'unsafe manifest path')
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'receipt file changed: '+name);count+=1
    return count

def selftest(data):
    controls=0
    def reject(bad):
        nonlocal controls
        try:validate(bad)
        except (ValueError,KeyError,TypeError):controls+=1;return
        raise AssertionError('false pass accepted')
    for target,value in [('cache_exit',1),('completed_replies',23)]:
        bad=copy.deepcopy(data);bad['summary.json'][target]=value;reject(bad)
    bad=copy.deepcopy(data);bad['qualification.json']['post_contract']['overall_contract_passed']=True;reject(bad)
    bad=copy.deepcopy(data);bad['qualification.json']['independent_authorization']['research_preflight']['research_failed_record_count']=1;reject(bad)
    for phase,index,value in [(0,1,0),(1,0,None),(1,0,False)]:
        bad=copy.deepcopy(data);bad['audits.json']['phases'][phase]['replies'][index]['cached']=value;reject(bad)
    bad=copy.deepcopy(data);bad['audits.json']['phases'][1]['replies'][0]['token_sha256']='0'*64;reject(bad)
    bad=copy.deepcopy(data);bad['readiness-cleanup.json']['ranks'][0]['native_memlock_hard']='8192';reject(bad)
    bad=copy.deepcopy(data);bad['readiness-cleanup.json']['final_release'][0]['lock_free']=False;reject(bad)
    print(json.dumps({'offline_receipt_controls_rejected':controls,'gpu_work_started':False}))

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-archive',type=pathlib.Path);p.add_argument('--selftest',action='store_true');a=p.parse_args()
data={name:load(ROOT/name) for name in FILES};validate(data);count=verify_manifest()
if a.selftest:selftest(data)
if a.raw_archive:
    archive=data['provenance.json']['raw_archive']
    require(hashlib.sha256(a.raw_archive.read_bytes()).hexdigest()==archive['sha256'],'wrong or changed raw archive')
    with tempfile.TemporaryDirectory(prefix='cache-receipt-verify-') as directory:
        raw=pathlib.Path(directory)
        with tarfile.open(a.raw_archive) as tar:
            for member in tar.getmembers():
                require(not pathlib.PurePosixPath(member.name).is_absolute() and '..' not in pathlib.PurePosixPath(member.name).parts,'unsafe archive path')
                path=raw/member.name
                if member.isdir():path.mkdir(parents=True,exist_ok=True)
                else:
                    require(member.isfile(),'links and special archive members refused');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(tar.extractfile(member).read())
        for name,value in derive(raw,archive).items():require(encoded(value)==(ROOT/name).read_bytes(),'rederived receipt differs: '+name)
        result=subprocess.run([sys.executable,str(ROOT/'tools/cache-wire-audit.py'),'--run',str(raw/'receipts/paired-20261030-r2-cache-resume')],capture_output=True,text=True,check=True)
        require(json.loads(result.stdout)==load(raw/'receipts/paired-20261030-r2-cache-resume/all-wire-audit.json'),'raw wire audit differs')
    print(json.dumps({'raw_archive_rederived':True,'raw_wire_audit_reproduced':True}))
print(json.dumps({'receipt_verified':True,'manifest_files':count,'cache_exit':0,'original_contract_exit':1,'overall_contract_passed':False,'gpu_work_started':False}))
