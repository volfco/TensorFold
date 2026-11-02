#!/usr/bin/env python3
"""Derive privacy-reviewed cache facts from the original raw evidence, without network or GPU work."""
import hashlib,json,pathlib
RUN='paired-20261030-r2-cache-resume'
CAMP='paired-20261030-r2'
TESTED_LIMITS='49d3db427b3bb04e1ec23ef0c727a91670cb5634e1496b0b1cd79393b6f700eb'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def encoded(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()

def derive(raw,archive):
    run=raw/'receipts'/RUN
    count=0
    for line in (run/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);path=run/name
        if pathlib.PurePosixPath(name).is_absolute() or '..' in pathlib.PurePosixPath(name).parts or sha(path)!=digest:
            raise ValueError('raw manifest mismatch')
        count+=1
    summary=load(run/'final-summary.json');wire=load(run/'all-wire-audit.json');context=load(run/'qualification-context.json')
    if (summary.get('scope')!='cache-correctness-only' or summary.get('cache_passed') is not True or
        summary.get('cache_exit')!=0 or summary.get('completed_replies')!=24 or summary.get('raw_wire_responses')!=54 or
        summary.get('positive_warm_reuses')!=9 or summary.get('fresh_strict_zero_counters')!=12 or
        summary.get('all_phases_cleaned') is not True or summary.get('both_gpu_locks_free') is not True or
        summary.get('original_post_receipts_unchanged') is not True or wire.get('passed') is not True):
        raise ValueError('cache qualification is incomplete')
    if summary['post_contract']!=context['post_contract'] or summary['post_contract']['overall_exit']!=1 or summary['post_contract']['overall_contract_passed'] is not False:
        raise ValueError('original red post-contract status was changed')
    phases=[];readiness=[];canonical={};positive=fresh=responses=0
    for phase in wire['phases']:
        name=phase['phase'];mode,role=name.split('-');folder=run/name
        if phase.get('passed') is not True or phase.get('cleanup_verified') is not True or len(phase['replies'])!=4:
            raise ValueError('incomplete phase audit')
        ready=load(folder/'ready.json');cleanup=load(folder/'cleanup.json');complete=load(folder/'complete.json')
        if (complete!={'client_exit':0,'ready_verified':True,'cleanup_cleared':True,'error':None} or
            cleanup.get('cleared') is not True or ready.get('native_uid_memlock_verified') is not True or ready.get('roce_verified') is not True):
            raise ValueError('readiness or cleanup did not pass')
        for row in phase['replies']:
            if type(row['cached']) is not int or not 0<row['tokens']<=32:raise ValueError('invalid counters')
            pair=(row['request_sha256'],row['token_sha256'],row['prompt'],row['tokens'])
            if row['i'] in canonical and canonical[row['i']]!=pair:raise ValueError('frozen workload or full output changed')
            canonical[row['i']]=pair
            if role=='fresh':
                if row['cached']!=0:raise ValueError('fresh reuse is not zero')
                fresh+=1
            elif row['i']==0:
                if row['cached']!=0:raise ValueError('initial warm request was not cold')
            elif row['cached']<=0:raise ValueError('warm reuse witness missing')
            else:positive+=1
        if [x['i'] for x in phase['replies']]!=list(range(4)):raise ValueError('request identities incomplete')
        for rank in [0,1]:
            info=ready['ranks'][rank];native=[x for x in info['processes'] if x['exe']=='tensorfold-native']
            if len(native)!=1:raise ValueError('actual native proof missing')
            n=native[0];uid=list(map(int,n['status']['Uid'].split()))
            locked=[x for x in n['limits'].splitlines() if x.startswith('Max locked memory')]
            env=n['env'];flags={k:env[k] for k in ['TF_GLM_KDA_VECTOR','TF_EXL3_N256','TF_EXL3_FUSED_EPILOGUES']}
            expected={'vector':['1','0','0'],'n256':['0','1','0'],'fusion':['0','0','1']}[mode]
            if (uid!=[1000]*4 or len(locked)!=1 or locked[0].split()[-3:-1]!=['unlimited','unlimited'] or
                list(flags.values())!=expected or env['TF_ROCE']!='1' or env['NCCL_IB_DISABLE']!='0'):
                raise ValueError('actual native user, memory limit or transport validation failed')
            done=cleanup['ranks'][rank]
            if done.get('remaining',{}).get('processes')!=[]:raise ValueError('owned processes remain')
            wrapper=load(folder/f'rank{rank}.memlock.json')
            if wrapper!={'uid':1000,'euid':1000,'memlock':[-1,-1]}:raise ValueError('wrapper guard did not preserve user/limits')
            readiness.append({'phase':name,'rank':rank,'cache_gib':ready['cache_gib'],'health':ready['health']['status'],
                'served_model_verified':any(x['id']=='kernel-research' for x in ready['models']['data']),
                'actual_native_uid_fields':uid,'native_memlock_soft':'unlimited','native_memlock_hard':'unlimited',
                'wrapper_guard':wrapper,'feature_flags':flags,'transport':{'TF_ROCE':env['TF_ROCE'],'TF_ROCE_MAX_KB':env['TF_ROCE_MAX_KB'],'NCCL_IB_DISABLE':env['NCCL_IB_DISABLE'],'positive_two_hca_witness':True},
                'serve_arguments':info['effective_serve_arguments'],'cleanup':{'cleared':True,'remaining_process_count':0,'term_action':done.get('term',{}).get('cleanup'),'kill_action':done.get('kill',{}).get('cleanup'),'transport_exit':done.get('transport_exit')},
                'raw_ready_sha256':sha(folder/'ready.json'),'raw_cleanup_sha256':sha(folder/'cleanup.json'),'raw_complete_sha256':sha(folder/'complete.json')})
        phases.append(phase);responses+=phase['raw_wire_responses']
    if (len(phases),positive,fresh,responses)!=(6,9,12,54):raise ValueError('cache matrix incomplete')
    for mode in ['vector','n256','fusion']:
        if load(run/(mode+'-comparison.json'))!={'passed':True,'requests':4,'positive_warm_reuses_required':3,'fresh_zero_counters_required':4,'problems':[]}:
            raise ValueError('mode comparison failed')
    released=load(run/'final-released-nodes.json');release=[]
    for info in released:
        if info.get('lock_free') is not True or info['probe']['processes']:raise ValueError('final GPU lease release incomplete')
        release.append({'rank':info['rank'],'lock_free':True,'model_or_gpu_test_process_count':0,'gpu_query':info['probe']['gpu']})
    if len(release)!=2:raise ValueError('both nodes must be released')
    auth=load(raw/'receipts'/CAMP/'cache-handoff.authorization.json')
    source=raw/'candidate/zig/tests/cuda/contract_limits.zig'
    if sha(source)!=TESTED_LIMITS:raise ValueError('tested source pin changed')
    privacy={'scope':'Public synthetic system guide and synthetic tool replies; thinking disabled. No user or repository content supplied.',
        'excluded_from_repository':['raw request/response payloads','rendered token arrays','node hostnames and non-loopback network addresses','usernames and absolute home/model paths','process IDs and RDMA/interface identifiers','operational logs'],
        'retained':['all 24 per-request full output-token hashes and request-body hashes','typed cache counters, prompt and completion counts','all 12 rank readiness/cleanup facts','both final physical process/lock release facts','tested binary/tool/source hashes','original red contract exits, summaries and raw-log hashes'],
        'portable_tool_changes':'SSH alias worker; repository-relative offline fixture path; node-local launcher configuration; reject disabled audit assertions. Original tools and tested source snapshots remain in the separate raw archive.',
        'raw_archive_sha256':archive['sha256'],'raw_manifest_entries_verified':count,'raw_evidence_unchanged':True,
        'committed_scan':{'non_loopback_ipv4_files':0,'absolute_home_or_model_path_files':0,'credential_pattern_files':0}}
    provenance={'campaign':CAMP,'run':RUN,'source_base':'d33dcfc767ee093393c6bfa00b3949bd7b4d8e88','hardware':{'ranks':2,'gpu':'NVIDIA GB10','sm':121,'driver':'580.159.03','nccl':'2.30.7'},
        'checkpoint':{'repository':'Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit','revision':'57edefd2f5d9b371c8345883304d5af68b52fa24'},
        'raw_archive':archive,'raw_manifest_sha256':sha(run/'SHA256SUMS'),'raw_manifest_entries_verified':count,
        'raw_identity_sha256':sha(run/'identity.json'),'tested_tool_and_binary_identities':load(run/'identity.json'),
        'reviewed_current_research_sources':auth['reviewed_current_research_sources'],'tested_contract_limits_sha256':TESTED_LIMITS,
        'post_qualification_comment_revision':{'current_sha256':'bd10c46e39c9d70af7aaa35266e4caa633fca118d55e88ff8119e2aac53e671d','scope':'One comment-only line; no binary rebuild or GPU retest. Tested source anchor remains unchanged.','receipt':'../../glm-kernel-post-contract-d33dcfc7/source-revisions.json'},
        'authorization_sha256':sha(raw/'receipts'/CAMP/'cache-handoff.authorization.json'),'raw_final_summary_sha256':sha(run/'final-summary.json'),
        'raw_all_wire_audit_sha256':sha(run/'all-wire-audit.json'),'raw_final_release_sha256':sha(run/'final-released-nodes.json')}
    return {'summary.json':summary,'qualification.json':context,'audits.json':{'scope':'cache-correctness-only','phases':phases},
            'readiness-cleanup.json':{'ranks':readiness,'final_release':release},'privacy-review.json':privacy,'provenance.json':provenance,
            'comparisons.json':{x:load(run/(x+'-comparison.json')) for x in ['vector','n256','fusion']}}
