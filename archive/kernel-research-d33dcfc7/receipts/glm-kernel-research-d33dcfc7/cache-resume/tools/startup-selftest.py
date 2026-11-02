#!/usr/bin/env python3
"""CPU-only negative fixtures for actual-native user, memlock and transport checks."""
import copy,importlib.util,json,pathlib
spec=importlib.util.spec_from_file_location('controller',pathlib.Path(__file__).with_name('cache-resume-controller.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
uid,flags,phase=1000,(1,0,0),'vector-warm'
native={'limits':'Max locked memory         unlimited            unlimited            bytes\n',
        'status':{'Uid':'1000 1000 1000 1000'},'env':{'TF_GLM_RANK':'0','TF_GLM_WORLD':'2','TF_GLM_PROF':'0',
        'TF_ROCE':'1','TF_ROCE_MAX_KB':'512','NCCL_IB_DISABLE':'0','TF_GLM_KDA_VECTOR':'1','TF_EXL3_N256':'0',
        'TF_EXL3_FUSED_EPILOGUES':'0','NIGHT_CAMPAIGN_ID':phase}}
c.validate_native(native,uid,0,flags,phase);assert c.validate_transport([c.ROCE_READY,c.ROCE_READY])
controls=0
for path,value in [('limits',''),('limits','Max locked memory 8388608 unlimited bytes'),
                   ('limits','Max locked memory unlimited 8388608 bytes'),('Uid','1000 0 0 0'),
                   ('Uid','1000 1000'),('TF_ROCE','0'),('TF_ROCE_MAX_KB','256'),('NCCL_IB_DISABLE','1'),
                   ('TF_GLM_KDA_VECTOR','0'),('NIGHT_CAMPAIGN_ID','other-phase')]:
 bad=copy.deepcopy(native)
 if path=='limits':bad['limits']=value
 elif path=='Uid':bad['status']['Uid']=value
 else:bad['env'][path]=value
 try:c.validate_native(bad,uid,0,flags,phase)
 except RuntimeError:controls+=1
 else:raise AssertionError('invalid native startup accepted: '+path)
for marker in c.R2_FATAL_MARKERS:
 try:c.validate_transport([c.ROCE_READY,marker+'\n'+c.ROCE_READY])
 except RuntimeError:controls+=1
 else:raise AssertionError('fatal startup accepted: '+marker)
assert not c.validate_transport([c.ROCE_READY,'missing positive witness']);controls+=1
print(json.dumps({'offline_startup_selftest':'pass','native_uid_limits_transport_controls_rejected':controls,'gpu_work_started':False}))
