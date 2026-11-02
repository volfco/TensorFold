import argparse,json,os,pathlib,signal,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--pgid',type=int,default=0);p.add_argument('--cleanup',choices=['TERM','KILL']);p.add_argument('--mode-dir',default='');a=p.parse_args()
r={'time':time.time(),'pgid':a.pgid,'processes':[],'thermal':{},'memory':{},'rdma':{}}
keys={'TF_GLM_RANK','TF_GLM_WORLD','TF_GLM_PROF','TF_GLM_KDA_VECTOR','TF_EXL3_N256','TF_EXL3_FUSED_EPILOGUES','TF_HEAT_HIGH','TF_HEAT_LOW','TF_ROCE','TF_ROCE_HCA','TF_ROCE_MAX_KB','NCCL_IB_DISABLE','NCCL_IB_HCA','NCCL_SOCKET_IFNAME','NIGHT_CAMPAIGN_ID'}
if a.cleanup and a.pgid<=0:raise SystemExit('refusing nonpositive process group')
own=False
for d in pathlib.Path('/proc').iterdir():
    if not d.name.isdigit():continue
    try:
        pid=int(d.name)
        if a.pgid and os.getpgid(pid)!=a.pgid:continue
        argv=(d/'cmdline').read_bytes().split(b'\0')
        if a.mode_dir and a.mode_dir.encode() in argv:own=True
        exe=os.path.basename(os.readlink(d/'exe'))
        if not a.pgid and 'tensorfold' not in exe and exe!='tf-cuda-test':continue
        status={}
        for line in (d/'status').read_text().splitlines():
            k,_,v=line.partition(':')
            if k in {'Name','State','VmRSS','VmHWM','VmSize','VmSwap','Uid','Gid'}:status[k]=v.strip()
        env={}
        if exe=='tensorfold-native':
            for item in (d/'environ').read_bytes().split(b'\0'):
                k,_,v=item.partition(b'=')
                if k.decode(errors='replace') in keys:env[k.decode()]=v.decode(errors='replace')
        if a.mode_dir and env.get('NIGHT_CAMPAIGN_ID')==pathlib.Path(a.mode_dir).name and exe=='tensorfold-native':own=True
        r['processes'].append({'pid':pid,'pgid':os.getpgid(pid),'exe':exe,'status':status,'env':env,'limits':(d/'limits').read_text() if exe=='tensorfold-native' else ''})
    except (OSError,ValueError,ProcessLookupError):pass
if a.cleanup:
    if not r['processes']:r['cleanup']='already-absent'
    elif not own:r['cleanup']='refused-unowned-group'
    else:
        targets=[x['pid'] for x in r['processes'] if x['exe']=='tensorfold-native' and x['env'].get('NIGHT_CAMPAIGN_ID')==pathlib.Path(a.mode_dir).name]
        if a.cleanup=='TERM' and targets:
            for pid in targets:os.kill(pid,signal.SIGTERM)
            r['cleanup']='sent-TERM-native';r['targets']=targets
        else:os.killpg(a.pgid,getattr(signal,'SIG'+a.cleanup));r['cleanup']='sent-'+a.cleanup
    print(json.dumps(r));raise SystemExit(0 if r['cleanup']!='refused-unowned-group' else 2)
for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
    k,_,v=line.partition(':')
    if k in {'MemTotal','MemAvailable','SwapTotal','SwapFree'}:r['memory'][k]=v.strip()
for z in pathlib.Path('/sys/class/thermal').glob('thermal_zone*'):
    try:r['thermal'][z.name]={'type':(z/'type').read_text().strip(),'millidegrees':int((z/'temp').read_text())}
    except (OSError,ValueError):pass
try:
    x=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=8)
    r['gpu']={'exit':x.returncode,'text':x.stdout.strip(),'error':x.stderr.strip()}
except Exception as e:r['gpu']={'error':repr(e)}
for h in pathlib.Path('/sys/class/infiniband').glob('*'):
    counters={}
    for name in ['port_xmit_data','port_rcv_data','port_xmit_packets','port_rcv_packets']:
        try:counters[name]=int((h/'ports/1/counters'/name).read_text())
        except (OSError,ValueError):pass
    r['rdma'][h.name]=counters
print(json.dumps(r))
