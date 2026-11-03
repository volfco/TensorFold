#!/usr/bin/env bash
# The live-half keep (TF_GLM_KV_LIVE_HALF=1) against the whole arena head on the same binary: lazy codec with the
# ledger and the dump on in both, nine receipt turns each, replies compared with each other and with the earlier
# off run (another binary); production stopped first and restored at the end.
set -u
N=$HOME/kvprobe/native; OUT=$HOME/kvprobe/receipt-half; T=$HOME/kvprobe/TensorFold/tools/kvprobe
log() { echo "=== $1 $(date -u +%H:%M:%S)"; }
mkdir -p $OUT
log "binary $(sha256sum $HOME/kvprobe/TensorFold/zig-out/native/bin/tensorfold-native | cut -c1-16) spark1 $(ssh -o BatchMode=yes 192.168.100.2 sha256sum '~/kvprobe/TensorFold/zig-out/native/bin/tensorfold-native' | cut -c1-16)"
log "stop production"; (cd ~/glm53-tensorfold-spark && env -i HOME=$HOME PATH=$PATH USER=$USER bash scripts/serve.sh stop 2>&1 | tail -1); sleep 25
run_cfg() { # run_cfg NAME "ENV"
  local cfg=$1 penv=$2
  log "config $cfg"; mkdir -p $OUT/$cfg $OUT/dump-$cfg; ssh -o BatchMode=yes 192.168.100.2 "mkdir -p $OUT/dump-$cfg"
  $N/start-ranks2.sh "PROBE_CONTEXT=65536 TF_GLM_HOST_CACHE=1 TF_GLM_KV_LEDGER=1 TF_GLM_KV_DUMP=$OUT/dump-$cfg TF_GLM_KV_CODEC=1 TF_GLM_KV_CODEC_LAZY=1 TF_GLM_KV_CODEC_LAZY_MS=20000 TF_GLM_KV_CODEC_CPUS=5-9 $penv" --no-drafts | tail -2
  grep -m1 -E "KVCODEC rank=0 mode=" $N/rank0.log | cut -c1-160
  if curl -s -m 3 http://127.0.0.1:18080/v1/models | grep -q id; then
    python3 -u $T/lazy_receipt.py run --base http://127.0.0.1:18080 --config $cfg --out $OUT/$cfg --log $N/rank0.log --lazy-ms 20000 2>&1 | tail -12 | cut -c1-200
  fi
  $N/stop-ranks.sh; sleep 3
  cp $N/rank0.log $OUT/$cfg/server-r0.log; ssh -o BatchMode=yes 192.168.100.2 "cat $N/rank1.log" > $OUT/$cfg/server-r1.log
  python3 -u $T/lazy_receipt.py collect $OUT/$cfg --log $OUT/$cfg/server-r0.log --log $OUT/$cfg/server-r1.log 2>&1 | tail -4 | cut -c1-200
  grep -h "KVCODEC rank=0 mode=.*keeps=" $OUT/$cfg/server-r0.log | tail -1 | cut -c1-300
  grep -h -E "KVLEDGER rank=0 keep=[0-9]+ class=prefix" $OUT/$cfg/server-r0.log | sed -E 's/ spans=.*modeled=([0-9]+).*/ modeled=\1/' | head -9
  ls $OUT/dump-$cfg | grep -c "\.plan$"
}
run_cfg both "TF_GLM_KV_LIVE_HALF=0"
run_cfg live "TF_GLM_KV_LIVE_HALF=1"
log "compare"; python3 -u $T/lazy_receipt.py compare $HOME/kvprobe/receipt/off $OUT/both $OUT/live --json $OUT/compare.json 2>&1 | cut -c1-200 | head -40
log "restore production"; (cd ~/glm53-tensorfold-spark && timeout 1200 env -i HOME=$HOME PATH=$PATH USER=$USER bash scripts/serve.sh start 2>&1 | tail -4)
docker ps --format "{{.Names}} {{.Status}}" | grep glm53; curl -s -m 5 http://127.0.0.1:8000/v1/models | grep -o "context_length[^,]*"
log "DONE"
