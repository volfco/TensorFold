#!/usr/bin/env bash
set -uo pipefail
night_rank=${1:?rank}
night_variant=${2:?variant}
night_out=${3:?output}
night_vector=${4:?vector}
night_n256=${5:?n256}
night_fusion=${6:?fusion}
mkdir -p "$night_out" || exit 1
printf '%s\n' "$$" > "$night_out/rank$night_rank.pgid" || exit 1
case ${NIGHT_REUSE_CACHE_GIB-} in 0|1) ;; *) echo "refusing unreviewed cache budget" >&2; exit 2 ;; esac
night_rc=0
trap 'night_rc=$?; printf "%s\n" "$night_rc" > "$night_out/rank$night_rank.wrapper.exit"; date -u +%Y-%m-%dT%H:%M:%SZ > "$night_out/rank$night_rank.stopped.txt"' EXIT
trap ':' TERM INT
export TF_GLM_PROF=0 TF_GLM_KDA_VECTOR=$night_vector TF_EXL3_N256=$night_n256 TF_EXL3_FUSED_EPILOGUES=$night_fusion
export NIGHT_PROMPT_CACHE_GIB=${NIGHT_REUSE_CACHE_GIB:?explicit-cache-budget} NIGHT_CAMPAIGN_ID=${night_out##*/}
night_uid=$(id -u)
if [[ $(ulimit -Sl) != unlimited || $(ulimit -Hl) != unlimited ]]; then
    sudo -n prlimit --pid $$ --memlock=unlimited:unlimited || { echo 'memlock validation failed' >&2; exit 1; }
fi
if [[ $(id -u) != "$night_uid" || $(ulimit -Sl) != unlimited || $(ulimit -Hl) != unlimited ]]; then
    echo 'memlock validation failed' >&2
    exit 1
fi
python3 -c 'import json,os,resource,sys; json.dump({"uid":os.getuid(),"euid":os.geteuid(),"memlock":resource.getrlimit(resource.RLIMIT_MEMLOCK)},open(sys.argv[1],"w"))' "$night_out/rank$night_rank.memlock.json" || exit 1
bash "$HOME/tensorfold-kernel-night/launch-native.sh" "$night_rank" "$night_variant" &
night_child=$!
printf '%s\n' "$night_child" > "$night_out/rank$night_rank.child.pid"
wait "$night_child"
night_rc=$?
if kill -0 "$night_child" 2>/dev/null; then
    wait "$night_child"
    night_rc=$?
fi
printf '%s\n' "$night_rc" > "$night_out/rank$night_rank.child.exit"
exit "$night_rc"
