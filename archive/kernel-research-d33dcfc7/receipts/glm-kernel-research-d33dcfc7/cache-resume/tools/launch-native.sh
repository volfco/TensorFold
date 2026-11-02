#!/usr/bin/env bash
set -euo pipefail
night_rank=${1:?rank}
night_variant=${2:?variant}
night_root=$HOME/tensorfold-kernel-night
night_binary=$night_root/$night_variant/zig-out/native/bin/tensorfold-native
source "$night_root/cache-node.env"
night_model=${NIGHT_MODEL:?pinned-checkpoint-directory}
export LD_LIBRARY_PATH=$night_root/runtime:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TF_GLM_RANK=$night_rank TF_GLM_WORLD=2 NCCL_COMM_ID=${NIGHT_NCCL_COMM_ID:?rank-zero-endpoint}
export NCCL_IB_DISABLE=0 NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=${NIGHT_SOCKET_IFNAME:?node-local-interface} TF_ROCE_HCA=${NIGHT_ROCE_HCA:?two-node-local-HCAs}
export NCCL_IB_HCA=$TF_ROCE_HCA TF_ROCE=1 TF_ROCE_MAX_KB=512
export TENSORFOLD_CUDA_KERNELS=native TF_GLM_PROF=${TF_GLM_PROF:-1} TF_HEAT_HIGH=90 TF_HEAT_LOW=85
export TF_EXL3_N256=${TF_EXL3_N256:-0} TF_EXL3_FUSED_EPILOGUES=${TF_EXL3_FUSED_EPILOGUES:-0}
export TF_GLM_KDA_VECTOR=${TF_GLM_KDA_VECTOR:-0}
exec flock -n /tmp/atlas-bench.lock taskset -c 0-4,10-14 "$night_binary" serve "$night_model" \
  --backend cuda --host 127.0.0.1 --port 18080 --name kernel-research --context 131072 \
  --parallel 4 --prompt-cache-gib "${NIGHT_PROMPT_CACHE_GIB:-0}" --keep-warm 0 --idle-unload 0 --no-thinking --no-update-check
