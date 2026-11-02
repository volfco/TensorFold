#!/usr/bin/env bash
set -u
cd "$HOME/tensorfold-kernel-night/epilogue-tests"
export LD_LIBRARY_PATH="$HOME/tensorfold-kernel-night/runtime:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TF_EXL3_N256=0 TF_EXL3_FUSED_EPILOGUES=0 TF_GLM_KDA_VECTOR=0
exec 9>>/tmp/atlas-bench.lock
if ! flock -n 9; then
    printf 'GPU lock busy\n' >&2
    exit 75
fi
date -u +%Y-%m-%dT%H:%M:%SZ > started.txt
sha256sum tf-cuda-test > runner.sha256
uname -m > architecture.txt
uname -r > kernel.txt
nvidia-smi --query-gpu=name,driver_version,utilization.gpu,temperature.gpu,power.draw --format=csv > gpu-before.csv
night_status=0
for night_test in exl3-fused exl3-fused-contract; do
    timeout 1800 ./tf-cuda-test "$night_test" > "$night_test.log" 2>&1
    night_rc=$?
    printf '%s\n' "$night_rc" > "$night_test.exit"
    printf '%s exit=%s\n' "$night_test" "$night_rc"
    if [ "$night_rc" != 0 ]; then night_status=$night_rc; fi
    if [ "$night_rc" = 124 ]; then break; fi
done
nvidia-smi --query-gpu=name,driver_version,utilization.gpu,temperature.gpu,power.draw --format=csv > gpu-after.csv
date -u +%Y-%m-%dT%H:%M:%SZ > finished.txt
printf '%s\n' "$night_status" > campaign.exit
exit "$night_status"
