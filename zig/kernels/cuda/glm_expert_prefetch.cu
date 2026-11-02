// TF_GLM_EXPERT_PREFETCH (cuda_expert_prefetch.zig): after the top-k pick, the head of every column-block run of each
// picked NVFP4 expert into L2. UNBUILT AND UNTESTED: written without nvcc; not in zig/build/cuda_kernels.zig's list.
//
// Loads only. Nothing here writes memory the model reads, so no output bit can change; only where the decode expert
// kernel (nvfp4_experts.cu, unit_full) finds its first stages. The mechanism follows the Python stack's L2 prefetch
// patch: cp.async.bulk.prefetch.L2.global (sm_90+, GB10 is sm_121) or one prefetch.global.L2 a 128-byte line.
//
// Layout (cuda.experts.Layer, nvfp4): expert e's column block cb starts at base + (e * nb + cb) * run bytes, and the
// decode kernel walks each run from its front. Block u of the grid takes picks u, u + gridDim.x, ...; its threads take
// the runs of both matrices. A pick outside [0, experts) (the shared slot's id is `experts`) is skipped, so no address
// outside the expert tensors is ever formed.

#include <stdint.h>

namespace tf_glm_expert_prefetch {

constexpr uint32_t LINE = 128;
constexpr uint32_t BULK_MAX = 1u << 20;  // one bulk instruction's bytes here (a head is a few KiB)

__device__ __forceinline__ void bulk(const uint8_t* p, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
#else
  for (uint32_t o = 0; o < bytes; o += LINE) asm volatile("prefetch.global.L2 [%0];" ::"l"(p + o));
#endif
}

__device__ __forceinline__ void lines(const uint8_t* p, uint32_t bytes) {
  for (uint32_t o = 0; o < bytes; o += LINE) asm volatile("prefetch.global.L2 [%0];" ::"l"(p + o));
}

template <bool Bulk>
__device__ __forceinline__ void matrix(const uint8_t* base, int64_t e, int64_t nb, int64_t run, int64_t head) {
  const uint8_t* expert = base + e * nb * run;
  const uint32_t h = static_cast<uint32_t>(head < BULK_MAX ? head : BULK_MAX);
  for (int64_t cb = threadIdx.x; cb < nb; cb += blockDim.x) {
    const uint8_t* p = expert + cb * run;
    if (Bulk)
      bulk(p, h);
    else
      lines(p, h);
  }
}

template <bool Bulk>
__device__ __forceinline__ void body(const int32_t* __restrict__ picks, int n, int experts, const uint8_t* up,
                                     int64_t up_nb, int64_t up_run, int64_t up_head, const uint8_t* down,
                                     int64_t down_nb, int64_t down_run, int64_t down_head) {
  for (int u = blockIdx.x; u < n; u += gridDim.x) {
    const int e = picks[u];
    if (e < 0 || e >= experts) continue;
    matrix<Bulk>(up, e, up_nb, up_run, up_head);
    matrix<Bulk>(down, e, down_nb, down_run, down_head);
  }
}

}  // namespace tf_glm_expert_prefetch

extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expert_prefetch_bulk(const int32_t* __restrict__ picks, int n, int experts, const uint8_t* up, int64_t up_nb,
                                int64_t up_run, int64_t up_head, const uint8_t* down, int64_t down_nb, int64_t down_run,
                                int64_t down_head) {
  tf_glm_expert_prefetch::body<true>(picks, n, experts, up, up_nb, up_run, up_head, down, down_nb, down_run, down_head);
}

extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expert_prefetch_lines(const int32_t* __restrict__ picks, int n, int experts, const uint8_t* up,
                                 int64_t up_nb, int64_t up_run, int64_t up_head, const uint8_t* down, int64_t down_nb,
                                 int64_t down_run, int64_t down_head) {
  tf_glm_expert_prefetch::body<false>(picks, n, experts, up, up_nb, up_run, up_head, down, down_nb, down_run,
                                      down_head);
}

// Bench only (glm_expert_prefetch_bench.zig): a single thread that spins for `cycles` clock cycles, standing in for the
// work between the pick and the routed experts (the plan, the shared expert) that a prefetch could hide behind.
extern "C" __global__ void tf_glm_expert_prefetch_spin(long long cycles) {
  const long long start = clock64();
  while (clock64() - start < cycles) {
  }
}
