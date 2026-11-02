// TF_NVFP4_DECODE_RING (cuda_expert_ring.zig): the NVFP4 decode expert kernel with D k32 groups of loads in flight in
// registers instead of nvfp4_experts.cu's two. UNBUILT AND UNTESTED: written without nvcc; not in
// zig/build/cuda_kernels.zig's list (built by hand, tools/kvprobe/STATUS-track4.md).
//
// Same bits by construction: k_loop_ring computes the groups g = 0, 1, ..., KG - 1 in order into the same
// accumulators with the same compute_stage, so only when each group's loads are issued changes. The cost to watch is
// registers: a Stage<M> is 8 M + 8 32-bit registers a lane (gate-up, M = 2: 24), so D = 4 holds 96 for loads alone and
// may spill or lower occupancy; tf-cuda-test kernel-contract-limits on the built image says which.
//
// This file includes nvfp4_experts.cu for its device helpers, so the image also carries that file's instances; the
// engine loads only the tf_nvfp4_ring_* entries from it.

#include "nvfp4_experts.cu"

namespace tf_nvfp4_experts {

template <int M, int D>
__device__ __forceinline__ void k_loop_ring(float (&acc)[M][1][NTW][4], const uint4* blk, int KG, int lane, int t,
                                            const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  static_assert(D >= 2 && D <= 4, "a ring of 2 to 4 groups");
  Stage<M> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<M>(st[d], blk, d, lane, t, x0, x1, v0, v1);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        compute_stage<M>(acc, st[d], v1);
        if (g + D < KG) load_stage<M>(st[d], blk, g + D, lane, t, x0, x1, v0, v1);
      }
    }
  }
}

// unit_full with k_loop_ring: every other line as nvfp4_experts.cu's.
template <int M, int EPI, int D>
__device__ __forceinline__ void unit_ring(int unit, int lane, int gq, int t, const __nv_bfloat16* __restrict__ X,
                                          int x_stride, int slots, const uint4* __restrict__ W,
                                          const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                                          const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                                          int skip) {
  const int it = unit / NB, cb = unit - it * NB;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  if (e == skip) return;
  const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
  for (int r0 = 0; r0 < cnt; r0 += 16) {
    const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
    const int pr0 = v0 ? __ldg(members + first + r0 + gq) : 0;
    const int pr1 = v1 ? __ldg(members + first + r0 + gq + 8) : 0;
    const int x0r = slots ? pr0 / slots : pr0, x1r = slots ? pr1 / slots : pr1;
    const __nv_bfloat16* x0 = X + (size_t)x0r * x_stride;
    const __nv_bfloat16* x1 = X + (size_t)x1r * x_stride;
    float acc[M][1][NTW][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][0][j][0] = acc[m][0][j][1] = acc[m][0][j][2] = acc[m][0][j][3] = 0.f;
    k_loop_ring<M, D>(acc, blk, KG, lane, t, x0, x1, v0, v1);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float g = __ldg(scale + e * M + m);
#pragma unroll
      for (int j = 0; j < NTW; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[m][0][j][q] *= g;
    }
    epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * t, pr0, pr1, v0, v1, limit);
  }
}

template <int M, int EPI, int D>
__device__ __forceinline__ void ring_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                            const uint4* __restrict__ W, const float* __restrict__ scale, int KG,
                                            int NB, const int* __restrict__ items, const int* __restrict__ counts,
                                            const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                                            int skip) {
  constexpr int WARPS = 4;
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS)
    unit_ring<M, EPI, D>(unit, lane, gq, t, X, x_stride, slots, W, scale, KG, NB, items, members, out, N, limit, skip);
}

}  // namespace tf_nvfp4_experts

// nvfp4_expert_kernel<M, EPI, 4>'s arguments, in its order; GLM's decode takes gate-up (M 2, EPI 2) and fp32 down
// (M 1, EPI 0). D = 2 is built too: it must equal the stock kernel's bits AND its time (the harness's own control).
#define TF_RING(NAME, M, EPI, D)                                                                                        \
  extern "C" __global__ void __launch_bounds__(128)                                                                    \
      NAME(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,                  \
           const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,                             \
           const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N, float limit, \
           int skip) {                                                                                                  \
    tf_nvfp4_experts::ring_kernel<M, EPI, D>(X, x_stride, slots, W, scale, KG, NB, items, counts, members, out, N,     \
                                             limit, skip);                                                             \
  }

TF_RING(tf_nvfp4_ring_gate_up_d2, 2, 2, 2)
TF_RING(tf_nvfp4_ring_gate_up_d3, 2, 2, 3)
TF_RING(tf_nvfp4_ring_gate_up_d4, 2, 2, 4)
TF_RING(tf_nvfp4_ring_down_f32_d2, 1, 0, 2)
TF_RING(tf_nvfp4_ring_down_f32_d3, 1, 0, 3)
TF_RING(tf_nvfp4_ring_down_f32_d4, 1, 0, 4)
