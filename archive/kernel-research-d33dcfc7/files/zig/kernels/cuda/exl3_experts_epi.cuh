// Common EXL3 decode tails: the existing partials and epilogues, with scratch-owned last-arrival counters.
#pragma once

namespace tf_exl3 {

struct EpiArgs {
    const int* pick;
    const half *sv0, *sv1, *sd;
    half* xd;
    float* y;
    int* ticket;
    int experts;
    float limit;
    int act_mode;
};

struct FusedArgs { GroupedArgs g; EpiArgs e; };

__device__ __forceinline__ void fused_gateup_row(const GroupedArgs& a, const EpiArgs& ep, int p, int e,
                                               int blk, int lane) {
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < a.sk; ++s) {
            sg += __ldcg(a.z + ((size_t)s * a.p + p) * a.n + n + j);
            su += __ldcg(a.z + ((size_t)(a.sk + s) * a.p + p) * a.n + n + j);
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (ep.act_mode == 0) {
            float gg = bf16r(gv[j] * HAD_SCALE * __half2float(ep.sv0[(size_t)e * a.n + n + j]));
            float uu = bf16r(uv[j] * HAD_SCALE * __half2float(ep.sv1[(size_t)e * a.n + n + j]));
            if (ep.limit > 0) { gg = upTo(gg, ep.limit); uu = within(uu, ep.limit); }
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = gv[j] * HAD_SCALE * __half2float(ep.sv0[(size_t)e * a.n + n + j]);
            float uu = uv[j] * HAD_SCALE * __half2float(ep.sv1[(size_t)e * a.n + n + j]);
            if (ep.limit > 0) { gg = upTo(gg, ep.limit); uu = within(uu, ep.limit); }
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(ep.sd[(size_t)e * a.n + n + j]);
    }
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) ep.xd[(size_t)p * a.n + n + j] = __float2half_rn(v[j] * HAD_SCALE);
}

template <int CB, bool GU>
__device__ __forceinline__ void fused_body(const FusedArgs& f) {
    const GroupedArgs& a = f.g;
    const EpiArgs& ep = f.e;
    const int u = blockIdx.x, blk = blockIdx.y;
    if (u >= a.ucount[0]) return;
    const int mt = (a.maxm + 15) / 16, tile = blockIdx.z % mt;
    const int first = tile * 16;
    if (first >= a.maxm || a.members[u * a.maxm + first] < 0) return;
    grouped_body<CB, 8, 4>(a, u, blk, blockIdx.z);
    __threadfence();
    __syncthreads();
    if constexpr (GU) {
        __shared__ int last;
        if (threadIdx.x == 0) {
            int* tk = ep.ticket + ((size_t)u * mt + tile) * gridDim.y + blk;
            last = atomicAdd(tk, 1) == 2 * a.sk - 1;
            if (last) *tk = 0;
        }
        __syncthreads();
        if (!last) return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int row = first + warp; row < first + 16 && row < a.maxm; row += 4) {
        const int code = a.members[u * a.maxm + row];
        if (code < 0) continue;
        const int p = (code >> 5) * a.slots + (code & 31), e = ep.pick[p];
        if (e < 0 || e >= ep.experts) continue;
        if constexpr (GU) {
            fused_gateup_row(a, ep, p, e, blk, lane);
        } else {
            const int n = blk * 128 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = 0.f;
                s += __ldcg(a.z + (size_t)p * a.n + n + j);
                v[j] = s;
            }
            fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                ep.y[(size_t)p * a.n + n + j] = v[j] * HAD_SCALE * __half2float(ep.sv0[(size_t)e * a.n + n + j]);
        }
    }
}

} // namespace tf_exl3

#define TF_EXL3_FUSED(CB) \
extern "C" __global__ void tf_exl3_fused_gu_##CB(tf_exl3::FusedArgs a) { tf_exl3::fused_body<CB, true>(a); } \
extern "C" __global__ void tf_exl3_fused_down_##CB(tf_exl3::FusedArgs a) { tf_exl3::fused_body<CB, false>(a); }
TF_EXL3_FUSED(0)
TF_EXL3_FUSED(1)
TF_EXL3_FUSED(2)
#undef TF_EXL3_FUSED
