// N256 MCg-4 expert products with the N128 kernel's K slices, MMA sequence and ordered warp fold.
#pragma once

namespace tf_exl3 {

__device__ __forceinline__ int n256_row(const GroupedArgs& a, int u, int m) {
    const int count = a.standard ? a.uids[3 * u + 2] : a.maxm;
    if (m >= count) return -1;
    const int first = a.standard ? a.uids[3 * u + 1] : u * a.maxm;
    const int code = a.members[first + m];
    return a.standard ? code : (code >= 0 ? (code >> 5) * a.slots + (code & 31) : -1);
}

__device__ __forceinline__ void grouped_n256_body(const GroupedArgs a) {
    const int u = blockIdx.x;
    if (u >= a.ucount[0]) return;
    const int mt = (a.maxm + 15) / 16;
    const int member0 = (blockIdx.z % mt) * 16;
    if (n256_row(a, u, member0) < 0) return;
    const int split = (blockIdx.z / mt) % a.sk;
    const int mat = blockIdx.z / mt / a.sk;
    const int e = a.standard ? a.uids[3 * u] : a.uids[u];
    if (a.standard && e == a.skip) return;
    if ((mat ? a.k2_1[e] : a.k2_0[e]) != 8) __trap();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int r0 = n256_row(a, u, member0 + g);
    const int r1 = n256_row(a, u, member0 + g + 8);
    const half* x = mat ? a.x1 : a.x0;
    const half* x0 = x + (size_t)(r0 < 0 ? 0 : r0) * a.k + 2 * t;
    const half* x1 = x + (size_t)(r1 < 0 ? 0 : r1) * a.k + 2 * t;
    const uint32_t* tr = reinterpret_cast<const uint32_t*>(mat ? a.tp1[e] : a.tp0[e]);
    const int per_warp = (a.k >> 4) / a.sk / 4;
    const int kt0 = split * (per_warp * 4) + warp * per_warp;
    const int nt0 = blockIdx.y * 16;
    float acc[16][2][4];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][h][j] = 0.f;
    warp_tiles<1, 8, 16>(tr, a.ns > 0 ? a.ns : (a.n >> 4), kt0, per_warp, nt0,
                        x0, x1, r0 >= 0, r1 >= 0, lane, acc);
    __shared__ __align__(16) float red[3][16][256];
    if (warp > 0) {
#pragma unroll
        for (int i = 0; i < 16; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    red[warp - 1][g + (j >> 1) * 8][i * 16 + h * 8 + 2 * t + (j & 1)] = acc[i][h][j];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int i = 0; i < 16; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = j < 2 ? r0 : r1;
                    if (r < 0) continue;
                    const int col = i * 16 + h * 8 + 2 * t + (j & 1);
                    float v = acc[i][h][j];
#pragma unroll
                    for (int w = 0; w < 3; ++w) v += red[w][g + (j >> 1) * 8][col];
                    a.z[(((size_t)mat * a.sk + split) * a.p + r) * a.n + nt0 * 16 + col] = v;
                }
    }
}

}  // namespace tf_exl3

extern "C" __global__ void __launch_bounds__(128) tf_exl3_grouped_256_1(tf_exl3::GroupedArgs a) {
    tf_exl3::grouped_n256_body(a);
}
