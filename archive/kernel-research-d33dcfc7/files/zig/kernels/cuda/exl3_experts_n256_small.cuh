// MCg4 down products at up to eight rows: padded MMA rows need no persistent accumulators.
#pragma once

namespace tf_exl3 {

__device__ __forceinline__ void n256_small_mma(float& d0, float& d1, const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
    float padded0, padded1;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%10,%10};\n"
                 : "+f"(d0), "+f"(d1), "=f"(padded0), "=f"(padded1)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.f));
}

__device__ __forceinline__ void grouped_n256_small_body(const GroupedArgs a) {
    if (a.maxm <= 0 || a.maxm > 8 || a.standard != 0 || a.mats != 1 || a.sk != 1 || a.ns != 0 ||
        a.n <= 0 || a.n % 256 != 0 || a.k <= 0 || a.k % 64 != 0) __trap();
    const int u = blockIdx.x;
    if (u >= a.ucount[0]) return;
    if (a.members[u * a.maxm] < 0) return;
    const int e = a.uids[u];
    if (a.k2_0[e] != 8) __trap();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int code = g < a.maxm ? a.members[u * a.maxm + g] : -1;
    const int r = code < 0 ? -1 : (code >> 5) * a.slots + (code & 31);
    const half* x = a.x0 + (size_t)(r < 0 ? 0 : r) * a.k + 2 * t;
    const uint32_t* tr = reinterpret_cast<const uint32_t*>(a.tp0[e]);
    const int per = (a.k >> 4) / 4, kt0 = warp * per;
    const int nt0 = blockIdx.y * 16, ntiles = a.n >> 4;
    float acc[16][2][2];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int j = 0; j < 2; ++j) acc[i][h][j] = 0.f;
    for (int it = 0; it < per; ++it) {
        const uint32_t* p = tr + ((size_t)(kt0 + it) * ntiles + nt0) * tile_words<8>();
        uint32_t words[16][lane_words<8>()];
#pragma unroll
        for (int i = 0; i < 16; ++i) ldg_lane_words<8>(p + i * tile_words<8>(), lane, words[i]);
        const int k = (kt0 + it) * 16;
        const uint32_t av[4] = {r < 0 ? 0u : *reinterpret_cast<const uint32_t*>(x + k), 0u,
                               r < 0 ? 0u : *reinterpret_cast<const uint32_t*>(x + k + 8), 0u};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint32_t b0[2], b1[2];
            decode_lane<8, 1>(words[i], lane, b0, b1);
            n256_small_mma(acc[i][0][0], acc[i][0][1], av, b0);
            n256_small_mma(acc[i][1][0], acc[i][1][1], av, b1);
        }
    }
    __shared__ __align__(16) float red[3][8][256];
    if (warp > 0) {
#pragma unroll
        for (int i = 0; i < 16; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int j = 0; j < 2; ++j) red[warp - 1][g][i * 16 + h * 8 + 2 * t + j] = acc[i][h][j];
    }
    __syncthreads();
    if (warp == 0 && r >= 0) {
#pragma unroll
        for (int i = 0; i < 16; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                float2 value = {acc[i][h][0], acc[i][h][1]};
#pragma unroll
                for (int w = 0; w < 3; ++w) {
                    value.x += red[w][g][col];
                    value.y += red[w][g][col + 1];
                }
                *reinterpret_cast<float2*>(a.z + (size_t)r * a.n + nt0 * 16 + col) = value;
            }
    }
}

}  // namespace tf_exl3

extern "C" __global__ void __launch_bounds__(128) tf_exl3_grouped_256_small_1(tf_exl3::GroupedArgs a) {
    tf_exl3::grouped_n256_small_body(a);
}
