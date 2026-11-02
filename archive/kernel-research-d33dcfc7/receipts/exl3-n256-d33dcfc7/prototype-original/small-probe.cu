#include "zig/kernels/cuda/exl3_experts.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(call) do { cudaError_t rc = (call); if (rc != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s at line %d\n", cudaGetErrorString(rc), __LINE__); std::exit(2); } } while (0)

namespace tf_n256_small {
__device__ __forceinline__ void mma2(float& d0, float& d1, const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    float pad0, pad1;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%10,%10};\n"
                 : "+f"(d0), "+f"(d1), "=f"(pad0), "=f"(pad1)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.f));
}

__global__ void __launch_bounds__(128) small(tf_exl3::GroupedArgs a) {
    const int u = blockIdx.x;
    if (u >= a.ucount[0]) return;
    const int mat = blockIdx.z / a.sk, split = blockIdx.z % a.sk;
    const int e = a.uids[u];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int r = tf_exl3::n256_row(a, u, g);
    const half* x = (mat ? a.x1 : a.x0) + (size_t)(r < 0 ? 0 : r) * a.k + 2 * t;
    const uint32_t* tr = reinterpret_cast<const uint32_t*>(mat ? a.tp1[e] : a.tp0[e]);
    const int per = (a.k >> 4) / a.sk / 4, kt0 = split * per * 4 + warp * per;
    const int nt0 = blockIdx.y * 16, ntiles = a.ns > 0 ? a.ns : a.n / 16;
    float acc[16][2][2];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int j = 0; j < 2; ++j) acc[i][h][j] = 0;
    for (int it = 0; it < per; ++it) {
        const uint32_t* p = tr + ((size_t)(kt0 + it) * ntiles + nt0) * 32;
        uint32_t words[16][tf_exl3::lane_words<8>()];
#pragma unroll
        for (int i = 0; i < 16; ++i) tf_exl3::ldg_lane_words<8>(p + i * 32, lane, words[i]);
        const int k = (kt0 + it) * 16;
        const uint32_t av[4] = {r < 0 ? 0u : *reinterpret_cast<const uint32_t*>(x + k), 0u,
                               r < 0 ? 0u : *reinterpret_cast<const uint32_t*>(x + k + 8), 0u};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint32_t b0[2], b1[2];
            tf_exl3::decode_lane<8, 1>(words[i], lane, b0, b1);
            mma2(acc[i][0][0], acc[i][0][1], av, b0);
            mma2(acc[i][1][0], acc[i][1][1], av, b1);
        }
    }
    __shared__ float red[3][8][256];
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
                for (int w = 0; w < 3; ++w) { value.x += red[w][g][col]; value.y += red[w][g][col + 1]; }
                *reinterpret_cast<float2*>(a.z + (((size_t)mat * a.sk + split) * a.p + r) * a.n + nt0 * 16 + col) = value;
            }
    }
}
}

template<class T> T* upload(const std::vector<T>& data) {
    T* p; CHECK(cudaMalloc(&p, data.size() * sizeof(T)));
    CHECK(cudaMemcpy(p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}

int main() {
    std::mt19937 rng(256);
    for (int rows : {1, 4, 8}) for (bool gu : {true, false}) {
        const int k = gu ? 4096 : 1024, n = gu ? 1024 : 4096, sk = gu ? 4 : 1, mats = gu ? 2 : 1;
        std::vector<half> x(rows * k), nx(rows * k);
        for (size_t i = 0; i < x.size(); ++i) { const float v = (int(rng() % 4096) - 2048) / 2048.f; x[i] = __float2half(v); nx[i] = __float2half(-v); }
        half *dx = upload(x), *dnx = upload(nx);
        int *ids = upload(std::vector<int>{0}), *count = upload(std::vector<int>{1}), *k2 = upload(std::vector<int>{8});
        std::vector<int> members(rows); for (int r = 0; r < rows; ++r) members[r] = r * 32;
        int* dm = upload(members);
        std::vector<uint32_t> words(k * n / 8); for (auto& v : words) v = rng();
        std::vector<uint32_t*> weights; std::vector<long long*> banks;
        for (int i = 0; i < 16; ++i) { uint32_t* w = upload(words); weights.push_back(w); banks.push_back(upload(std::vector<long long>{reinterpret_cast<long long>(w)})); }
        const size_t len = size_t(mats) * sk * rows * n;
        float *base, *fast; CHECK(cudaMalloc(&base, len * 4)); CHECK(cudaMalloc(&fast, len * 4));
        tf_exl3::GroupedArgs a = {dx, dnx, banks[0], banks[0], k2, k2, ids, count, dm, base, k, n, rows, sk, rows, 1, 1, mats, 0, 0, -1};
        const dim3 bg(1, n / 128, mats * sk), fg(1, n / 256, mats * sk);
        tf_exl3_grouped_1<<<bg, 128>>>(a); a.z = fast; tf_n256_small::small<<<fg, 128>>>(a); CHECK(cudaDeviceSynchronize());
        std::vector<float> hb(len), hf(len); CHECK(cudaMemcpy(hb.data(), base, len * 4, cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(hf.data(), fast, len * 4, cudaMemcpyDeviceToHost));
        if (std::memcmp(hb.data(), hf.data(), len * 4)) { std::fprintf(stderr, "BITS FAIL rows=%d gu=%d\n", rows, gu); return 1; }
        cudaEvent_t start, end; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
        auto measure = [&](bool candidate) {
            CHECK(cudaEventRecord(start));
            for (int loop = 0; loop < 10; ++loop) for (auto bank : banks) {
                a.tp0 = a.tp1 = bank; a.z = fast;
                if (candidate) tf_n256_small::small<<<fg, 128>>>(a); else tf_exl3_grouped_1<<<bg, 128>>>(a);
            }
            CHECK(cudaEventRecord(end)); CHECK(cudaEventSynchronize(end)); float ms; CHECK(cudaEventElapsedTime(&ms, start, end)); return ms / 160;
        };
        for (int rep = 0; rep < 6; ++rep) {
            float baseline, candidate;
            if (rep % 2 == 0) { baseline = measure(false); candidate = measure(true); } else { candidate = measure(true); baseline = measure(false); }
            std::printf("SMALL rows=%d gu=%d rep=%d warmup=%d n128_ms=%.6f small_ms=%.6f\n", rows, gu, rep, rep == 0, baseline, candidate); std::fflush(stdout);
        }
        CHECK(cudaMemcpy(hf.data(), fast, len * 4, cudaMemcpyDeviceToHost)); if (std::memcmp(hb.data(), hf.data(), len * 4)) return 1;
        CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(end));
        for (auto p : weights) CHECK(cudaFree(p)); for (auto p : banks) CHECK(cudaFree(p));
        CHECK(cudaFree(dx)); CHECK(cudaFree(dnx)); CHECK(cudaFree(ids)); CHECK(cudaFree(count)); CHECK(cudaFree(k2)); CHECK(cudaFree(dm)); CHECK(cudaFree(base)); CHECK(cudaFree(fast));
    }
}
