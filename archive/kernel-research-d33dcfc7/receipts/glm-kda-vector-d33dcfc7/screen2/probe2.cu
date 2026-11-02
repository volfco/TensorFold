#include "glm_kda.cu"
#include "candidate.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)

template<int W, int TR>
__global__ __launch_bounds__(W * 32) void probe(int H, const float* st, const float* q, const float* k,
        const __nv_bfloat16* v, const float* g, const float* b, int rows, __nv_bfloat16* y, float* so) {
    step_kernel_body<W, TR>(H, st, q, k, v, g, b, rows, y, so);
}

template<int W, int TR, bool VEC, int UNROLL>
__global__ __launch_bounds__(W * 32) void candidate(int H, const float* st, const float* q, const float* k,
        const __nv_bfloat16* v, const float* g, const float* b, int rows, __nv_bfloat16* y, float* so) {
    candidate_body<W, TR, VEC, UNROLL>(H, st, q, k, v, g, b, rows, y, so);
}

struct Buffer {
    unsigned char* all = nullptr;
    size_t bytes;
    explicit Buffer(size_t n): bytes(n) {
        CK(cudaMalloc(&all, bytes + 512));
        CK(cudaMemset(all, 0xa5, bytes + 512));
    }
    ~Buffer() { if (all) cudaFree(all); }
    void* ptr() const { return all + 256; }
    void put(const void* p) { CK(cudaMemcpy(ptr(), p, bytes, cudaMemcpyHostToDevice)); }
    std::vector<unsigned char> get() const {
        std::vector<unsigned char> p(bytes);
        CK(cudaMemcpy(p.data(), ptr(), bytes, cudaMemcpyDeviceToHost));
        return p;
    }
    void guards() const {
        unsigned char x[512];
        CK(cudaMemcpy(x, all, 256, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(x + 256, all + 256 + bytes, 256, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 512; ++i) if (x[i] != 0xa5) { std::fprintf(stderr, "GUARD FAIL %d\n", i); std::exit(3); }
    }
};

static unsigned rng_state;
static float randf(float lo, float hi) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return lo + (float)(rng_state & 0xffffff) * (1.0f / 16777216.0f) * (hi - lo);
}

struct Fixture {
    int H, rows;
    size_t nr, ns;
    Buffer st, q, k, v, g, b, y, so, refy, refso;
    std::vector<std::vector<unsigned char>> input_copy;
    Fixture(int h, int r, unsigned seed, bool nonfinite = false):
        H(h), rows(r), nr((size_t)h * r * 128), ns((size_t)h * 128 * 128),
        st(ns * 4), q(nr * 4), k(nr * 4), v(nr * 2), g(nr * 4), b((size_t)h * r * 4),
        y(nr * 2), so(ns * 4), refy(nr * 2), refso(ns * 4) {
        rng_state = seed;
        std::vector<float> x(std::max(nr, ns));
        for (size_t i = 0; i < ns; ++i) x[i] = randf(-0.01f, 0.01f);
        st.put(x.data());
        for (size_t i = 0; i < nr; ++i) x[i] = randf(-0.15f, 0.15f);
        if (nonfinite) { x[0] = NAN; x[128] = INFINITY; x[256] = -INFINITY; }
        q.put(x.data());
        for (size_t i = 0; i < nr; ++i) x[i] = randf(-0.15f, 0.15f);
        k.put(x.data());
        for (size_t i = 0; i < nr; ++i) x[i] = randf(0.95f, 1.0f);
        g.put(x.data());
        for (size_t i = 0; i < (size_t)H * rows; ++i) x[i] = randf(0.1f, 0.9f);
        b.put(x.data());
        std::vector<__nv_bfloat16> bv(nr);
        for (auto& t : bv) t = __float2bfloat16_rn(randf(-1.0f, 1.0f));
        v.put(bv.data());
        for (Buffer* p : inputs()) input_copy.push_back(p->get());
    }
    std::vector<Buffer*> inputs() { return {&st, &q, &k, &v, &g, &b}; }
    void intact() {
        int at = 0;
        for (Buffer* p : inputs()) {
            p->guards();
            if (p->get() != input_copy[at++]) { std::fprintf(stderr, "INPUT MUTATION\n"); std::exit(3); }
        }
        for (Buffer* p : {&y, &so, &refy, &refso}) p->guards();
    }
};

template<int W, int TR>
void launch(Fixture& f, bool baseline) {
    probe<W, TR><<<dim3(f.H, 128/(4*W)), W*32>>>(f.H,
        (float*)f.st.ptr(), (float*)f.q.ptr(), (float*)f.k.ptr(), (__nv_bfloat16*)f.v.ptr(),
        (float*)f.g.ptr(), (float*)f.b.ptr(), f.rows,
        (__nv_bfloat16*)(baseline ? f.refy.ptr() : f.y.ptr()), (float*)(baseline ? f.refso.ptr() : f.so.ptr()));
    CK(cudaGetLastError());
}

template<int W, int TR, bool VEC, int UNROLL>
void launch_candidate(Fixture& f, bool baseline) {
    candidate<W, TR, VEC, UNROLL><<<dim3(f.H, 128/(4*W)), W*32>>>(f.H,
        (float*)f.st.ptr(), (float*)f.q.ptr(), (float*)f.k.ptr(), (__nv_bfloat16*)f.v.ptr(),
        (float*)f.g.ptr(), (float*)f.b.ptr(), f.rows,
        (__nv_bfloat16*)(baseline ? f.refy.ptr() : f.y.ptr()), (float*)(baseline ? f.refso.ptr() : f.so.ptr()));
    CK(cudaGetLastError());
}

struct Variant { const char* name; int w, tr; void (*fn)(Fixture&, bool); const void* kernel; };
#define V(w, tr) {"w" #w "t" #tr, w, tr, launch<w, tr>, (const void*)probe<w, tr>}
#define C(w, tr, v, u) {"w" #w "t" #tr "v" #v "u" #u, w, tr, launch_candidate<w, tr, v, u>, (const void*)candidate<w, tr, v, u>}
static Variant variants[] = {
    V(8, 16), C(8, 16, false, 1), C(8, 16, true, 1), C(8, 16, true, 2),
    C(8, 16, true, 4), C(8, 16, false, 2), C(8, 16, false, 4), C(8, 16, false, 8),
    C(4, 8, true, 2), C(8, 8, true, 2)
};

static double timed(Fixture& f, const Variant& v, int reps) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) v.fn(f, false);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms * 1000.0 / reps;
}

static void qualify(int h, int rows, unsigned seed, bool nonfinite) {
    Fixture f(h, rows, seed, nonfinite);
    launch<8, 16>(f, true);
    CK(cudaDeviceSynchronize());
    auto y = f.refy.get(), so = f.refso.get();
    for (auto& v : variants) {
        v.fn(f, false);
        CK(cudaDeviceSynchronize());
        auto goty = f.y.get(), gots = f.so.get();
        size_t yd = 0, sd = 0;
        for (size_t i = 0; i < y.size(); ++i) yd += goty[i] != y[i];
        for (size_t i = 0; i < so.size(); ++i) sd += gots[i] != so[i];
        std::printf("QUALIFY h=%d rows=%d seed=%u nf=%d variant=%s y_diff_bytes=%zu state_diff_bytes=%zu\n",
            h, rows, seed, nonfinite, v.name, yd, sd);
        if (yd || sd) std::exit(4);
    }
    f.intact();
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    std::printf("DEVICE %s sms=%d compute=%d%d\n", p.name, p.multiProcessorCount, p.major, p.minor);
    for (auto& v : variants) {
        cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, v.kernel));
        std::printf("RESOURCE variant=%s regs=%d shared=%zu local=%zu threads=%d\n", v.name, a.numRegs,
            a.sharedSizeBytes, a.localSizeBytes, a.maxThreadsPerBlock);
    }
    const int hs[] = {21, 22, 32, 64};
    const int rs[] = {1, 3, 15, 16, 17, 63, 64, 65, 256, 1024, 2048};
    for (int h : hs) for (int rows : rs) qualify(h, rows, 0x12345678u + h * rows, false);
    for (int h : hs) qualify(h, 65, 0x76543210u + h, true);
    std::puts("QUALIFY_SUMMARY shapes=48 variants=10 comparisons=480 y_state_mismatches=0 guards=pass inputs=unchanged");
    if (argc > 1 && std::strcmp(argv[1], "--qualify-only") == 0) return 0;
    const Variant& baseline = variants[0];
    for (int h : hs) for (int rows : {64, 256, 1024, 2048}) {
        Fixture f(h, rows, 0x12345678u + h * rows);
        for (auto& v : variants) for (int w = 0; w < 5; ++w) v.fn(f, false);
        CK(cudaDeviceSynchronize());
        for (int rep = 0; rep < 5; ++rep) for (auto& v : variants) {
            double base_us, us;
            if (rep & 1) { us = timed(f, v, 20); base_us = timed(f, baseline, 20); }
            else { base_us = timed(f, baseline, 20); us = timed(f, v, 20); }
            std::printf("TIME h=%d rows=%d rep=%d variant=%s baseline_us=%.6f candidate_us=%.6f ratio=%.6f\n",
                h, rows, rep, v.name, base_us, us, base_us/us);
        }
        f.intact();
    }
    std::puts("COMPLETE");
}
