#define main small_probe_unused_main
#include "small-probe.cu"
#undef main

int main() {
    std::mt19937 rng(8256);
    constexpr int experts = 8, slots = 8, bank_count = 2;
    for (int rows : {1, 4, 8, 16}) for (bool gu : {true, false}) {
        const int k = gu ? 4096 : 1024, n = gu ? 1024 : 4096;
        const int sk = gu ? 4 : 1, mats = gu ? 2 : 1, pairs = rows * slots;
        std::vector<half> x(pairs * k), nx(pairs * k);
        for (size_t i = 0; i < x.size(); ++i) {
            const float value = (int(rng() % 4096) - 2048) / 2048.f;
            x[i] = __float2half(value); nx[i] = __float2half(-value);
        }
        half *dx = upload(x), *dnx = upload(nx);
        std::vector<int> ids(experts), widths(experts, 8), members(experts * rows);
        for (int e = 0; e < experts; ++e) {
            ids[e] = e;
            for (int r = 0; r < rows; ++r) members[e * rows + r] = r * 32 + e;
        }
        int *di = upload(ids), *dc = upload(std::vector<int>{experts}), *dk = upload(widths), *dm = upload(members);
        std::vector<std::vector<uint32_t>> tables(experts * mats);
        for (auto& table : tables) { table.resize(k * n / 8); for (auto& value : table) value = rng(); }
        std::vector<uint32_t*> weights;
        std::vector<long long*> banks;
        for (int bank = 0; bank < bank_count; ++bank) for (int m = 0; m < mats; ++m) {
            std::vector<long long> pointers;
            for (int e = 0; e < experts; ++e) {
                uint32_t* w = upload(tables[m * experts + e]); weights.push_back(w);
                pointers.push_back(reinterpret_cast<long long>(w));
            }
            banks.push_back(upload(pointers));
        }
        const size_t len = size_t(mats) * sk * pairs * n;
        float *base, *fast; CHECK(cudaMalloc(&base, len * 4)); CHECK(cudaMalloc(&fast, len * 4));
        tf_exl3::GroupedArgs a = {dx, dnx, banks[0], banks[mats - 1], dk, dk, di, dc, dm, base,
                                k, n, pairs, sk, rows, slots, experts, mats, 0, 0, -1};
        const dim3 bg(experts, n / 128, mats * sk), fg(experts, n / 256, mats * sk);
        auto launch = [&](int kind) {
            if (kind == 0) tf_exl3_grouped_1<<<bg, 128>>>(a);
            else if (kind == 1) tf_exl3_grouped_256_1<<<fg, 128>>>(a);
            else tf_n256_small::small<<<fg, 128>>>(a);
        };
        launch(0); CHECK(cudaDeviceSynchronize());
        std::vector<float> hb(len), hf(len); CHECK(cudaMemcpy(hb.data(), base, len * 4, cudaMemcpyDeviceToHost));
        const int kinds = rows <= 8 ? 3 : 2;
        for (int kind = 1; kind < kinds; ++kind) {
            a.z = fast; launch(kind); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(hf.data(), fast, len * 4, cudaMemcpyDeviceToHost));
            if (std::memcmp(hb.data(), hf.data(), len * 4)) { std::fprintf(stderr, "BITS FAIL rows=%d gu=%d kind=%d\n", rows, gu, kind); return 1; }
        }
        cudaEvent_t start, end; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
        auto measure = [&](int kind) {
            CHECK(cudaEventRecord(start));
            for (int loop = 0; loop < 80; ++loop) for (int bank = 0; bank < bank_count; ++bank) {
                a.tp0 = banks[bank * mats]; a.tp1 = banks[bank * mats + mats - 1]; a.z = fast; launch(kind);
            }
            CHECK(cudaEventRecord(end)); CHECK(cudaEventSynchronize(end));
            float ms; CHECK(cudaEventElapsedTime(&ms, start, end)); return ms / 160;
        };
        for (int rep = 0; rep < 6; ++rep) {
            float time[3] = {};
            for (int at = 0; at < kinds; ++at) { const int kind = rep % 2 == 0 ? at : kinds - 1 - at; time[kind] = measure(kind); }
            std::printf("EXPERT8 rows=%d gu=%d experts=8 distinct_matrices=%d bank_bytes=%zu rep=%d warmup=%d n128_ms=%.6f n256_ms=%.6f small_ms=%.6f\n",
                        rows, gu, mats, size_t(bank_count) * mats * experts * k * n / 2, rep, rep == 0, time[0], time[1], time[2]);
            std::fflush(stdout);
        }
        CHECK(cudaMemcpy(hf.data(), fast, len * 4, cudaMemcpyDeviceToHost));
        if (std::memcmp(hb.data(), hf.data(), len * 4)) return 1;
        CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(end));
        for (auto p : weights) CHECK(cudaFree(p)); for (auto p : banks) CHECK(cudaFree(p));
        CHECK(cudaFree(dx)); CHECK(cudaFree(dnx)); CHECK(cudaFree(di)); CHECK(cudaFree(dc)); CHECK(cudaFree(dk)); CHECK(cudaFree(dm));
        CHECK(cudaFree(base)); CHECK(cudaFree(fast));
    }
    std::puts("PASS expert8: all baseline/candidate output bytes match");
}
