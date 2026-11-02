template <int WARPS, int TR, bool VEC, int UNROLL>
__device__ __forceinline__ void candidate_body(
        int H, const float* __restrict__ state_in, const float* __restrict__ q_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        int rows, __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out) {
    constexpr int NT = WARPS * 32, VR = WARPS * 4;
    const int h = blockIdx.x, v0 = blockIdx.y * VR;
    const int lw = threadIdx.x >> 5, lane = threadIdx.x & 31, warp = blockIdx.y * WARPS + lw;
    __shared__ __align__(16) float kt[TR][DK];
    __shared__ __align__(16) float qt[TR][DK];
    __shared__ __align__(16) float gt[TR][DK];
    __shared__ __align__(16) float vt[TR][VR];
    __shared__ float bt[TR];
    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r0 = 0; r0 < rows; r0 += TR) {
        const int n = rows - r0 < TR ? rows - r0 : TR;
        __syncthreads();
        if constexpr (VEC) {
            for (int x = threadIdx.x; x < n * (DK / 4); x += NT) {
                const int rr = x / (DK / 4), c = x % (DK / 4);
                const size_t base = ((size_t)(r0 + rr) * H + h) * (DK / 4) + c;
                reinterpret_cast<float4*>(kt[rr])[c] = reinterpret_cast<const float4*>(k_save)[base];
                reinterpret_cast<float4*>(qt[rr])[c] = reinterpret_cast<const float4*>(q_in)[base];
                reinterpret_cast<float4*>(gt[rr])[c] = reinterpret_cast<const float4*>(g_save)[base];
            }
        } else {
        for (int x = threadIdx.x; x < n * DK; x += NT) {
            const int rr = x / DK, c = x % DK;
            const size_t base = ((size_t)(r0 + rr) * H + h) * DK + c;
            kt[rr][c] = k_save[base];
            qt[rr][c] = q_in[base];
            gt[rr][c] = g_save[base];
        }
        }
        for (int x = threadIdx.x; x < n * VR; x += NT) {
            const int rr = x / VR, c = x % VR;
            vt[rr][c] = __bfloat162float(v_save[((size_t)(r0 + rr) * H + h) * DV + v0 + c]);
        }
        if (threadIdx.x < n) bt[threadIdx.x] = b_save[(r0 + threadIdx.x) * H + h];
        __syncthreads();
        for (int rr0 = 0; rr0 < n; rr0 += UNROLL) {
#pragma unroll
          for (int u = 0; u < UNROLL; ++u) {
            const int rr = rr0 + u;
            if (rr < n) {
            const float4 k4 = reinterpret_cast<const float4*>(kt[rr])[lane];      // one 16-byte load a lane:
            const float4 q4 = reinterpret_cast<const float4*>(qt[rr])[lane];      // no bank conflicts
            const float4 g4 = reinterpret_cast<const float4*>(gt[rr])[lane];
            const float4 v4 = reinterpret_cast<const float4*>(vt[rr])[lw];
            float kk[4] = {k4.x, k4.y, k4.z, k4.w}, qq[4] = {q4.x, q4.y, q4.z, q4.w};
            float gg[4] = {g4.x, g4.y, g4.z, g4.w}, vv[4] = {v4.x, v4.y, v4.z, v4.w};
            update4(s, kk, gg, vv, bt[rr], lane);
            float part[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float o = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
                part[j] = o;
            }
            const float o = warp_sum4(part, lane);                                // read-out lane >> 3
            if ((lane & 7) == 0)
                y_out[((size_t)(r0 + rr) * H + h) * DV + warp * 4 + (lane >> 3)] = __float2bfloat16_rn(o);
        }
    }
            }
          }
    if (state_out != nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
    }
}

