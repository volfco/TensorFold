// TF_GLM_INDEX_PRUNE: an exact page prune of the DSA index scan (cuda_index_prune_dev.zig; STATUS-prune.md: why exact).
#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include "glm_index_scores.cuh"
#include <stdint.h>

namespace glm_prune {

constexpr int D = 128, PP = 128, POOL = 4, T = 256, W = T / 32, HMAX = 32, MAXP = 1024;  // MAXP: pages a row plans
constexpr int PB = 4 * W, CAP = 2048, PER = CAP / T;  // pages a plan block bounds; scored keys the cut ranks, a thread's

__device__ __forceinline__ bool is_nan(float x) { return x != x; }
__device__ __forceinline__ bool is_finite(float x) { return fabsf(x) <= 3.40282347e38f; }
__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }

// Element offset of logical page g's strip (its minima, then its maxima) in a cache of `page_elems`-element pages.
__device__ __forceinline__ int64_t strip_at(const int* pt, int64_t page_elems, int64_t g) {
  return (int64_t)pt[g] * page_elems + (int64_t)PP * D;
}

// sparse._order_key, as glm_index::order_key: unsigned order is the scores' (-0 as +0, any NaN above +inf).
__device__ __forceinline__ uint32_t order_key(float s) {
  const int bits = __float_as_int(__fadd_rn(s, 0.0f));
  return (uint32_t)(bits ^ ((bits >> 31) | (int)0x80000000u));
}

// A page's bound on any pool's score (a lane a head, q [HMAX][D + 1], lh its minima then maxima); NaN never prunes.
template <bool SOUND>
__device__ __forceinline__ float page_bound(float (*q)[D + 1], const float* w, float absw, const float* lh,
                                            int heads, float scale) {
  const int lane = threadIdx.x & 31;
  float up = 0.0f, down = 0.0f, abs = 0.0f, bad = 0.0f;
  for (int d = lane; d < 2 * D; d += 32) bad += is_finite(lh[d]) ? 0.0f : 1.0f;
  if (lane < heads)
    for (int d = 0; d < D; ++d) {
      const float x = q[lane][d], lo = lh[d], hi = lh[D + d];
      const float a = x * lo, b = x * hi;  // exact: bf16 times bf16
      up += fmaxf(a, b);
      down += fminf(a, b);
      abs += fabsf(x) * fmaxf(fabsf(lo), fabsf(hi));
    }
  const float wh = lane < heads ? w[lane] : 0.0f;
  // w >= 0 takes the dot's upper end, w < 0 its lower (relu monotone); SOUND false: the planted fault, every head up
  float term = (wh >= 0.0f || !SOUND) ? wh * fmaxf(scale * up, 0.0f) : wh * fmaxf(scale * down, 0.0f);
  float mag = fabsf(wh) * scale * abs;
  if (lane >= heads) term = mag = 0.0f;
  for (int o = 16; o > 0; o >>= 1) {
    term += __shfl_xor_sync(0xffffffffu, term, o);
    mag += __shfl_xor_sync(0xffffffffu, mag, o);
    bad += __shfl_xor_sync(0xffffffffu, bad, o);
  }
  if (bad != 0.0f || !(mag < 1e30f) || is_nan(term)) return __int_as_float(0x7fc00000);  // non-finite or huge
  // 2^-8 of the magnitude covers every fp32 rounding of the scores kernel and of this sum; the rest covers underflow
  return term + mag * (1.0f / 256.0f) + absw * 1e-33f + 1e-30f;
}

// A row's plan: blocks bound PB pages each; the last to count makes finite bounds candidates (1) but the seeds (0).
template <bool SOUND>
__device__ __forceinline__ void plan_row(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride,
                                         const __nv_bfloat16* pk, const int* pt, int64_t ps, const int* pos, float scale,
                                         float wscale, int64_t heads, uint8_t* plan, float* ub, int64_t pages,
                                         int seeds, int* count) {
  __shared__ float q[HMAX][D + 1];
  __shared__ float w[HMAX];
  __shared__ float lh[W][2 * D];
  __shared__ float us[MAXP];
  __shared__ uint8_t pl[MAXP];
  __shared__ float bv[T];
  __shared__ int bi[T];
  __shared__ float absw;
  __shared__ int last_block;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int64_t r = blockIdx.x;
  const int64_t npool = ((int64_t)pos[0] + r + 1) / POOL;
  const int64_t whole = min(npool / PP, pages);  // whole visible pages: their strips are current
  const int64_t g0 = (int64_t)blockIdx.y * PB, g1 = min(whole, g0 + PB);
  if (whole > seeds && g0 < g1) {
    for (int i = t; i < HMAX * D; i += T) {
      const int h = i / D, d = i % D;
      q[h][d] = h < heads ? bf(qi[(r * heads + h) * D + d]) : 0.0f;
    }
    if (t < HMAX) w[t] = t < heads ? __fmul_rn(bf(wts[r * w_stride + t]), wscale) : 0.0f;
    __syncthreads();
    if (t == 0) {
      float s = 0.0f;
      for (int h = 0; h < HMAX; ++h) s += fabsf(w[h]);
      absw = s;
    }
    __syncthreads();
    for (int64_t g = g0 + warp; g < g1; g += W) {
      const __nv_bfloat16* s = pk + strip_at(pt, ps / 2, g);
      for (int i = lane; i < 2 * D; i += 32) lh[warp][i] = bf(s[i]);
      __syncwarp();
      const float u = page_bound<SOUND>(q, w, absw, lh[warp], (int)heads, scale);
      if (lane == 0) ub[r * pages + g] = u;
      __syncwarp();
    }
  }
  __threadfence();  // this block's bounds reach the row's last block
  __syncthreads();
  if (t == 0) {
    last_block = atomicAdd(&count[r], 1) == (int)gridDim.y - 1;
    if (last_block) count[r] = 0;  // every block of the row has counted: ready for the next launch
  }
  __syncthreads();
  if (!last_block) return;
  __threadfence();
  for (int64_t g = t; g < pages; g += T) pl[g] = 0;
  __syncthreads();
  if (whole > seeds) {
    for (int64_t g = t; g < whole; g += T) {
      const float u = __ldcg(&ub[r * pages + g]);  // past L1: other blocks wrote it
      us[g] = u;
      pl[g] = is_nan(u) ? 0 : 1;
    }
    __syncthreads();
    for (int s = 0; s < seeds; ++s) {  // the next seed: the largest bound among the candidates, ties to the lower page
      float best = 0.0f;
      int at = -1;
      for (int64_t g = t; g < whole; g += T)
        if (pl[g] == 1 && (at < 0 || us[g] > best)) {
          best = us[g];
          at = (int)g;
        }
      bv[t] = best;
      bi[t] = at;
      __syncthreads();
      for (int o = T / 2; o > 0; o >>= 1) {
        if (t < o) {
          const int j = bi[t + o];
          if (j >= 0 && (bi[t] < 0 || bv[t + o] > bv[t] || (bv[t + o] == bv[t] && j < bi[t]))) {
            bv[t] = bv[t + o];
            bi[t] = j;
          }
        }
        __syncthreads();
      }
      if (t == 0 && bi[0] >= 0) pl[bi[0]] = 0;
      __syncthreads();
    }
  }
  for (int64_t g = t; g < pages; g += T) plan[r * pages + g] = pl[g];
}

}  // namespace glm_prune

// The strips of the pages a write of `rows` rows at `pos` completes (pos null: a restore, from page 0).
extern "C" __global__ void __launch_bounds__(glm_prune::D)
    tf_glm_prune_bounds(__nv_bfloat16* pk, const int* pt, int64_t ps, const int* pos, int64_t rows) {
  using namespace glm_prune;
  const int64_t P = pos ? (int64_t)pos[0] : 0, done = (P + rows) / POOL;
  const int64_t g = P / POOL / PP + blockIdx.x;
  if ((g + 1) * PP > done) return;  // not whole yet (its strip is written when its last pool is)
  const int d = threadIdx.x;
  const __nv_bfloat16* keys = pk + (int64_t)pt[g] * (ps / 2) + d;
  float lo = bf(keys[0]), hi = lo;
  for (int i = 1; i < PP; ++i) {
    const float v = bf(keys[(int64_t)i * D]);
    lo = is_nan(v) || v < lo ? v : lo;  // a NaN, once taken, stays: v < NaN is false
    hi = is_nan(v) || v > hi ? v : hi;
  }
  __nv_bfloat16* s = pk + strip_at(pt, ps / 2, g);
  s[d] = __float2bfloat16_rn(lo);
  s[D + d] = __float2bfloat16_rn(hi);
}

// The plan pass: grid (rows, cdiv(pages, PB)); plan and ub [rows][pages]; count [rows] is 0 between launches.
extern "C" __global__ void __launch_bounds__(glm_prune::T)
    tf_glm_prune_plan(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride, const __nv_bfloat16* pk,
                      const int* pt, int64_t ps, const int* pos, float scale, float wscale, int64_t heads, uint8_t* plan,
                      float* ub, int64_t pages, int seeds, int* count) {
  glm_prune::plan_row<true>(qi, wts, w_stride, pk, pt, ps, pos, scale, wscale, heads, plan, ub, pages, seeds, count);
}

namespace glm_prune {

// Row r's cut (a block of T threads): candidates below the k-th best scored key are written -inf (2), the rest planned.
__device__ __forceinline__ void cut_row(float* scores, int64_t np, const int* pos, int k, uint8_t* plan, const float* ub,
                                        int64_t pages, unsigned long long* stats, int64_t r) {
  constexpr int LISTP = CAP / PP;  // the scored pages whose keys the cut ranks (any subset of scored pools is sound)
  __shared__ uint8_t pl[MAXP];
  __shared__ int list[LISTP];
  __shared__ int wsum[W];
  __shared__ int hist[256];
  __shared__ int sh_n, sh_valid, sh_need, sh_digit;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int64_t npool = ((int64_t)pos[0] + r + 1) / POOL;
  const int64_t limit = min(np, max(npool, (int64_t)k));  // the pools the select reads
  const int64_t lp = min(pages, (limit + PP - 1) / PP), whole = min(npool / PP, pages);
  float* row = scores + r * np;
  for (int64_t g = t; g < pages; g += T) pl[g] = plan[r * pages + g];
  if (t == 0) sh_n = sh_valid = 0;
  __syncthreads();
  // the first LISTP scored pages under limit, in page order: T pages at a time, a block scan of their flags
  for (int64_t base = 0; base < lp && sh_n < LISTP; base += T) {
    const int64_t g = base + t;
    const int f = g < lp && pl[g] == 0;
    int inc = f;
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, inc, o);
      if (lane >= o) inc += v;
    }
    if (lane == 31) wsum[warp] = inc;
    __syncthreads();
    int before = inc - f, total = 0;
    for (int i = 0; i < W; ++i) {
      before += i < warp ? wsum[i] : 0;
      total += wsum[i];
    }
    const int at = sh_n + before;
    if (f && at < LISTP) list[at] = (int)g;
    __syncthreads();
    if (t == 0) sh_n += total;
    __syncthreads();
  }
  const int n = min(sh_n, LISTP) * PP;
  uint32_t u[PER];
  bool in[PER];
  int mine = 0;
  for (int i = 0; i < PER; ++i) {
    const int j = t + i * T;
    const int64_t p = j < n ? (int64_t)list[j / PP] * PP + j % PP : limit;
    in[i] = p < limit;
    u[i] = in[i] ? order_key(__ldcg(&row[p])) : 0u;  // past L1: other CTAs wrote it
    mine += in[i];
  }
  atomicAdd(&sh_valid, mine);
  __syncthreads();
  uint32_t cut = 0;  // key 0: nothing lies below it, so nothing is skipped
  if (sh_valid >= k) {
    uint32_t prefix = 0, fixed = 0;
    int need = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
      hist[t] = 0;  // T == 256 bins
      __syncthreads();
      for (int i = 0; i < PER; ++i)
        if (in[i] && (u[i] & fixed) == prefix) atomicAdd(&hist[(u[i] >> shift) & 0xFF], 1);
      __syncthreads();
      if (warp == 0) {  // lane l holds bins 255 - 8 l .. 248 - 8 l; the one lane where the count reaches need decides
        int h[8], own = 0;
        for (int j = 0; j < 8; ++j) own += h[j] = hist[255 - 8 * lane - j];
        int upto = own;
        for (int o = 1; o < 32; o <<= 1) {
          const int v = __shfl_up_sync(0xffffffffu, upto, o);
          if (lane >= o) upto += v;
        }
        int above = upto - own;
        if (above < need && upto >= need) {
          int digit = 0;
          for (int j = 0; j < 8; ++j) {
            if (above + h[j] >= need) {
              digit = 255 - 8 * lane - j;
              break;
            }
            above += h[j];
          }
          sh_need = need - above;
          sh_digit = digit;
        }
      }
      __syncthreads();
      need = sh_need;
      prefix |= (uint32_t)sh_digit << shift;
      fixed |= 0xFFu << shift;
      __syncthreads();
    }
    cut = prefix;  // the k-th best key among scored pools: at most the k-th best of the whole row
  }
  int skipped = 0;
  for (int64_t g = t; g < pages; g += T) {
    uint8_t v = 2;
    if (pl[g] == 1) {
      const float b = ub[r * pages + g];
      const bool skip = !is_nan(b) && order_key(b) < cut;
      v = skip ? 3 : 0;  // 3: skipped, its pools written -inf below
      skipped += skip;
    }
    pl[g] = v;
  }
  __syncthreads();
  for (int64_t i = t; i < whole * PP; i += T)
    if (pl[i / PP] == 3) row[i] = -INFINITY;
  for (int64_t g = t; g < pages; g += T) plan[r * pages + g] = pl[g] == 3 ? 2 : pl[g];
  if (stats) {
    atomicAdd(&stats[0], (unsigned long long)skipped);
    if (t == 0) {
      atomicAdd(&stats[1], (unsigned long long)whole);
      atomicAdd(&stats[2], 1ull);
    }
  }
  __syncthreads();  // the next row's cut reuses the shared arrays
}

}  // namespace glm_prune

// The planned scores (tf_glm_index_scores' bits); k > 0: each 8-row group's last CTA (group_count) cuts its rows after.
extern "C" __global__ void __launch_bounds__(glm_index::RB * 32)
    tf_glm_index_scores_prune(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride,
                              const __nv_bfloat16* pk, float* out, const int* pos, int64_t rows, int64_t np, int64_t nb,
                              float scale, float wscale, int64_t heads, int loop, const int* pt, int64_t ps,
                              uint8_t* plan, int64_t pages, int k, const float* ub, int* group_count,
                              unsigned long long* stats) {
  glm_index::scores_body<true>(qi, wts, w_stride, pk, out, pos, rows, np, nb, scale, wscale, heads, loop, pt, ps, plan,
                               pages);
  if (k <= 0) return;
  __shared__ int last_cta;
  __threadfence();  // this CTA's scores reach the group's last CTA
  __syncthreads();
  if (threadIdx.x == 0) {
    last_cta = atomicAdd(&group_count[blockIdx.x], 1) == (int)gridDim.y - 1;
    if (last_cta) group_count[blockIdx.x] = 0;
  }
  __syncthreads();
  if (!last_cta) return;
  __threadfence();
  const int64_t r1 = min(rows, (int64_t)(blockIdx.x + 1) * glm_index::RB);
  for (int64_t r = (int64_t)blockIdx.x * glm_index::RB; r < r1; ++r)
    glm_prune::cut_row(out, np, pos, k, plan, ub, pages, stats, r);
}
