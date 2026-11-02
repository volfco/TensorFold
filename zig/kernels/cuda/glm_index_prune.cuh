// TF_GLM_INDEX_PRUNE: an exact page prune of the DSA index scan (cuda_index_prune_dev.zig launches it).
// Each 128-pool page of pooled keys carries a strip after its keys: the per-dimension minimum, then maximum, of its
// pools' keys (bf16, exact: a min or max of bf16 values). A row's bound on any pool score in a whole page is sound for
// tf_glm_index_scores' arithmetic (bf16 products, fp32 sums); a page whose bound's order key lies strictly below the
// row's k-th best order key among scores already computed holds no selected pool, whatever the tie rule.
#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace glm_prune {

constexpr int D = 128, PP = 128, POOL = 4, T = 256, W = T / 32, HMAX = 32, MAXP = 1024;  // MAXP: pages a row plans

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

// Lane h's (one head a lane) part of a page's bound, summed over the warp. q: [HMAX][D + 1] floats (bf16 values),
// w: head weights as the scores kernel forms them, lh: the page's minima then maxima. A positive weight takes the dot's
// upper end, a negative one its lower end (relu monotone, scale positive); SOUND false bounds every head from above
// (the planted fault the emulation must catch). The margin, 2^-8 of the terms' magnitude, covers every fp32 rounding
// of the scores kernel (tensor-core sums of 128 exact products, the scale, the 32-head sum) and of this sum; the
// absolute terms cover underflow. A non-finite strip, query, weight or a magnitude of 1e30 or more gives NaN, which
// never prunes.
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
  float term = (wh >= 0.0f || !SOUND) ? wh * fmaxf(scale * up, 0.0f) : wh * fmaxf(scale * down, 0.0f);
  float mag = fabsf(wh) * scale * abs;
  if (lane >= heads) term = mag = 0.0f;
  for (int o = 16; o > 0; o >>= 1) {
    term += __shfl_xor_sync(0xffffffffu, term, o);
    mag += __shfl_xor_sync(0xffffffffu, mag, o);
    bad += __shfl_xor_sync(0xffffffffu, bad, o);
  }
  if (bad != 0.0f || !(mag < 1e30f) || is_nan(term)) return __int_as_float(0x7fc00000);
  return term + mag * (1.0f / 256.0f) + absw * 1e-33f + 1e-30f;
}

// The plan of one row (a block of T threads a row): every whole visible page's bound into ub[r][g]; pages with a
// finite bound are candidates (1), the `seeds` of largest bound (ties to the lower page) and every other page are
// scored first (0). Rows with `seeds` whole pages or fewer plan nothing (all 0).
template <bool SOUND>
__device__ __forceinline__ void plan_row(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride,
                                         const __nv_bfloat16* pk, const int* pt, int64_t ps, const int* pos, float scale,
                                         float wscale, int64_t heads, uint8_t* plan, float* ub, int64_t pages,
                                         int seeds) {
  __shared__ float q[HMAX][D + 1];
  __shared__ float w[HMAX];
  __shared__ float lh[W][2 * D];
  __shared__ float us[MAXP];
  __shared__ uint8_t pl[MAXP];
  __shared__ float bv[T];
  __shared__ int bi[T];
  __shared__ float absw;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int64_t r = blockIdx.x;
  const int64_t npool = ((int64_t)pos[0] + r + 1) / POOL;
  const int64_t whole = min(npool / PP, pages);  // whole visible pages: their strips are current
  for (int64_t g = t; g < pages; g += T) pl[g] = 0;
  __syncthreads();
  if (whole > seeds) {
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
    for (int64_t g = warp; g < whole; g += W) {
      const __nv_bfloat16* s = pk + strip_at(pt, ps / 2, g);
      for (int i = lane; i < 2 * D; i += 32) lh[warp][i] = bf(s[i]);
      __syncwarp();
      const float u = page_bound<SOUND>(q, w, absw, lh[warp], (int)heads, scale);
      if (lane == 0) {
        us[g] = u;
        pl[g] = is_nan(u) ? 0 : 1;
      }
      __syncwarp();
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
    for (int64_t g = t; g < whole; g += T) ub[r * pages + g] = us[g];
  }
  for (int64_t g = t; g < pages; g += T) plan[r * pages + g] = pl[g];
}

}  // namespace glm_prune

// A prompt chunk's or decode step's pooled keys just written (tf_glm_pool_keys at `pos`, `rows` rows): the strip of
// every page they complete, one block (a thread a dimension) a page; grid cdiv(rows, 512) + 2 blocks. With pos null,
// pages 0 on of a restored state's `rows` rows. NaN in a column gives NaN in both its strip entries.
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

// The plan pass: grid `rows` blocks of T threads; plan and ub [rows][pages] (pages at most MAXP).
extern "C" __global__ void __launch_bounds__(glm_prune::T)
    tf_glm_prune_plan(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride, const __nv_bfloat16* pk,
                      const int* pt, int64_t ps, const int* pos, float scale, float wscale, int64_t heads, uint8_t* plan,
                      float* ub, int64_t pages, int seeds) {
  glm_prune::plan_row<true>(qi, wts, w_stride, pk, pt, ps, pos, scale, wscale, heads, plan, ub, pages, seeds);
}

// The cut, after the scores of the planned pages (0) are in: the row's k-th best order key among them (none when
// fewer than k of its visible pools were scored), then each candidate (1, written -inf) is skipped (2) when its bound's
// key lies strictly below it, else rescored (0); the pages already scored are left (2). stats (or null): skipped
// pages, whole visible pages and rows, added. Grid `rows` blocks of T threads.
extern "C" __global__ void __launch_bounds__(glm_prune::T)
    tf_glm_prune_cut(const float* scores, int64_t np, const int* pos, int k, uint8_t* plan, const float* ub,
                     int64_t pages, int* stats) {
  using namespace glm_prune;
  __shared__ uint8_t pl[MAXP];
  __shared__ int list[MAXP];
  __shared__ int cnt[T];
  __shared__ int hist[256];
  __shared__ int sh_n, sh_valid, sh_need;
  __shared__ uint32_t sh_prefix;
  const int t = threadIdx.x;
  const int64_t r = blockIdx.x;
  const int64_t npool = ((int64_t)pos[0] + r + 1) / POOL;
  const int64_t limit = min(np, max(npool, (int64_t)k));  // the pools the select reads
  const int64_t lp = min(pages, (limit + PP - 1) / PP);
  const float* row = scores + r * np;
  for (int64_t g = t; g < pages; g += T) pl[g] = plan[r * pages + g];
  __syncthreads();
  // the scored pages under limit, in page order: a run of pages a thread, counted, scanned, written
  const int64_t per = (lp + T - 1) / T, g0 = min(lp, t * per), g1 = min(lp, g0 + per);
  int c = 0;
  for (int64_t g = g0; g < g1; ++g) c += pl[g] == 0;
  cnt[t] = c;
  __syncthreads();
  if (t == 0) {
    int run = 0;
    for (int i = 0; i < T; ++i) {
      const int v = cnt[i];
      cnt[i] = run;
      run += v;
    }
    sh_n = run;
    sh_valid = run * PP;
  }
  __syncthreads();
  int at = cnt[t];
  for (int64_t g = g0; g < g1; ++g)
    if (pl[g] == 0) list[at++] = (int)g;
  __syncthreads();
  if (t == 0 && sh_n > 0 && list[sh_n - 1] == lp - 1) sh_valid -= (int)(lp * PP - limit);
  __syncthreads();
  const int n = sh_n * PP;
  uint32_t cut = 0;  // key 0: nothing lies below it, so nothing is skipped
  if (sh_valid >= k) {
    uint32_t prefix = 0, fixed = 0;
    int need = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int b = t; b < 256; b += T) hist[b] = 0;
      __syncthreads();
      for (int j = t; j < n; j += T) {
        const int64_t p = (int64_t)list[j / PP] * PP + j % PP;
        if (p >= limit) continue;
        const uint32_t u = order_key(row[p]);
        if ((u & fixed) == prefix) atomicAdd(&hist[(u >> shift) & 0xFF], 1);
      }
      __syncthreads();
      if (t == 0) {
        int above = 0, digit = 0;
        for (int b = 255; b >= 0; --b) {
          if (above + hist[b] >= need) {
            digit = b;
            break;
          }
          above += hist[b];
        }
        sh_need = need - above;
        sh_prefix = prefix | ((uint32_t)digit << shift);
      }
      __syncthreads();
      need = sh_need;
      prefix = sh_prefix;
      fixed |= 0xFFu << shift;
      __syncthreads();
    }
    cut = prefix;  // the k-th best key among the scored pools: at most the k-th best of the whole row
  }
  int skipped = 0;
  for (int64_t g = t; g < pages; g += T) {
    uint8_t v = 2;
    if (pl[g] == 1) {
      const float u = ub[r * pages + g];
      const bool skip = !is_nan(u) && order_key(u) < cut;
      v = skip ? 2 : 0;
      skipped += skip;
    }
    plan[r * pages + g] = v;
  }
  if (stats) {
    atomicAdd(&stats[0], skipped);
    if (t == 0) {
      atomicAdd(&stats[1], (int)min(npool / PP, pages));
      atomicAdd(&stats[2], 1);
    }
  }
}
