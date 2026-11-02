// TF_GLM_INDEX_PRUNE's kernels on the host (zig build test-glm-index-prune-emulate): glm_index_prune.cuh's strips,
// plan and cut and tf_glm_index_scores_prune, run as the engine runs them on synthetic pooled keys in shuffled pages;
// each row's selection must equal the full scan's (tf_glm_index_scores, then a stable top-K by order key), and two
// planted faults (negative head weights bounded from above, strips never written) must be caught.
#define GLM_INDEX_HOST
#include <cstdio>
#include <functional>
#include <random>
#include <thread>
#include <vector>
#include "cuda_bf16.h"
#include "cuda_runtime.h"

namespace glm_index {  // the same stand-ins as emulate.cpp: cp.async, ldmatrix and the m16n8k16 mma
inline void cp16(void* d, const void* s, bool ok) { ok ? (void)std::memcpy(d, s, 16) : (void)std::memset(d, 0, 16); }
inline void cp_commit() {}
template <int N> inline void cp_wait() {}
inline void ldm4(uint32_t* r, const void* p) {
  const int t = (int)threadIdx.x, l = t & 31, base = t - l;
  emu::pbuf[t] = p;
  warp_sync();
  for (int i = 0; i < 4; ++i) std::memcpy(&r[i], (const char*)emu::pbuf[base + 8 * i + l / 4] + 4 * (l % 4), 4);
  warp_sync();
}
inline float bf_at(uint32_t w, int half) { return __bfloat162float({(uint16_t)(half ? w >> 16 : w & 0xFFFF)}); }
inline void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  static uint32_t xa[1024][6];
  const int t = (int)threadIdx.x, l = t & 31, base = t - l, g = l / 4, q = l % 4;
  for (int i = 0; i < 4; ++i) xa[t][i] = a[i];
  xa[t][4] = b[0];
  xa[t][5] = b[1];
  warp_sync();
  float out[4];
  for (int j = 0; j < 4; ++j) {
    const int row = g + 8 * (j / 2), col = 2 * q + j % 2;
    double sum = 0;
    for (int k = 0; k < 16; ++k) {
      const uint32_t aw = xa[base + (row % 8) * 4 + (k % 8) / 2][(row < 8 ? 0 : 1) + (k < 8 ? 0 : 2)];
      const uint32_t bw = xa[base + col * 4 + (k % 8) / 2][4 + (k < 8 ? 0 : 1)];
      sum += (double)bf_at(aw, k % 2) * (double)bf_at(bw, k % 2);
    }
    out[j] = (float)((double)c[j] + sum);
  }
  warp_sync();
  for (int j = 0; j < 4; ++j) c[j] = out[j];
}
}  // namespace glm_index

#include "../../zig/kernels/cuda/glm_index_pools.cuh"
#include "../../zig/kernels/cuda/glm_index_scores.cuh"
#include "../../zig/kernels/cuda/glm_index_prune.cuh"

// The planted fault: tf_glm_prune_plan with every head bounded from above.
static void plan_unsound(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride, const __nv_bfloat16* pk,
                         const int* pt, int64_t ps, const int* pos, float scale, float wscale, int64_t heads,
                         uint8_t* plan, float* ub, int64_t pages, int seeds, int* count) {
  glm_prune::plan_row<false>(qi, wts, w_stride, pk, pt, ps, pos, scale, wscale, heads, plan, ub, pages, seeds, count);
}

static void run_grid(unsigned gx, unsigned gy, int threads, const std::function<void()>& kernel) {
  for (unsigned by = 0; by < gy; by++)
    for (unsigned bx = 0; bx < gx; bx++) {
      std::barrier<> block(threads);
      std::vector<std::barrier<>*> warps;
      for (int w = 0; w < threads / 32; w++) warps.push_back(new std::barrier<>(32));
      emu::block_bar = &block;
      for (int w = 0; w < threads / 32; w++) emu::warp_bar[w] = warps[w];
      std::vector<std::thread> ts;
      for (int t = 0; t < threads; t++)
        ts.emplace_back([=, &kernel] {
          blockIdx = {bx, by, 0};
          threadIdx = {(unsigned)t, 0, 0};
          kernel();
        });
      for (auto& t : ts) t.join();
      for (auto* w : warps) delete w;
    }
}

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
static __nv_bfloat16 tobf(float f) { return __float2bfloat16_rn(f); }

constexpr int D = 128, PP = 128, W = 40, EXTRA = 512;  // EXTRA: other rows' elements a page holds past the strip
constexpr int64_t PE = PP * D + 2 * D + EXTRA;            // a page's elements

enum Keys { normal, clustered, ties, equal, nan_page, nonfinite };
struct Case { int64_t pos, rows, np; int heads, k, loop, seeds; Keys keys; bool negative; const char* name; };

// The pooled keys: `pools` rows of D, as a page table of shuffled pages holds them (strips poisoned until written).
struct Pages {
  std::vector<__nv_bfloat16> keys, store;
  std::vector<int> pt;
};

static Pages make_pages(std::mt19937_64& g, int64_t pools, Keys kind) {
  Pages p;
  const int64_t n = (pools + PP - 1) / PP + 1;
  p.keys.resize(n * PP * D);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (int64_t i = 0; i < n * PP; i++) {
    const int64_t page = i / PP;
    const bool cl = kind == clustered || kind == nonfinite;
    const float shift = cl ? (float)(int)(page % 7) - 3.0f : 0.0f;
    for (int d = 0; d < D; d++) {
      float v = nd(g) * (cl ? 0.4f : 1.0f) + shift;
      if (kind == ties && page % 3 == 0) v = 0.5f;  // whole pages of one key: equal scores across pages
      if (kind == equal) v = 0.75f;                  // every key equal: every score ties
      p.keys[i * D + d] = tobf(v);
    }
  }
  if (kind == nan_page) p.keys[(3 * PP + 5) * D + 7] = tobf(NAN);
  if (kind == nonfinite) {  // +inf and -inf keys in two pages, a NaN in a third
    p.keys[(5 * PP + 1) * D + 4] = tobf(INFINITY);
    p.keys[(9 * PP + 7) * D + 60] = tobf(-INFINITY);
    p.keys[(12 * PP + 3) * D + 100] = tobf(NAN);
  }
  std::vector<int> order(n);
  for (int i = 0; i < n; i++) order[i] = i;
  std::shuffle(order.begin(), order.end(), g);
  p.pt = order;
  p.store.assign(n * PE, tobf(NAN));
  for (int64_t i = 0; i < n * PP; i++) std::memcpy(&p.store[order[i / PP] * PE + (i % PP) * D], &p.keys[i * D], D * 2);
  return p;
}

// The pooled keys' writes as the engine makes them: chunks of `rows` rows at increasing positions, each followed by
// tf_glm_prune_bounds; then every whole page's strip against the host's min and max (NaN when its column holds one).
static int write_strips(Pages& p, int64_t tokens, std::mt19937_64& g) {
  int64_t at = 0;
  while (at < tokens) {
    const int64_t rows = std::min<int64_t>(tokens - at, 1 + (int64_t)(g() % 3000));
    const int pos = (int)at;
    run_grid((unsigned)(rows / 512 + 2), 1, D, [&] { tf_glm_prune_bounds(p.store.data(), p.pt.data(), PE * 2, &pos, rows); });
    at += rows;
  }
  int bad = 0;
  for (int64_t page = 0; page < tokens / 4 / PP; page++)
    for (int d = 0; d < D; d++) {
      float lo = bf(p.keys[page * PP * D + d]), hi = lo;
      bool nan = false;
      for (int i = 0; i < PP; i++) {
        const float v = bf(p.keys[(page * PP + i) * D + d]);
        nan = nan || v != v;
        lo = std::min(lo, v);
        hi = std::max(hi, v);
      }
      const float glo = bf(p.store[p.pt[page] * PE + PP * D + d]), ghi = bf(p.store[p.pt[page] * PE + PP * D + D + d]);
      bad += nan ? !(glo != glo && ghi != ghi) : (bits(glo) != bits(lo) || bits(ghi) != bits(hi));
    }
  return bad;
}

static std::vector<int64_t> top_k(const float* row, int64_t limit, int k) {
  std::vector<int64_t> o(limit);
  for (int64_t i = 0; i < limit; i++) o[i] = i;
  std::stable_sort(o.begin(), o.end(), [&](int64_t a, int64_t b) { return glm_prune::order_key(row[a]) > glm_prune::order_key(row[b]); });
  std::vector<int64_t> best(o.begin(), o.begin() + k);
  std::sort(best.begin(), best.end());
  return best;
}

struct Result { int64_t differ = 0, rows = 0, skipped = 0, whole = 0, stray = 0; };

// One case: full scan and pruned scan of every row, selections compared. `fault`: 0 none, 1 unsound plan, 2 strips
// left unwritten (zero).
static Result run_case(std::mt19937_64& g, const Case& c, int fault) {
  const int64_t tokens = c.pos + c.rows;
  Pages pages = make_pages(g, tokens / 4 + 1, c.keys);
  Result res;
  const int strip_bad = write_strips(pages, tokens, g);
  if (strip_bad) {
    printf("%s: %d strip values differ from the host's min and max\n", c.name, strip_bad);
    res.differ += strip_bad;
  }
  if (fault == 2)
    for (int64_t i = 0; i < (int64_t)pages.pt.size(); i++)
      for (int d = 0; d < 2 * D; d++) pages.store[i * PE + PP * D + d] = tobf(0.0f);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_real_distribution<float> ud(0.0f, 1.0f);
  std::vector<__nv_bfloat16> q(c.rows * c.heads * D), w(c.rows * W);
  for (auto& x : q) x = tobf(nd(g) * 0.3f + (c.keys == clustered || c.keys == nonfinite ? 0.15f : 0.0f));
  if (c.keys == nonfinite) {  // the contract's gate-1 plants: a NaN query value, and an infinite key (make_pages)
    q[3] = tobf(NAN);
    q[(1 * c.heads + 2) * D + 9] = tobf(INFINITY);
  }
  for (int64_t r = 0; r < c.rows; r++)
    for (int h = 0; h < W; h++) w[r * W + h] = tobf(c.negative && h % 2 ? -ud(g) : ud(g));
  const float scale = 0.0883883461f, wscale = 0.176776692f;
  const int pos = (int)c.pos;
  const int64_t nb = (std::min(c.np, std::max<int64_t>(512, tokens / 4)) + 63) / 64, pages_n = (c.np + PP - 1) / PP;
  std::vector<float> full(c.rows * c.np, 1e9f), pruned(c.rows * c.np, 7e8f), ub(c.rows * pages_n, 0.0f);
  std::vector<uint8_t> plan(c.rows * pages_n, 9);
  unsigned long long stats[3] = {0, 0, 0};
  const __nv_bfloat16* pk = pages.store.data();
  run_grid((c.rows + 7) / 8, (nb + c.loop - 1) / c.loop, 256, [&] {
    tf_glm_index_scores(q.data(), w.data(), W, pk, full.data(), &pos, c.rows, c.np, nb, scale, wscale, c.heads, c.loop,
                        pages.pt.data(), PE * 2);
  });
  std::vector<int> count(c.rows, 0), group((c.rows + 7) / 8, 0);
  auto scores = [&](int k) {  // k > 0: the first pass, which cuts each row group at its end
    const unsigned gx = (unsigned)((c.rows + 7) / 8), gy = (unsigned)((nb + c.loop - 1) / c.loop);
    run_grid(gx, gy, 256, [&] {
      gridDim = {gx, gy, 1};
      tf_glm_index_scores_prune(q.data(), w.data(), W, pk, pruned.data(), &pos, c.rows, c.np, nb, scale, wscale, c.heads,
                                c.loop, pages.pt.data(), PE * 2, plan.data(), pages_n, k, ub.data(), group.data(), stats);
    });
  };
  // the engine's sequence twice on the same scratch: the plan's row counters must be back to 0 after each launch
  std::vector<float> first;
  for (int pass = 0; pass < 2; pass++) {
    std::fill(pruned.begin(), pruned.end(), 7e8f);
    stats[0] = stats[1] = stats[2] = 0;
    const unsigned gy = (unsigned)((pages_n + glm_prune::PB - 1) / glm_prune::PB);
    run_grid(c.rows, gy, glm_prune::T, [&] {
      gridDim = {(unsigned)c.rows, gy, 1};
      if (fault == 1)
        plan_unsound(q.data(), w.data(), W, pk, pages.pt.data(), PE * 2, &pos, scale, wscale, c.heads, plan.data(),
                     ub.data(), pages_n, c.seeds, count.data());
      else
        tf_glm_prune_plan(q.data(), w.data(), W, pk, pages.pt.data(), PE * 2, &pos, scale, wscale, c.heads, plan.data(),
                          ub.data(), pages_n, c.seeds, count.data());
    });
    for (int v : count) res.differ += v != 0;
    scores(c.k);
    for (int v : group) res.differ += v != 0;
    scores(0);
    if (pass == 0) first = pruned;
    else res.differ += std::memcmp(first.data(), pruned.data(), first.size() * 4) != 0;
  }
  for (int64_t r = 0; r < c.rows; r++) {
    const int64_t vis = (c.pos + r + 1) / 4, limit = std::min(c.np, std::max<int64_t>(vis, c.k));
    const float* a = &full[r * c.np];
    const float* b = &pruned[r * c.np];
    for (int64_t p = 0; p < limit; p++)  // every pool the pruned scan wrote is the full scan's, or -inf
      res.stray += bits(b[p]) != bits(a[p]) && b[p] != -INFINITY;
    res.differ += top_k(a, limit, c.k) != top_k(b, limit, c.k);
  }
  res.rows = c.rows;
  res.skipped = stats[0];
  res.whole = stats[1];
  return res;
}

int main() {
  std::mt19937_64 g(0x5eed);
  const Case cases[] = {
      {16384 - 3, 9, 8192, 32, 512, 16, 8, clustered, true, "clustered, negative weights"},
      {24000, 16, 8192, 32, 512, 16, 8, clustered, false, "clustered, positive weights"},
      {24000, 8, 8192, 32, 64, 1, 8, clustered, true, "clustered, k 64, decode tiling"},
      {20000, 8, 8192, 20, 128, 16, 4, clustered, true, "20 heads, 4 seeds"},
      {20000, 8, 8192, 32, 512, 16, 8, normal, true, "unclustered"},
      {20000, 8, 8192, 32, 512, 16, 8, ties, true, "tied pages"},
      {20000, 8, 8192, 32, 512, 16, 8, equal, true, "every key equal"},
      {20000, 8, 8192, 32, 256, 16, 8, nan_page, true, "a NaN key"},
      {20000, 8, 8192, 32, 256, 16, 8, nonfinite, true, "non-finite keys and query (gate 1)"},
      {20000, 4, 8192, 32, 256, 1, 8, nonfinite, false, "non-finite, decode tiling"},
      {3000, 8, 1024, 32, 512, 16, 8, clustered, true, "short: no whole pages past the seeds"},
  };
  int fails = 0;
  int64_t skipped = 0, whole = 0;
  for (const Case& c : cases) {
    const Result r = run_case(g, c, 0);
    printf("%-40s rows %lld: %lld selections differ, %lld stray scores, %lld of %lld whole pages skipped\n", c.name,
           (long long)r.rows, (long long)r.differ, (long long)r.stray, (long long)r.skipped, (long long)r.whole);
    fails += r.differ != 0 || r.stray != 0;
    skipped += r.skipped;
    whole += r.whole;
  }
  if (skipped == 0) {
    printf("no page was skipped: the comparison proved nothing\n");
    fails++;
  }
  for (int fault : {1, 2}) {
    int64_t caught = 0;
    for (int i = 0; i < 4 && caught == 0; i++) caught += run_case(g, cases[fault == 1 ? 0 : 1], fault).differ;
    printf("planted fault %s: %s\n", fault == 1 ? "negative weights bounded from above" : "strips never written",
           caught ? "caught (selections differ)" : "NOT caught");
    fails += caught == 0;
  }
  printf("%s (%d failures; %lld of %lld whole pages skipped)\n", fails ? "FAIL" : "PASS", fails, (long long)skipped,
         (long long)whole);
  return fails != 0;
}
