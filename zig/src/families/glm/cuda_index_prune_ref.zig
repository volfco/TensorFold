//! TF_GLM_INDEX_PRUNE's device path mirrored on the host on its data layout (strips, bound, seeds, cut) for the oracle.
const std = @import("std");

pub const d = 128;
pub const page_pools = 128;

pub fn fromBf16(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

pub fn toBf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((b >> 16) | 0x40);
    return @intCast((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
}

/// sparse._order_key: unsigned order is the scores' (-0 as +0, any NaN above +inf).
pub fn orderKey(s: f32) u32 {
    const b: i32 = @bitCast(s + 0.0);
    return @bitCast(b ^ ((b >> 31) | @as(i32, @bitCast(@as(u32, 0x80000000)))));
}

/// Pooled keys on pages: `store` [pages][page_elems] bf16, logical page g at pt[g]; the strip after its 128 keys.
pub const Pages = struct {
    store: []u16,
    pt: []const u32,
    page_elems: usize,

    pub fn key(p: Pages, pool: usize) []const u16 {
        return p.store[p.pt[pool / page_pools] * p.page_elems + (pool % page_pools) * d ..][0..d];
    }

    pub fn strip(p: Pages, g: usize) []u16 {
        return p.store[p.pt[g] * p.page_elems + page_pools * d ..][0 .. 2 * d];
    }

    /// tf_glm_prune_bounds after a write of `rows` rows at `pos`: the strip of every page it completes.
    pub fn writeStrips(p: Pages, pos: usize, rows: usize) void {
        const done = (pos + rows) / 4;
        var g = pos / 4 / page_pools;
        while ((g + 1) * page_pools <= done) : (g += 1) {
            const s = p.strip(g);
            for (0..d) |c| {
                var lo = fromBf16(p.key(g * page_pools)[c]);
                var hi = lo;
                for (1..page_pools) |i| {
                    const v = fromBf16(p.key(g * page_pools + i)[c]);
                    lo = if (std.math.isNan(v) or v < lo) v else lo;
                    hi = if (std.math.isNan(v) or v > hi) v else hi;
                }
                s[c] = toBf16(lo);
                s[d + c] = toBf16(hi);
            }
        }
    }
};

/// One query row: `heads` rows of d bf16 values and their head weights as the scores kernel forms them (wv).
pub const Row = struct { q: []const u16, wv: [32]f32, heads: usize, scale: f32, pos: usize };

/// The planted fault the oracle must catch: negative head weights bounded from above.
pub const Bound = enum { sound, unsound };

/// glm_prune::page_bound in its fp32 order: each head's sums over d in order, the 32 lanes' butterfly, the margin.
pub fn bound(x: Row, s: []const u16, how: Bound) f32 {
    var term: [32]f32 = @splat(0);
    var mag: [32]f32 = @splat(0);
    var bad = false;
    for (s) |v| bad = bad or !std.math.isFinite(fromBf16(v));
    for (0..x.heads) |h| {
        var up: f32 = 0;
        var down: f32 = 0;
        var abs: f32 = 0;
        for (0..d) |c| {
            const q = fromBf16(x.q[h * d + c]);
            const lo = fromBf16(s[c]);
            const hi = fromBf16(s[d + c]);
            up += @max(q * lo, q * hi);
            down += @min(q * lo, q * hi);
            abs += @abs(q) * @max(@abs(lo), @abs(hi));
        }
        const w = x.wv[h];
        term[h] = if (w >= 0 or how == .unsound) w * @max(x.scale * up, 0) else w * @max(x.scale * down, 0);
        mag[h] = @abs(w) * x.scale * abs;
    }
    const t = butterfly(term);
    const m = butterfly(mag);
    var absw: f32 = 0;
    for (x.wv) |w| absw += @abs(w);
    if (bad or !(m < 1e30) or std.math.isNan(t)) return std.math.nan(f32);
    return t + m * (1.0 / 256.0) + absw * 1e-33 + 1e-30;
}

fn butterfly(v0: [32]f32) f32 {
    var v = v0;
    var o: usize = 16;
    while (o > 0) : (o >>= 1) {
        var n: [32]f32 = undefined;
        for (0..32) |l| n[l] = v[l] + v[l ^ o];
        v = n;
    }
    return v[0];
}

/// tf_glm_index_scores' score of `key` as emulate.cpp's model_score: exact k16 sums into fp32, then its head order.
pub fn score(x: Row, key: []const u16) f32 {
    var v: [32]f32 = @splat(0);
    for (0..x.heads) |h| {
        var acc: f32 = 0;
        for (0..d / 16) |kq| {
            var sum: f64 = 0;
            for (0..16) |k| sum += @as(f64, fromBf16(x.q[h * d + kq * 16 + k])) * fromBf16(key[kq * 16 + k]);
            acc = @floatCast(@as(f64, acc) + sum);
        }
        v[h] = @max(acc * x.scale, 0);
    }
    const w = x.wv;
    var s: [8]f32 = undefined;
    for (0..8) |g| {
        var a = w[g + 8] * v[g + 8];
        a = @mulAdd(f32, w[g], v[g], a);
        a = @mulAdd(f32, w[g + 16], v[g + 16], a);
        s[g] = @mulAdd(f32, w[g + 24], v[g + 24], a);
    }
    var t: [8]f32 = undefined;
    for (0..8) |g| t[g] = s[g] + s[g ^ 4];
    var u: [8]f32 = undefined;
    for (0..8) |g| u[g] = t[g] + t[g ^ 2];
    return u[0] + u[1];
}

/// The best `k` of row[0..limit) by order key, ties to the lower pool, ascending (tf_glm_select_rows' picks).
pub fn select(gpa: std.mem.Allocator, row: []const f32, limit: usize, k: usize) ![]u32 {
    const idx = try gpa.alloc(u32, limit);
    defer gpa.free(idx);
    for (idx, 0..) |*v, i| v.* = @intCast(i);
    std.mem.sort(u32, idx, row, struct {
        fn lt(r: []const f32, a: u32, b: u32) bool {
            const ka = orderKey(r[a]);
            const kb = orderKey(r[b]);
            return ka > kb or (ka == kb and a < b);
        }
    }.lt);
    const out = try gpa.dupe(u32, idx[0..k]);
    std.mem.sort(u32, out, {}, std.sort.asc(u32));
    return out;
}

pub const Stats = struct { whole: usize = 0, skipped: usize = 0 };

/// One row's scores over `np` pools, as the device leaves them: the full scan (`prune` false) or the planned passes.
pub fn rowScores(gpa: std.mem.Allocator, p: Pages, x: Row, np: usize, k: usize, seeds: usize, prune: bool, how: Bound, stats: *Stats) ![]f32 {
    const npool = (x.pos + 1) / 4;
    const row = try gpa.alloc(f32, np);
    for (row, 0..) |*v, i| v.* = if (i < npool) score(x, p.key(i)) else -std.math.inf(f32);
    if (!prune) return row;
    const pages = (np + page_pools - 1) / page_pools;
    const whole = @min(npool / page_pools, pages);
    stats.whole += whole;
    if (whole <= seeds) return row;
    const plan = try gpa.alloc(u8, pages);
    defer gpa.free(plan);
    const ub = try gpa.alloc(f32, pages);
    defer gpa.free(ub);
    @memset(plan, 0);
    for (0..whole) |g| {
        ub[g] = bound(x, p.strip(g), how);
        plan[g] = if (std.math.isNan(ub[g])) 0 else 1;
    }
    for (0..seeds) |_| { // the largest bound among the candidates, ties to the lower page
        var at: ?usize = null;
        for (0..whole) |g| if (plan[g] == 1 and (at == null or ub[g] > ub[at.?])) {
            at = g;
        };
        if (at) |g| plan[g] = 0;
    }
    // the cut: the k-th best key among the scored pools under limit (none when fewer than k)
    const limit = @min(np, @max(npool, k));
    var keys: std.ArrayList(u32) = .empty;
    defer keys.deinit(gpa);
    for (0..limit) |i| if (plan[i / page_pools] == 0) try keys.append(gpa, orderKey(row[i]));
    var cut: u32 = 0;
    if (keys.items.len >= k) {
        std.mem.sort(u32, keys.items, {}, std.sort.desc(u32));
        cut = keys.items[k - 1];
    }
    for (0..pages) |g| if (plan[g] == 1 and !std.math.isNan(ub[g]) and orderKey(ub[g]) < cut) {
        stats.skipped += 1;
        for (g * page_pools..@min(np, (g + 1) * page_pools)) |i| row[i] = -std.math.inf(f32);
    };
    return row;
}
