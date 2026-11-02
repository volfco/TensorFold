//! TF_GLM_INDEX_PRUNE: an exact page prune of the DSA indexer's scan from per-page key bounds (each whole page's
//! per-dimension min and max of its pooled keys: the hint sidecar's `bounds`, zig/src/core/kv_hints.zig, and on the
//! device the strip each page carries, cuda_index_prune_dev.zig). The indexer scores a pool as sum over heads of
//! w[h] * relu(scale * q[h] . k) and keeps the best `topk`; a page whose upper bound on that score lies strictly below
//! the k-th best of scores already computed cannot hold a selected pool, so skipping it leaves the selection
//! unchanged. This file is the host consumer and its oracle (scan pages best bound first); cuda_index_prune_ref.zig
//! mirrors the device path on its data layout for the same oracle.
const std = @import("std");

pub const Mode = enum {
    off,
    on,

    pub fn parse(text: ?[]const u8) !Mode {
        const t = text orelse return .off;
        if (t.len == 0 or std.mem.eql(u8, t, "0")) return .off;
        if (std.mem.eql(u8, t, "1")) return .on;
        return error.BadIndexPruneMode;
    }

    pub fn read() !Mode {
        const v = std.c.getenv("TF_GLM_INDEX_PRUNE") orelse return .off;
        return parse(std.mem.span(v));
    }
};

/// One query row: `heads` rows of `dim` values (bf16 in the engine, f32 here), its head weights and the score scale.
pub const Query = struct { q: []const f32, w: []const f32, heads: usize, dim: usize, scale: f32 };

/// The pool's score as the indexer defines it (f64 sums: the oracle's arithmetic, the same for both selections).
pub fn score(x: Query, key: []const f32) f64 {
    var s: f64 = 0;
    for (0..x.heads) |h| {
        var dot: f64 = 0;
        for (0..x.dim) |d| dot += @as(f64, x.q[h * x.dim + d]) * key[d];
        s += @as(f64, x.w[h]) * @max(@as(f64, x.scale) * dot, 0);
    }
    return s;
}

/// The page's upper bound on any pool's score from its per-dimension `lo` and `hi`. A positive head weight takes the
/// dot product's upper bound, a negative one its lower bound (relu is monotone, scale positive). `margin` covers the
/// device's f32 rounding: a relative 2^-8 of the bound's absolute terms, generous against a 128-term f32 chain.
/// `sound` false bounds every head from above: the unsound variant the oracle test must catch.
pub fn bound(x: Query, lo: []const f32, hi: []const f32, sound: bool) f64 {
    var s: f64 = 0;
    var mag: f64 = 0;
    for (0..x.heads) |h| {
        var up: f64 = 0;
        var down: f64 = 0;
        var abs: f64 = 0;
        for (0..x.dim) |d| {
            const q: f64 = x.q[h * x.dim + d];
            const a = q * lo[d];
            const b = q * hi[d];
            up += @max(a, b);
            down += @min(a, b);
            abs += @abs(q) * @max(@abs(lo[d]), @abs(hi[d]));
        }
        const w: f64 = x.w[h];
        const sc: f64 = x.scale;
        s += if (w >= 0 or !sound) w * @max(sc * up, 0) else w * @max(sc * down, 0);
        mag += @abs(w) * sc * abs;
    }
    return s + mag / 256 + 1e-30;
}

pub const Stats = struct { pages: usize = 0, skipped: usize = 0, scored: usize = 0 };

/// A pool with its score; the best come first, equal scores in pool order (the rule both selections share: it
/// depends only on the score and the pool, never on the scan order).
const Scored = struct { s: f64, pool: u32 };
fn better(a: Scored, b: Scored) bool {
    return a.s > b.s or (a.s == b.s and a.pool < b.pool);
}

/// The best `k` of `items` kept sorted best-first in `top` (insertion; k is the indexer's 512 at most).
fn offer(top: *std.ArrayList(Scored), gpa: std.mem.Allocator, k: usize, x: Scored) !void {
    if (top.items.len == k and !better(x, top.items[k - 1])) return;
    var i = top.items.len;
    while (i > 0 and better(x, top.items[i - 1])) i -= 1;
    try top.insert(gpa, i, x);
    if (top.items.len > k) _ = top.pop();
}

fn ascending(top: []const Scored, out: []u32) []u32 {
    for (top, 0..) |x, i| out[i] = x.pool;
    std.mem.sort(u32, out[0..top.len], {}, std.sort.asc(u32));
    return out[0..top.len];
}

/// The plain selection: every visible pool scored, the best `k` in ascending pool order (tf_glm_select_rows' form).
pub fn selectAll(gpa: std.mem.Allocator, x: Query, keys: []const f32, pools: usize, k: usize, out: []u32) ![]u32 {
    var top: std.ArrayList(Scored) = .empty;
    defer top.deinit(gpa);
    for (0..pools) |p| try offer(&top, gpa, k, .{ .s = score(x, keys[p * x.dim ..][0..x.dim]), .pool = @intCast(p) });
    return ascending(top.items, out);
}

/// The pruned selection: whole pages with bounds (`bounds` [pages][lo | hi][dim]) scanned best bound first until the
/// running k-th best is strictly above the next page's bound; the pools past them (no bound yet) always scanned.
pub fn selectPruned(gpa: std.mem.Allocator, x: Query, keys: []const f32, pools: usize, k: usize, page_pools: usize, bounds: []const f32, pages: usize, sound: bool, out: []u32, stats: *Stats) ![]u32 {
    std.debug.assert(pages * page_pools <= pools and x.scale > 0);
    var top: std.ArrayList(Scored) = .empty;
    defer top.deinit(gpa);
    for (pages * page_pools..pools) |p| try offer(&top, gpa, k, .{ .s = score(x, keys[p * x.dim ..][0..x.dim]), .pool = @intCast(p) });
    const order = try gpa.alloc(Scored, pages);
    defer gpa.free(order);
    for (order, 0..) |*o, p| {
        const b = bounds[p * 2 * x.dim ..];
        o.* = .{ .s = bound(x, b[0..x.dim], b[x.dim..][0..x.dim], sound), .pool = @intCast(p) };
    }
    std.mem.sort(Scored, order, {}, struct {
        fn lt(_: void, a: Scored, b: Scored) bool {
            return better(a, b);
        }
    }.lt);
    stats.pages += pages;
    for (order, 0..) |o, i| {
        if (top.items.len == k and top.items[k - 1].s > o.s) { // strictly: an equal score could still tie in
            stats.skipped += pages - i;
            break;
        }
        for (o.pool * page_pools..(o.pool + 1) * page_pools) |p| try offer(&top, gpa, k, .{ .s = score(x, keys[p * x.dim ..][0..x.dim]), .pool = @intCast(p) });
        stats.scored += 1;
    }
    return ascending(top.items, out);
}

/// Each whole page's [lo | hi] over its pools' keys, as the post-processor writes them (tools/kvprobe/hints.py).
pub fn pageBounds(keys: []const f32, dim: usize, page_pools: usize, pages: usize, out: []f32) void {
    for (0..pages) |p| {
        const b = out[p * 2 * dim ..][0 .. 2 * dim];
        @memcpy(b[0..dim], keys[p * page_pools * dim ..][0..dim]);
        @memcpy(b[dim..], keys[p * page_pools * dim ..][0..dim]);
        for (p * page_pools..(p + 1) * page_pools) |q| for (0..dim) |d| {
            b[d] = @min(b[d], keys[q * dim + d]);
            b[dim + d] = @max(b[dim + d], keys[q * dim + d]);
        };
    }
}
