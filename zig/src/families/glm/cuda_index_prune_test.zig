//! cuda_index_prune against its oracle: the pruned selection equals the full scan's on every case, including
//! negative head weights and tied scores, and an unsound bound is caught.
const std = @import("std");
const prune = @import("cuda_index_prune.zig");
const ref = @import("cuda_index_prune_ref.zig");
const t = std.testing;

const Case = struct { heads: usize = 4, dim: usize = 16, pages: usize = 24, page_pools: usize = 32, tail: usize = 7, k: usize = 48, clustered: bool = true, ties: bool = false, negative: bool = true };

const Run = struct { same: bool, stats: prune.Stats };

/// Keys in pages with their own offsets (clustered: some pages far from the query, so bounds can prune), queries
/// and weights at random; `ties` repeats one key across whole pages so equal scores straddle the k-th place.
fn run(seed: u64, c: Case, sound: bool) !Run {
    const gpa = t.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const pools = c.pages * c.page_pools + c.tail;
    const keys = try gpa.alloc(f32, pools * c.dim);
    defer gpa.free(keys);
    for (0..pools) |p| {
        const page = p / c.page_pools;
        const shift: f32 = if (c.clustered) @as(f32, @floatFromInt(page % 5)) - 2 else 0;
        for (0..c.dim) |d| keys[p * c.dim + d] = if (c.ties and page % 3 == 0) 0.5 else r.floatNorm(f32) * 0.5 + shift;
    }
    const q = try gpa.alloc(f32, c.heads * c.dim);
    defer gpa.free(q);
    for (q) |*v| v.* = r.floatNorm(f32) * 0.3 + 0.2;
    const w = try gpa.alloc(f32, c.heads);
    defer gpa.free(w);
    for (w, 0..) |*v, h| v.* = if (c.negative and h % 2 == 1) -r.float(f32) else r.float(f32);
    const x: prune.Query = .{ .q = q, .w = w, .heads = c.heads, .dim = c.dim, .scale = 0.125 };
    const bounds = try gpa.alloc(f32, c.pages * 2 * c.dim);
    defer gpa.free(bounds);
    prune.pageBounds(keys, c.dim, c.page_pools, c.pages, bounds);
    const a = try gpa.alloc(u32, c.k);
    defer gpa.free(a);
    const b = try gpa.alloc(u32, c.k);
    defer gpa.free(b);
    var stats: prune.Stats = .{};
    const all = try prune.selectAll(gpa, x, keys, pools, c.k, a);
    const some = try prune.selectPruned(gpa, x, keys, pools, c.k, c.page_pools, bounds, c.pages, sound, b, &stats);
    return .{ .same = std.mem.eql(u32, all, some), .stats = stats };
}

test "GLM index: TF_GLM_INDEX_PRUNE is off unless 1, and a typo is refused" {
    try t.expectEqual(prune.Mode.off, try prune.Mode.parse(null));
    try t.expectEqual(prune.Mode.off, try prune.Mode.parse("0"));
    try t.expectEqual(prune.Mode.on, try prune.Mode.parse("1"));
    try t.expectError(error.BadIndexPruneMode, prune.Mode.parse("yes"));
    try t.expectEqual(prune.Mode.off, try prune.Mode.read()); // unset in the test environment
}

test "GLM index: the pruned selection equals the full scan on every seed, and it does skip pages" {
    var skipped: usize = 0;
    for ([_]Case{ .{}, .{ .negative = false }, .{ .clustered = false }, .{ .ties = true }, .{ .k = 1 }, .{ .k = 24 * 32 + 7 }, .{ .tail = 0, .pages = 1 } }) |c| {
        for (0..40) |seed| {
            const r = try run(seed, c, true);
            if (!r.same) {
                std.debug.print("pruned selection differs: seed {d} case {any}\n", .{ seed, c });
                return error.PrunedSelectionDiffers;
            }
            skipped += r.stats.skipped;
        }
    }
    try t.expect(skipped > 0); // the clustered cases prune: the oracle compared a real skip, not a full scan
}

test "GLM index: an unsound bound (negative head weights bounded from above) is caught by the oracle" {
    var caught = false;
    for (0..200) |seed| {
        const r = try run(seed, .{ .clustered = true, .negative = true }, false);
        if (!r.same) {
            caught = true;
            break;
        }
    }
    try t.expect(caught);
}

/// The device layout's cases: keys as bf16 in shuffled pages of `page_elems` (keys, strip, other rows), strips
/// written chunk by chunk as indexUpdate writes them, query rows at the last positions.
const Dev = struct { heads: usize = 8, pos: usize = 12_000, np: usize = 4096, rows: usize = 2, k: usize = 128, seeds: usize = 4, keys: enum { clustered, normal, ties, equal, nan } = .clustered, negative: bool = true };

const DevRun = struct { differ: usize, stats: ref.Stats };

/// Full scan and the mirrored device prune for each row; `fault`: none, negative weights bounded from above, or the
/// strips left as zero (a restore that skipped them).
const Fault = enum { none, unsound, stale };

fn devRun(seed: u64, c: Dev, fault: Fault) !DevRun {
    const gpa = t.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const tokens = c.pos + c.rows;
    const pages = tokens / 4 / ref.page_pools + 2;
    const page_elems = ref.page_pools * ref.d + 2 * ref.d + 64;
    const store = try gpa.alloc(u16, pages * page_elems);
    defer gpa.free(store);
    @memset(store, ref.toBf16(std.math.nan(f32))); // strips poisoned until written
    const pt = try gpa.alloc(u32, pages);
    defer gpa.free(pt);
    for (pt, 0..) |*v, i| v.* = @intCast(i);
    r.shuffle(u32, pt);
    const p: ref.Pages = .{ .store = store, .pt = pt, .page_elems = page_elems };
    for (0..pages * ref.page_pools) |pool| {
        const g = pool / ref.page_pools;
        const shift: f32 = @as(f32, @floatFromInt(g % 7)) - 3;
        const key = store[pt[g] * page_elems + (pool % ref.page_pools) * ref.d ..][0..ref.d];
        for (key) |*v| v.* = ref.toBf16(switch (c.keys) {
            .clustered => r.floatNorm(f32) * 0.4 + shift,
            .normal, .nan => r.floatNorm(f32),
            .ties => if (g % 3 == 0) 0.5 else r.floatNorm(f32) * 0.4 + shift,
            .equal => 0.75,
        });
    }
    if (c.keys == .nan) store[pt[3] * page_elems + 5 * ref.d + 7] = ref.toBf16(std.math.nan(f32));
    var at: usize = 0;
    while (at < tokens) { // the writes: chunks of up to 2,048 rows, then decode-sized ones
        const n = @min(tokens - at, if (at < c.pos / 2) 2048 else 1 + r.uintLessThan(usize, 9));
        p.writeStrips(at, n);
        at += n;
    }
    if (fault == .stale) for (0..pages) |g| @memset(p.strip(g), 0);
    var out: DevRun = .{ .differ = 0, .stats = .{} };
    for (0..c.rows) |row| {
        const q = try gpa.alloc(u16, c.heads * ref.d);
        defer gpa.free(q);
        for (q) |*v| v.* = ref.toBf16(r.floatNorm(f32) * 0.3 + 0.15);
        var wv: [32]f32 = @splat(0);
        for (wv[0..c.heads], 0..) |*w, h| w.* = ref.fromBf16(ref.toBf16(if (c.negative and h % 2 == 1) -r.float(f32) else r.float(f32))) * 0.176776692;
        const x: ref.Row = .{ .q = q, .wv = wv, .heads = c.heads, .scale = 0.0883883461, .pos = c.pos + row };
        const limit = @min(c.np, @max((x.pos + 1) / 4, c.k));
        var unused: ref.Stats = .{};
        const full = try ref.rowScores(gpa, p, x, c.np, c.k, c.seeds, false, .sound, &unused);
        defer gpa.free(full);
        const pruned = try ref.rowScores(gpa, p, x, c.np, c.k, c.seeds, true, if (fault == .unsound) .unsound else .sound, &out.stats);
        defer gpa.free(pruned);
        const a = try ref.select(gpa, full, limit, c.k);
        defer gpa.free(a);
        const b = try ref.select(gpa, pruned, limit, c.k);
        defer gpa.free(b);
        out.differ += @intFromBool(!std.mem.eql(u32, a, b));
    }
    return out;
}

test "GLM index: on the device layout, strips written chunk by chunk give the full scan's selection, pages skipped" {
    var skipped: usize = 0;
    var whole: usize = 0;
    const cases = [_]Dev{ .{}, .{ .negative = false }, .{ .keys = .normal }, .{ .keys = .ties }, .{ .keys = .equal }, .{ .keys = .nan }, .{ .k = 512, .seeds = 8 }, .{ .heads = 32, .rows = 1 }, .{ .pos = 2000, .np = 1024 } };
    for (cases) |c| for (0..3) |seed| {
        const res = try devRun(seed, c, .none);
        if (res.differ != 0) {
            std.debug.print("device-layout prune differs: seed {d} case {any}\n", .{ seed, c });
            return error.PrunedSelectionDiffers;
        }
        skipped += res.stats.skipped;
        whole += res.stats.whole;
    };
    try t.expect(skipped > 0 and skipped < whole); // pages were skipped, and not every page
}

test "GLM index: on the device layout, the unsound bound and strips left unwritten are caught" {
    for ([_]Fault{ .unsound, .stale }) |fault| {
        var caught = false;
        for (0..20) |seed| {
            // zero strips bound every page near 0, so they show where scores run positive (positive head weights)
            if ((try devRun(seed, .{ .negative = fault == .unsound }, fault)).differ > 0) {
                caught = true;
                break;
            }
        }
        try t.expect(caught);
    }
}

test "GLM index: a strip is each dimension's min and max, NaN when its column holds one, written once whole" {
    const gpa = t.allocator;
    const page_elems = ref.page_pools * ref.d + 2 * ref.d;
    const store = try gpa.alloc(u16, 2 * page_elems);
    defer gpa.free(store);
    @memset(store, ref.toBf16(-7));
    const pt = [_]u32{ 1, 0 };
    const p: ref.Pages = .{ .store = store, .pt = &pt, .page_elems = page_elems };
    for (0..ref.page_pools) |i| for (0..ref.d) |c| {
        store[pt[0] * page_elems + i * ref.d + c] = ref.toBf16(@as(f32, @floatFromInt(i)) - 64 + @as(f32, @floatFromInt(c)) / 256);
    };
    store[pt[0] * page_elems + 9 * ref.d + 3] = ref.toBf16(std.math.nan(f32));
    p.writeStrips(0, 511); // 127 pools: not whole, no strip
    try t.expectEqual(ref.toBf16(-7), p.strip(0)[0]);
    p.writeStrips(511, 1);
    try t.expectEqual(ref.toBf16(-64), p.strip(0)[0]);
    try t.expectEqual(ref.toBf16(63 + 1.0 / 256.0), p.strip(0)[ref.d + 1]);
    try t.expect(std.math.isNan(ref.fromBf16(p.strip(0)[3])) and std.math.isNan(ref.fromBf16(p.strip(0)[ref.d + 3])));
    try t.expectEqual(ref.toBf16(-7), p.strip(1)[0]); // page 1 not reached
}
