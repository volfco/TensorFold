//! cuda_index_prune against its oracle: the pruned selection equals the full scan's on every case, including
//! negative head weights and tied scores, and an unsound bound is caught.
const std = @import("std");
const prune = @import("cuda_index_prune.zig");
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

test "GLM index: TF_GLM_INDEX_PRUNE is off unless 1, a typo is refused, and a device run refuses it by name" {
    try t.expectEqual(prune.Mode.off, try prune.Mode.parse(null));
    try t.expectEqual(prune.Mode.off, try prune.Mode.parse("0"));
    try t.expectEqual(prune.Mode.on, try prune.Mode.parse("1"));
    try t.expectError(error.BadIndexPruneMode, prune.Mode.parse("yes"));
    try prune.refuseOnDevice(); // unset in the test environment
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
