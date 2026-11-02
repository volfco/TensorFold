//! TF_GLM_INDEX_PRUNE on the device: page strips, plan, planned scores, cut (tools/kvprobe/STATUS-prune.md: exactness).
const std = @import("std");
const cuda = @import("cuda");

/// glm_prune::PP, MAXP: pools a page, pages a row's plan covers (its scratch: ub f32 and plan u8 per page a row).
pub const page_pools = 128;
pub const max_pages = 1024;
/// The strip a cache's part of a page gains: 128 minima and 128 maxima, bf16 (the index key width is 128).
pub const strip_bytes = 2 * 128 * 2;
/// Pages a row scores before its cut: 8 x 128 pools, twice the indexer's 512.
pub const seeds = 8;
/// glm_prune::T, D: the plan's and cut's block, the strips' block.
pub const threads = 256;
pub const strip_threads = 128;
/// The selects that prune: pool buckets past 4,096 (past 16,384 tokens, where the dump found prunable pages).
pub const from_pools = 4097;

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// The plan's pages for a select over `np` pools.
pub fn pagesFor(np: usize) usize {
    return cdiv(np, page_pools);
}

/// Whether a select over `np` pools of a cache with strips (`strip`) at `pk`, pages `ps` bytes apart, prunes.
pub fn applies(np: usize, strip: bool, pt: u64, pk: u64, ps: u64) bool {
    return strip and pt != 0 and np >= from_pools and pagesFor(np) <= max_pages and pk % 16 == 0 and ps % 16 == 0;
}

/// A sequence of `capacity` tokens: its rows' plans fit a row of the select scratch (`sparse.select_list` words).
pub fn fits(capacity: usize) bool {
    return pagesFor(capacity / 4) <= max_pages;
}

/// An engine's TF_GLM_INDEX_PRUNE (`s`: cuda_ranks.Settings): with its kernels, on pages, within a plan's pages.
pub fn admit(s: anytype, kernels: bool, capacity: usize) !void {
    if (s.index_prune != kernels) return error.IndexPruneSettings;
    if (!s.index_prune) return;
    if (!s.pages) return error.IndexPruneNeedsPages;
    if (!fits(capacity)) return error.IndexPruneCapacity;
}

/// The scores tiling: a prompt chunk's (16 tiles a CTA, as TF_GLM_INDEX_CUDA) or a decode window's (a tile a CTA).
pub fn loopFor(rows: usize) usize {
    return if (rows >= 64) 16 else 1;
}

/// Strip-writing grids: after a write of `rows` rows (every page it may complete), and for a restored `rows` rows.
pub fn writeGrid(rows: usize) usize {
    return rows / 512 + 2;
}

pub fn restoreGrid(rows: usize) usize {
    return rows / 4 / page_pools;
}

/// glm_prune::PB: pages one plan block bounds (a row's plan is cdiv(pages, PB) blocks).
pub const plan_pages = 32;
/// TF_GLM_INDEX_PRUNE_MIN_PAGES' default: whole pages before a select prunes (131,072 tokens; GB10 timing, 2026-11-02).
pub const default_min_pages = 256;

/// TF_GLM_INDEX_PRUNE_MIN_PAGES: unset (the default) or a page count from 0 to max_pages; anything else is refused.
pub fn parseMinPages(text: ?[]const u8) !usize {
    const v = text orelse return default_min_pages;
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadIndexPruneMinPages;
    return if (n <= max_pages) n else error.BadIndexPruneMinPages;
}

/// The skipped pages, whole visible pages and pruned rows the cut has counted (u64 each, as the device adds them).
pub const Counts = struct { skipped: u64 = 0, whole: u64 = 0, rows: u64 = 0 };

pub const Kernels = struct {
    module: cuda.Module,
    bounds_fn: cuda.Function,
    plan_fn: cuda.Function,
    scores_fn: cuda.Function,
    min_pages: usize = default_min_pages,
    counts: cuda.DeviceBuffer, // three u64 the cut adds to, read and cleared by `pass`
    total: Counts = .{},
    rank: usize = 0, // the last `pass`'s, for the totals line

    /// `image`: cuda.kernels.glm_index (an image without them is refused by name, before any request).
    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        if (image.len == 0) return error.IndexPruneNoKernels;
        const min_pages = try parseMinPages(if (std.c.getenv("TF_GLM_INDEX_PRUNE_MIN_PAGES")) |v| std.mem.span(v) else null);
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        var counts = try cuda.DeviceBuffer.alloc(d, 32);
        errdefer counts.free();
        try counts.fill8(0, null);
        return .{ .module = m, .bounds_fn = try m.function("tf_glm_prune_bounds"), .plan_fn = try m.function("tf_glm_prune_plan"), .scores_fn = try m.function("tf_glm_index_scores_prune"), .min_pages = min_pages, .counts = counts };
    }

    /// The totals line (INDEXPRUNE rank=N selects=... pages=... skipped=... share=...), then the module.
    pub fn unload(k: *Kernels) void {
        _ = k.pass(null, k.rank, 0) catch {};
        log("INDEXPRUNE rank={d} total min_pages={d}", k.rank, k.min_pages, k.total);
        k.counts.free();
        k.module.unload();
    }

    /// Whether a select of rows ending at token `tokens` over a cache like `applies`' prunes: past min_pages.
    pub fn takes(k: *const Kernels, np: usize, strip: bool, pt: u64, pk: u64, ps: u64, tokens: usize) bool {
        return applies(np, strip, pt, pk, ps) and tokens / (4 * page_pools) >= @max(k.min_pages, seeds + 1);
    }

    /// A prompt pass of `tokens` ended (`s`, or null): the counts since the last pass, logged and added to the total.
    pub fn pass(k: *Kernels, s: ?cuda.Stream, rank: usize, tokens: usize) !Counts {
        if (s) |x| try x.synchronize();
        var v: [4]u64 = undefined;
        try k.counts.download(0, std.mem.sliceAsBytes(&v));
        try k.counts.fill8(0, null);
        const c: Counts = .{ .skipped = v[0], .whole = v[1], .rows = v[2] };
        k.total.skipped += c.skipped;
        k.total.whole += c.whole;
        k.total.rows += c.rows;
        k.rank = rank;
        if (s != null and c.rows > 0) log("INDEXPRUNE rank={d} pass tokens={d}", rank, tokens, c);
        return c;
    }

    fn log(comptime head: []const u8, a: usize, b: usize, c: Counts) void {
        if (@import("builtin").is_test) return;
        const share = if (c.whole > 0) @as(f64, @floatFromInt(c.skipped)) / @as(f64, @floatFromInt(c.whole)) else 0;
        std.log.info(head ++ " selects={d} pages={d} skipped={d} share={d:.4}", .{ a, b, c.rows, c.whole, c.skipped, share });
    }

    /// The strips of the pages a write of `rows` rows at device `pos` completes (after tf_glm_pool_keys).
    pub fn bounds(k: *const Kernels, s: cuda.Stream, pk: u64, pt: u64, ps: u64, pos: u64, rows: usize) !void {
        try k.strips(s, pk, pt, ps, pos, rows, writeGrid(rows));
    }

    /// Every whole page's strip of a state restored with `rows` rows (its pooled keys were copied, not written).
    pub fn restored(k: *const Kernels, s: cuda.Stream, pk: u64, pt: u64, ps: u64, rows: usize) !void {
        const n = restoreGrid(rows);
        if (n > 0) try k.strips(s, pk, pt, ps, 0, rows, n);
    }

    fn strips(k: *const Kernels, s: cuda.Stream, pk: u64, pt: u64, ps: u64, pos: u64, rows: usize, grid: usize) !void {
        if (pt == 0) return error.IndexPruneNotPaged;
        var a: cuda.Args = .{};
        inline for (.{ pk, pt, @as(i64, @intCast(ps)), pos, @as(i64, @intCast(rows)) }) |v| a.add(v);
        try cuda.launch.launch(k.bounds_fn, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = strip_threads } }, s, &a);
    }

    /// Every cache of a state (`st`: cuda_state.State) on striped pages, after a restore to `pos` (`mtp_len` MTP rows).
    pub fn refresh(k: *const Kernels, s: cuda.Stream, st: anytype, pos: usize, mtp_len: usize) !void {
        for (st.caches) |c| if (c.strip) try k.restored(s, c.pk, c.pt, c.ps, pos);
        if (st.mtp_caches.strip) try k.restored(s, st.mtp_caches.pk, st.mtp_caches.pt, st.mtp_caches.ps, mtp_len);
    }

    /// One select block's row scores into out [n][np], pruned: the plan, planned scores cut at the end, planned scores.
    pub fn scores(k: *const Kernels, s: cuda.Stream, x: Select) !void {
        const pages = pagesFor(x.np);
        if (pages > max_pages or x.heads > 32) return error.IndexPruneShape;
        const c = counters(x.plan, x.n, pages);
        var pa: cuda.Args = .{};
        inline for (.{ x.qi, x.wts, @as(i64, @intCast(x.w_stride)), x.pk, x.pt, @as(i64, @intCast(x.ps)), x.at, x.scale, x.wscale, @as(i64, @intCast(x.heads)), x.plan, x.ub, @as(i64, @intCast(pages)), @as(c_int, seeds), c.plan }) |v| pa.add(v);
        try cuda.launch.launch(k.plan_fn, .{ .grid = .{ .x = @intCast(x.n), .y = @intCast(cdiv(pages, plan_pages)) }, .block = .{ .x = threads } }, s, &pa);
        try k.planned(s, x, pages, x.k, c.group); // cut at each row group's end
        try k.planned(s, x, pages, 0, 0);
    }

    /// tf_glm_index_scores_prune over the plan as it stands; `cut` > 0 also cuts (group counters at `group`).
    fn planned(k: *const Kernels, s: cuda.Stream, x: Select, pages: usize, cut: usize, group: u64) !void {
        const loop = loopFor(x.n);
        const stats = if (x.stats != 0) x.stats else k.counts.ptr;
        var a: cuda.Args = .{};
        inline for (.{ x.qi, x.wts, @as(i64, @intCast(x.w_stride)), x.pk, x.out, x.at, @as(i64, @intCast(x.n)), @as(i64, @intCast(x.np)), @as(i64, @intCast(x.nb)), x.scale, x.wscale, @as(i64, @intCast(x.heads)), @as(c_int, @intCast(loop)), x.pt, @as(i64, @intCast(x.ps)), x.plan, @as(i64, @intCast(pages)), @as(c_int, @intCast(cut)), x.ub, group, stats }) |v| a.add(v);
        try cuda.launch.launch(k.scores_fn, .{ .grid = .{ .x = @intCast(cdiv(x.n, 8)), .y = @intCast(@max(1, cdiv(x.nb, loop))) }, .block = .{ .x = threads } }, s, &a);
    }
};

/// The launches' counters after the plan bytes (zeroed once; each launch's last block clears its own).
pub fn counters(plan: u64, n: usize, pages: usize) struct { plan: u64, group: u64 } {
    const at = plan + std.mem.alignForward(usize, n * pages, 16);
    return .{ .plan = at, .group = at + n * 4 };
}

/// One select block's inputs: `n` rows at device `at`, `nb` 64-pool tiles; plan u8 and ub f32 [n][pagesFor(np)].
pub const Select = struct {
    qi: u64,
    wts: u64,
    w_stride: usize,
    pk: u64,
    pt: u64,
    ps: u64,
    out: u64,
    at: u64,
    n: usize,
    np: usize,
    nb: usize,
    scale: f32,
    wscale: f32,
    heads: usize,
    k: usize,
    plan: u64,
    ub: u64,
    stats: u64 = 0, // 0 (the kernels' counts), or three u64 the cut adds to: skipped pages, whole visible pages, rows
};

test "GLM index: TF_GLM_INDEX_PRUNE prunes striped pages past 16,384 tokens, within its plan scratch" {
    const t = std.testing;
    try t.expect(applies(8192, true, 0x40, 0x1000, 557568 * 12));
    try t.expect(!applies(4096, true, 0x40, 0x1000, 557568 * 12)); // a bucket of 16,384 tokens or fewer: plain scan
    try t.expect(!applies(8192, false, 0x40, 0x1000, 557568 * 12)); // pages without strips
    try t.expect(!applies(8192, true, 0, 0x1000, 557568 * 12)); // a window, not pages
    try t.expect(!applies(8192, true, 0x40, 0x1008, 557568 * 12)); // cp.async needs 16-byte keys
    try t.expect(applies(max_pages * page_pools, true, 0x40, 0x1000, 4096));
    try t.expect(!applies(max_pages * page_pools + 1, true, 0x40, 0x1000, 4096));
    try t.expect(fits(262144) and fits(524288) and !fits(524288 + 2048));
    // the plan's scratch, a row of each: u8 and f32 a page, within sparse.select_list's 4,096 bytes
    try t.expect(max_pages * 4 <= @import("cuda_sparse.zig").select_list * 4);
    try t.expectEqual(@as(usize, 16), loopFor(512));
    try t.expectEqual(@as(usize, 1), loopFor(8));
    // a write of 3,000 rows at any position completes at most 3000 / 512 + 1 pages after the first it touches
    for ([_]usize{ 0, 1, 511, 512, 2047, 100_003 }) |pos| for ([_]usize{ 1, 3, 512, 513, 2048, 3000 }) |rows| {
        const first = pos / 4 / page_pools;
        const last = (pos + rows) / 4 / page_pools; // pages below this are whole
        try t.expect(last -| first <= writeGrid(rows));
    };
    try t.expectEqual(@as(usize, 0), restoreGrid(511));
    try t.expectEqual(@as(usize, 400), restoreGrid(204800));
    try t.expectEqual(@as(usize, 1024 + 512), pagesFor(196_608));
}

test "GLM index: TF_GLM_INDEX_PRUNE_MIN_PAGES gates the prune by the select's whole pages; counters fit the scratch" {
    const t = std.testing;
    try t.expectEqual(@as(usize, default_min_pages), try parseMinPages(null));
    try t.expectEqual(@as(usize, 0), try parseMinPages("0"));
    try t.expectEqual(@as(usize, 1024), try parseMinPages("1024"));
    try t.expectError(error.BadIndexPruneMinPages, parseMinPages("1025"));
    try t.expectError(error.BadIndexPruneMinPages, parseMinPages("64k"));
    var k: Kernels = undefined;
    k.min_pages = 256;
    try t.expect(!k.takes(65536, true, 0x40, 0x1000, 4096, 256 * 512 - 1)); // 255 whole pages
    try t.expect(k.takes(65536, true, 0x40, 0x1000, 4096, 256 * 512));
    try t.expect(!k.takes(65536, false, 0x40, 0x1000, 4096, 1 << 20)); // not striped
    k.min_pages = 0;
    try t.expect(!k.takes(8192, true, 0x40, 0x1000, 4096, seeds * 512)); // no page past the seeds: nothing to skip
    try t.expect(k.takes(8192, true, 0x40, 0x1000, 4096, (seeds + 1) * 512));
    // the plan bytes, then a counter a row and one an 8-row group, within a row's 4,096 bytes of scratch a row
    for ([_]usize{ 1, 7, 8, 9, 512 }) |n| {
        const c = counters(0, n, max_pages);
        try t.expect(c.plan % 16 == 0 and c.group == c.plan + n * 4);
        try t.expect(c.group + (n + 7) / 8 * 4 <= n * @import("cuda_sparse.zig").select_list * 4);
    }
}

test "GLM index: TF_GLM_INDEX_PRUNE is refused by name without its kernels, without pages or past its plan" {
    const t = std.testing;
    const S = struct { index_prune: bool = false, pages: bool = false };
    try admit(S{}, false, 262144); // off: nothing asked
    try t.expectError(error.IndexPruneSettings, admit(S{}, true, 262144));
    try t.expectError(error.IndexPruneSettings, admit(S{ .index_prune = true, .pages = true }, false, 262144));
    try t.expectError(error.IndexPruneNeedsPages, admit(S{ .index_prune = true }, true, 262144));
    try t.expectError(error.IndexPruneCapacity, admit(S{ .index_prune = true, .pages = true }, true, 1 << 20));
    try admit(S{ .index_prune = true, .pages = true }, true, 262144);
}
