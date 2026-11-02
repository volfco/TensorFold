//! glm-index-prune (GPU): TF_GLM_INDEX_PRUNE's pruned index scans against full ones, byte for byte (RUNBOOK-prune.md).
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm = @import("glm_cuda");
const sp = glm.sparse;
const pr = glm.index_prune;
const ref = glm.index_prune_ref;

const d = 128;
const heads = 32;
const cap = 262144 / 4; // a 262,144-token context's pools
const page_elems = ref.page_pools * d + 2 * d + 512; // keys, strip, other rows' bytes
const page_bytes = page_elems * 2; // 34,816: whole 1,024-byte latent rows (the token lists' unit)
const lengths = [_]usize{ 32768, 131072, 204800 };

const Keys = enum { clustered, normal, ties, equal, nan };
const Query = enum { topical, normal };
const Case = struct { keys: Keys, query: Query = .topical, negative: bool = true, name: []const u8 };
const cases = [_]Case{
    .{ .keys = .clustered, .name = "clustered keys, topical rows, negative weights" },
    .{ .keys = .clustered, .negative = false, .name = "clustered keys, positive weights" },
    .{ .keys = .clustered, .query = .normal, .name = "clustered keys, random rows" },
    .{ .keys = .normal, .query = .normal, .name = "random keys and rows" },
    .{ .keys = .ties, .name = "tied pages" },
    .{ .keys = .equal, .name = "every key equal" },
    .{ .keys = .nan, .name = "a NaN key" },
};

const Bufs = std.ArrayList(cuda.DeviceBuffer);

fn zeros(gpu: check.Gpu, bufs: *Bufs, n: usize) !u64 {
    var b = try cuda.DeviceBuffer.alloc(gpu.d, n);
    errdefer b.free();
    try b.fill8(0, null);
    try gpu.d.synchronize();
    try bufs.append(gpu.gpa, b);
    return b.ptr;
}

fn fromHost(gpu: check.Gpu, bufs: *Bufs, bytes: []const u8) !u64 {
    var b = try cuda.DeviceBuffer.fromHost(gpu.d, bytes);
    errdefer b.free();
    try bufs.append(gpu.gpa, b);
    return b.ptr;
}

fn download(gpu: check.Gpu, ptr: u64, out: []u8) !void {
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = ptr, .len = out.len, .borrowed = true }).download(0, out);
}

/// The keys: `topics` directions, each page's pools near one of them (clustered), or not.
const World = struct {
    store: []u16, // [pages][page_elems] bf16, strips poisoned (NaN) until written
    pt: []u32,
    topics: []f32,
    page_topic: []usize,

    fn init(gpa: std.mem.Allocator, r: std.Random, tokens: usize, keys: Keys) !World {
        const n = (tokens + 4096) / 512 + 2;
        const w: World = .{ .store = try gpa.alloc(u16, n * page_elems), .pt = try gpa.alloc(u32, n), .topics = try gpa.alloc(f32, 16 * d), .page_topic = try gpa.alloc(usize, n) };
        @memset(w.store, ref.toBf16(std.math.nan(f32)));
        for (w.pt, 0..) |*v, i| v.* = @intCast(i);
        r.shuffle(u32, w.pt);
        for (w.topics) |*v| v.* = r.floatNorm(f32);
        for (w.page_topic) |*v| v.* = r.uintLessThan(usize, 16);
        const p = w.view();
        for (0..n * ref.page_pools) |pool| {
            const g = pool / ref.page_pools;
            const key = p.store[p.pt[g] * page_elems + (pool % ref.page_pools) * d ..][0..d];
            for (key, 0..) |*v, c| v.* = ref.toBf16(switch (keys) {
                .clustered, .nan => 2 * w.topics[w.page_topic[g] * d + c] + 0.5 * r.floatNorm(f32),
                .normal => r.floatNorm(f32),
                .ties => if (g % 3 == 0) 0.5 else 2 * w.topics[w.page_topic[g] * d + c] + 0.5 * r.floatNorm(f32),
                .equal => 0.75,
            });
        }
        if (keys == .nan) p.store[p.pt[7] * page_elems + 9 * d + 3] = ref.toBf16(std.math.nan(f32));
        return w;
    }

    fn deinit(w: World, gpa: std.mem.Allocator) void {
        gpa.free(w.store);
        gpa.free(w.pt);
        gpa.free(w.topics);
        gpa.free(w.page_topic);
    }

    fn view(w: World) ref.Pages {
        return .{ .store = w.store, .pt = w.pt, .page_elems = page_elems };
    }

    /// `rows` query rows of `heads` heads (topical: each row near one topic) and their head weights.
    fn queries(w: World, gpa: std.mem.Allocator, r: std.Random, rows: usize, c: Case) !struct { q: []u16, wts: []u16 } {
        const q = try gpa.alloc(u16, rows * heads * d);
        const wts = try gpa.alloc(u16, rows * heads);
        for (0..rows) |row| {
            const topic = r.uintLessThan(usize, 16);
            for (0..heads) |h| for (0..d) |x| {
                const base: f32 = if (c.query == .topical) 0.5 * w.topics[topic * d + x] else 0;
                q[(row * heads + h) * d + x] = ref.toBf16(base + r.floatNorm(f32));
            };
            for (0..heads) |h| wts[row * heads + h] = ref.toBf16(if (c.negative) r.floatNorm(f32) else @abs(r.floatNorm(f32)));
        }
        return .{ .q = q, .wts = wts };
    }
};

fn scratch(gpu: check.Gpu, bufs: *Bufs, rows: usize, np: usize) !sp.Scratch {
    const blk: usize = @min(rows, sp.select_rows);
    return .{ .scores = try zeros(gpu, bufs, blk * np * 4), .ls = try zeros(gpu, bufs, rows * sp.select_list * 4), .li = try zeros(gpu, bufs, rows * sp.select_list * 4), .pools = try zeros(gpu, bufs, rows * sp.topk_pools * 8), .at = try zeros(gpu, bufs, 4), .prune_stats = try zeros(gpu, bufs, 32) };
}

/// One scan's outputs, read back.
const Out = struct { pools: []u8, tokens: []u8, counts: []u8 };

/// The scans compared: the glue's, or with TF_GLM_INDEX_CUDA=1's kernels (`quick`), each pruned or not.
const Way = enum { plain, pruned, quick, quick_pruned };

const Env = struct { gpu: check.Gpu, s: cuda.Stream, ops: cuda.Module, sets: [4]glm.glue.Set, k: *const pr.Kernels };

/// One selectTokens call's device side: `rows` query rows at `pos` over the uploaded pages, its scratch made once.
const Scan = struct {
    e: Env,
    dk: u64,
    table: glm.latent.Table,
    q: u64,
    wts: u64,
    pos: usize,
    rows: usize,
    s: sp.Scratch,
    dpos: u64,
    tok: u64,
    cnt: u64,

    fn init(e: Env, bufs: *Bufs, dk: u64, table: glm.latent.Table, q: u64, wts: u64, pos: usize, rows: usize) !Scan {
        const p0: i32 = @intCast(pos);
        return .{ .e = e, .dk = dk, .table = table, .q = q, .wts = wts, .pos = pos, .rows = rows, .s = try scratch(e.gpu, bufs, rows, sp.poolBucket(pos, rows, cap)), .dpos = try fromHost(e.gpu, bufs, std.mem.asBytes(&p0)), .tok = try zeros(e.gpu, bufs, rows * sp.width * 4), .cnt = try zeros(e.gpu, bufs, rows * 4) };
    }

    /// selectTokens one way, to completion; `events`: recorded on the stream around its launches (GPU time).
    fn go(x: Scan, way: Way) !void {
        return x.goTimed(way, null);
    }

    fn goTimed(x: Scan, way: Way, events: ?[2]cuda.Event) !void {
        var t = x.table;
        t.strip = way == .pruned or way == .quick_pruned;
        if (events) |ev| try ev[0].record(x.e.s);
        try sp.selectTokens(x.e.sets[@intFromEnum(way)], x.e.ops, x.e.s, x.q, x.wts, heads, x.dk, t, x.dpos, x.pos, x.rows, heads, d, cap, x.s, x.tok, x.cnt);
        if (events) |ev| try ev[1].record(x.e.s);
        try x.e.s.synchronize();
    }

    /// Its outputs read back, and the skipped pages, whole pages and rows the cut has counted since the last read.
    fn read(x: Scan, stats: *[3]u64) !Out {
        const gpa = x.e.gpu.gpa;
        const out: Out = .{ .pools = try gpa.alloc(u8, x.rows * sp.topk_pools * 8), .tokens = try gpa.alloc(u8, x.rows * sp.width * 4), .counts = try gpa.alloc(u8, x.rows * 4) };
        try download(x.e.gpu, x.s.pools, out.pools);
        try download(x.e.gpu, x.tok, out.tokens);
        try download(x.e.gpu, x.cnt, out.counts);
        var st: [4]u64 = undefined;
        try download(x.e.gpu, x.s.prune_stats, std.mem.sliceAsBytes(&st));
        stats.* = st[0..3].*;
        try (cuda.DeviceBuffer{ .d = x.e.gpu.d, .ptr = x.s.prune_stats, .len = 32, .borrowed = true }).fill8(0, null);
        try x.e.gpu.d.synchronize();
        return out;
    }
};

/// selectTokens over `rows` rows at `pos` one way; its outputs and (pruned) the skipped and whole pages.
fn scan(e: Env, bufs: *Bufs, dk: u64, table: glm.latent.Table, q: u64, wts: u64, pos: usize, rows: usize, way: Way, stats: *[3]u64) !Out {
    const x = try Scan.init(e, bufs, dk, table, q, wts, pos, rows);
    try x.go(way);
    return x.read(stats);
}

fn sameOut(a: Out, b: Out) bool {
    return std.mem.eql(u8, a.pools, b.pools) and std.mem.eql(u8, a.tokens, b.tokens) and std.mem.eql(u8, a.counts, b.counts);
}

fn freeOut(gpa: std.mem.Allocator, o: Out) void {
    gpa.free(o.pools);
    gpa.free(o.tokens);
    gpa.free(o.counts);
}

/// The pages on the device, strips written as the engine writes them and checked; `stale`: then zeroed (the control).
fn upload(e: Env, bufs: *Bufs, w: World, tokens: usize, r: std.Random, stale: bool) !struct { dk: u64, table: glm.latent.Table } {
    const dk = try fromHost(e.gpu, bufs, std.mem.sliceAsBytes(w.store));
    const pt = try fromHost(e.gpu, bufs, std.mem.sliceAsBytes(w.pt));
    var at: usize = 0;
    while (at < tokens) {
        const n = @min(tokens - at, if (tokens - at > 64) @as(usize, 2048) else 1 + r.uintLessThan(usize, 8));
        const p0: i32 = @intCast(at);
        try e.k.bounds(e.s, dk, pt, page_bytes, try fromHost(e.gpu, bufs, std.mem.asBytes(&p0)), n);
        at += n;
    }
    try e.s.synchronize();
    const host = try e.gpu.gpa.alloc(u16, w.store.len);
    defer e.gpu.gpa.free(host);
    try download(e.gpu, dk, std.mem.sliceAsBytes(host));
    const want = try e.gpu.gpa.dupe(u16, w.store);
    defer e.gpu.gpa.free(want);
    const p: ref.Pages = .{ .store = want, .pt = w.pt, .page_elems = page_elems };
    p.writeStrips(0, tokens);
    var bad: usize = 0;
    for (0..tokens / 4 / ref.page_pools) |g| {
        const got = host[w.pt[g] * page_elems + ref.page_pools * d ..][0 .. 2 * d];
        for (got, p.strip(g)) |a, b| bad += @intFromBool(a != b and !(std.math.isNan(ref.fromBf16(a)) and std.math.isNan(ref.fromBf16(b))));
    }
    try check.expect(bad == 0, "glm-index-prune: {d} strip values differ from the host's min and max at {d} tokens", .{ bad, tokens });
    if (stale) for (0..w.pt.len) |g| {
        const zero: [2 * d]u16 = @splat(0);
        try (cuda.DeviceBuffer{ .d = e.gpu.d, .ptr = dk + (w.pt[g] * page_elems + ref.page_pools * d) * 2, .len = 4 * d, .borrowed = true }).upload(0, std.mem.sliceAsBytes(&zero));
    };
    return .{ .dk = dk, .table = .{ .pt = pt, .ps = page_bytes, .page_rows = page_bytes / 1024 } };
}

fn ms(a: std.Io.Timestamp, b: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(a.durationTo(b).nanoseconds)) / 1e6;
}

pub fn run(gpu: check.Gpu, timing: bool) !void {
    const gpa = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var glue_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer glue_mod.unload();
    var ops = try cuda.Module.load(gpu.d, cuda.kernels.glm_ops);
    defer ops.unload();
    const ks = try glm.glue.Kernels.load(glue_mod);
    var k = try pr.Kernels.load(gpu.d, cuda.kernels.glm_index);
    defer k.unload();
    k.min_pages = 0; // every length prunes here; the served default is TF_GLM_INDEX_PRUNE_MIN_PAGES's
    std.debug.print("glm-index-prune: TF_GLM_INDEX_PRUNE_MIN_PAGES default {d} pages; forced to 0 for this check\n", .{pr.default_min_pages});
    var idx = try glm.index.Kernels.load(gpu.d, cuda.kernels.glm_index, .both); // TF_GLM_INDEX_CUDA=1's
    defer idx.unload();
    const sets = [4]glm.glue.Set{ .{ .glue = &ks }, .{ .glue = &ks, .prune = &k }, .{ .glue = &ks, .index = &idx }, .{ .glue = &ks, .index = &idx, .prune = &k } };
    const e: Env = .{ .gpu = gpu, .s = s, .ops = ops, .sets = sets, .k = &k };
    var rng = std.Random.DefaultPrng.init(0x9e1d);
    const r = rng.random();
    var fails: usize = 0;
    for (lengths) |tokens| for (cases) |c| {
        if (timing and c.keys != .clustered) continue;
        var bufs: Bufs = .empty;
        defer {
            for (bufs.items) |*b| b.free();
            bufs.deinit(gpa);
        }
        const w = try World.init(gpa, r, tokens, c.keys);
        defer w.deinit(gpa);
        const dev = try upload(e, &bufs, w, tokens, r, false);
        // a prompt block of 512 rows, a decode window of 8 and a decode row, each ending at the last token
        for ([_]usize{ 512, 8, 1 }) |rows| {
            const qs = try w.queries(gpa, r, rows, c);
            defer gpa.free(qs.q);
            defer gpa.free(qs.wts);
            const q = try fromHost(gpu, &bufs, std.mem.sliceAsBytes(qs.q));
            const wt = try fromHost(gpu, &bufs, std.mem.sliceAsBytes(qs.wts));
            const pos = tokens - rows;
            var stats: [3]u64 = undefined;
            var none: [3]u64 = undefined;
            const a = try scan(e, &bufs, dev.dk, dev.table, q, wt, pos, rows, .plain, &none);
            defer freeOut(gpa, a);
            const b = try scan(e, &bufs, dev.dk, dev.table, q, wt, pos, rows, .pruned, &stats);
            defer freeOut(gpa, b);
            var same = sameOut(a, b);
            if (rows == 512) { // prompt chunks under TF_GLM_INDEX_CUDA=1: tf_glm_index_pools over the pruned rows
                const qa = try scan(e, &bufs, dev.dk, dev.table, q, wt, pos, rows, .quick, &none);
                defer freeOut(gpa, qa);
                const qb = try scan(e, &bufs, dev.dk, dev.table, q, wt, pos, rows, .quick_pruned, &none);
                defer freeOut(gpa, qb);
                same = same and sameOut(qa, qb) and sameOut(a, qa);
            }
            fails += @intFromBool(!same);
            const share = if (stats[1] > 0) @as(f64, @floatFromInt(stats[0])) / @as(f64, @floatFromInt(stats[1])) else 0;
            std.debug.print("{s} {d} tokens, {d} rows, {s}: pages skipped {d} of {d} ({d:.3})\n", .{ if (same) "PASS" else "FAIL", tokens, rows, c.name, stats[0], stats[1], share });
            if (timing) try time(try Scan.init(e, &bufs, dev.dk, dev.table, q, wt, pos, rows), tokens);
        }
    };
    if (!timing) fails += try sequence(e, r);
    if (!timing) fails += try control(e, r);
    try check.expect(fails == 0, "glm-index-prune: {d} comparisons failed", .{fails});
    check.pass("glm-index-prune: pruned selections equal the full scan's at {d} lengths, {d} cases, {d} engine shapes on one scratch, the control caught", .{ lengths.len, cases.len, steps.len });
}

/// Microseconds a query row of selectTokens, plain and pruned: wall clock and CUDA events, medians of five runs.
fn time(x: Scan, tokens: usize) !void {
    var ev = [2]cuda.Event{ try cuda.Event.init(x.e.gpu.d, true), try cuda.Event.init(x.e.gpu.d, true) };
    defer for (&ev) |*v| v.deinit();
    var wall: [2][5]f64 = undefined;
    var gpu: [2][5]f64 = undefined;
    var counts: [3]u64 = undefined;
    freeOut(x.e.gpu.gpa, try x.read(&counts)); // clear the counts: the pruned runs' share below is theirs
    for (0..6) |rep| for (0..2) |which| {
        const t0 = std.Io.Timestamp.now(x.e.gpu.io, .awake);
        try x.goTimed(if (which == 1) .pruned else .plain, ev);
        const t1 = std.Io.Timestamp.now(x.e.gpu.io, .awake);
        const per = 1000 / @as(f64, @floatFromInt(x.rows));
        if (rep > 0) wall[which][rep - 1] = ms(t0, t1) * per;
        if (rep > 0) gpu[which][rep - 1] = @as(f64, try cuda.Event.elapsedMs(ev[0], ev[1])) * per;
    };
    for (&wall) |*v| std.mem.sort(f64, v, {}, std.sort.asc(f64));
    for (&gpu) |*v| std.mem.sort(f64, v, {}, std.sort.asc(f64));
    freeOut(x.e.gpu.gpa, try x.read(&counts));
    const share = if (counts[1] > 0) @as(f64, @floatFromInt(counts[0])) / @as(f64, @floatFromInt(counts[1])) else 0;
    std.debug.print("TIME {d} tokens, {d} rows: wall full {d:.1} pruned {d:.1} us a row (x{d:.2}); GPU full {d:.1} pruned {d:.1} us a row (x{d:.2}); pages skipped {d:.3}\n", .{ tokens, x.rows, wall[0][2], wall[1][2], wall[0][2] / wall[1][2], gpu[0][2], gpu[1][2], gpu[0][2] / gpu[1][2], share });
}

/// The engine's shapes in turn on one scratch each way, as a forward's buffers serve every select (STATUS-prune).
const Step = struct { pos: usize, rows: usize, what: []const u8 };
const steps = [_]Step{
    .{ .pos = 28672, .rows = 2048, .what = "a 2048-row prompt chunk (4 blocks of 512)" },
    .{ .pos = 30720, .rows = 493, .what = "a 31,213-token pass's last chunk" },
    .{ .pos = 30720, .rows = 492, .what = "its MTP rows (one fewer)" },
    .{ .pos = 30720, .rows = 1024, .what = "a 1024-row chunk" },
    .{ .pos = 131071, .rows = 1, .what = "a decode row at 131,072 tokens" },
    .{ .pos = 31213, .rows = 1, .what = "a decode row of a shorter request" },
    .{ .pos = 31206, .rows = 8, .what = "a decode window of 8" },
};

fn sequence(e: Env, r: std.Random) !usize {
    const gpa = e.gpu.gpa;
    var bufs: Bufs = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpa);
    }
    const c = cases[0];
    const tokens = 131072;
    const w = try World.init(gpa, r, tokens, c.keys);
    defer w.deinit(gpa);
    const dev = try upload(e, &bufs, w, tokens, r, false);
    const ss = [2]sp.Scratch{ try scratch(e.gpu, &bufs, 2048, cap), try scratch(e.gpu, &bufs, 2048, cap) }; // plain, pruned
    var fails: usize = 0;
    for (steps) |st| {
        const qs = try w.queries(gpa, r, st.rows, c);
        defer gpa.free(qs.q);
        defer gpa.free(qs.wts);
        const q = try fromHost(e.gpu, &bufs, std.mem.sliceAsBytes(qs.q));
        const wt = try fromHost(e.gpu, &bufs, std.mem.sliceAsBytes(qs.wts));
        const p0: i32 = @intCast(st.pos);
        var outs: [2]Out = undefined;
        var stats: [2][3]u64 = undefined;
        for (0..2) |i| {
            const x: Scan = .{ .e = e, .dk = dev.dk, .table = dev.table, .q = q, .wts = wt, .pos = st.pos, .rows = st.rows, .s = ss[i], .dpos = try fromHost(e.gpu, &bufs, std.mem.asBytes(&p0)), .tok = try zeros(e.gpu, &bufs, st.rows * sp.width * 4), .cnt = try zeros(e.gpu, &bufs, st.rows * 4) };
            try x.go(if (i == 0) .plain else .pruned);
            outs[i] = try x.read(&stats[i]);
        }
        defer for (outs) |o| freeOut(gpa, o);
        const ok = sameOut(outs[0], outs[1]) and stats[1][2] == st.rows; // equal, and every row planned and cut once
        fails += @intFromBool(!ok);
        std.debug.print("{s} one scratch, {s}: {d} rows at {d}: pages skipped {d} of {d}, rows cut {d}\n", .{ if (ok) "PASS" else "FAIL", st.what, st.rows, st.pos, stats[1][0], stats[1][1], stats[1][2] });
    }
    return fails;
}

/// The planted control: zeroed strips (bounds near 0, scores positive) must change the selection; 1 if they did not.
fn control(e: Env, r: std.Random) !usize {
    const gpa = e.gpu.gpa;
    var bufs: Bufs = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpa);
    }
    const c: Case = .{ .keys = .clustered, .negative = false, .name = "control" };
    const tokens = lengths[0];
    const w = try World.init(gpa, r, tokens, c.keys);
    defer w.deinit(gpa);
    const dev = try upload(e, &bufs, w, tokens, r, true);
    const qs = try w.queries(gpa, r, 8, c);
    defer gpa.free(qs.q);
    defer gpa.free(qs.wts);
    const q = try fromHost(e.gpu, &bufs, std.mem.sliceAsBytes(qs.q));
    const wt = try fromHost(e.gpu, &bufs, std.mem.sliceAsBytes(qs.wts));
    var st: [3]u64 = undefined;
    const a = try scan(e, &bufs, dev.dk, dev.table, q, wt, tokens - 8, 8, .plain, &st);
    defer freeOut(gpa, a);
    const b = try scan(e, &bufs, dev.dk, dev.table, q, wt, tokens - 8, 8, .pruned, &st);
    defer freeOut(gpa, b);
    const caught = !std.mem.eql(u8, a.pools, b.pools);
    std.debug.print("{s} control: strips left unwritten {s} the selection ({d} of {d} pages skipped)\n", .{ if (caught) "PASS" else "FAIL", if (caught) "change" else "do not change", st[0], st[1] });
    return @intFromBool(!caught);
}
