//! kv-codec-bench: what the lossless kept-state codec costs and saves (docs/kv-cache-compression-plan.md Phase 0 item 5
//! and Phase 1). For each stream (synthetic byte classes, or a TF_GLM_KV_DUMP capture): memcpy as the floor, deflate
//! levels 1 and 6 on the same blocks (the plan's "obvious alternative"), then the codec's plain, atlas and
//! atlas-exhaustive candidate sets at 1 and N threads. Every round trip is compared byte for byte before any time is
//! reported; each figure is the median of `reps` timed runs after one discarded warmup.
//!
//!   zig build kv-codec-bench -- [--mib N] [--threads N] [--reps N] [--file DUMP] [--state-mib N,...]
const std = @import("std");
const kv = @import("kv");
const codec = kv.codec;
const flate = std.compress.flate;
const ch = @import("kv_codec_bench_ch.zig");

const usage =
    \\usage: tf-kv-codec-bench [--mib N] [--threads N] [--reps N] [--file DUMP] [--state-mib A,B,...]
    \\  --mib        bytes a synthetic stream (default 64)
    \\  --threads    the parallel leg's threads (default: up to 8 CPUs)
    \\  --reps       timed runs after one warmup (default 3)
    \\  --file       a TF_GLM_KV_DUMP .bin (its .parts beside it: "class length [row-bytes]" lines) instead of synthetic data
    \\  --width      the --file stream's value width in bytes for the ClickHouse legs: 1, 2 or 4 (default 2)
    \\  --state-mib  kept-state sizes to report the implied keep/restore milliseconds for (default 256,1024,4096)
    \\  --suite      all (default), codec (memcpy, deflate, plain/atlas) or clickhouse (memcpy, plain, ClickHouse codecs)
    \\ ClickHouse's LZ4, LZ4HC and ZSTD legs need a build with -Dkv-bench-libs=true.
;

const Config = struct {
    mib: usize = 64,
    threads: usize = 0,
    reps: usize = 3,
    file: ?[]const u8 = null,
    states: []const usize = &.{ 256, 1024, 4096 },
    suite: enum { all, codec, clickhouse } = .all,
    width: u8 = 2,
};

fn parse(a: std.mem.Allocator, argv: []const [:0]const u8) !Config {
    var c: Config = .{};
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const k = argv[i];
        if (i + 1 >= argv.len) return error.Usage;
        const v = argv[i + 1];
        i += 1;
        if (std.mem.eql(u8, k, "--mib")) c.mib = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--threads")) c.threads = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--reps")) c.reps = @max(1, try std.fmt.parseInt(usize, v, 10)) else if (std.mem.eql(u8, k, "--file")) c.file = v else if (std.mem.eql(u8, k, "--state-mib")) {
            var list: std.ArrayList(usize) = .empty;
            var it = std.mem.tokenizeScalar(u8, v, ',');
            while (it.next()) |x| try list.append(a, try std.fmt.parseInt(usize, x, 10));
            c.states = list.items;
        } else if (std.mem.eql(u8, k, "--width")) {
            c.width = try std.fmt.parseInt(u8, v, 10);
            if (c.width != 1 and c.width != 2 and c.width != 4) return error.Usage;
        } else if (std.mem.eql(u8, k, "--suite")) c.suite = std.meta.stringToEnum(@TypeOf(c.suite), v) orelse return error.Usage else return error.Usage;
    }
    if (c.threads == 0) c.threads = @min(8, std.Thread.getCpuCount() catch 1);
    return c;
}

const Stream = struct { name: []const u8, raw: []u8, spans: []const codec.Span, width: u8 = 2 };

/// The median of `reps` timings of `run` after one warmup, in nanoseconds.
fn time(io: std.Io, reps: usize, ctx: anytype, comptime run: fn (@TypeOf(ctx)) anyerror!void) !u64 {
    var ns: [64]u64 = undefined;
    const n = @min(reps, ns.len);
    try run(ctx); // warmup, discarded
    for (ns[0..n]) |*x| {
        const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
        try run(ctx);
        x.* = @intCast(std.Io.Clock.awake.now(io).toNanoseconds() - t0);
    }
    std.mem.sort(u64, ns[0..n], {}, std.sort.asc(u64));
    return ns[n / 2];
}

fn gbs(bytes: usize, ns: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(@max(ns, 1)));
}

/// A stream cut into blocks, coded into per-block slots (each payload at its block's raw offset: no compaction pass).
const Coded = struct {
    s: *const Stream,
    cuts: []codec.Cut,
    blocks: []codec.Block,
    slots: []u8,
    back: []u8,
    scratch: []codec.Scratch,
    threads: usize,
    o: codec.Options,
    failed: ?anyerror = null,

    fn encodeOne(c: *Coded, i: usize, worker: usize) void {
        const x = c.cuts[i];
        const r = codec.encodeBlock(&c.scratch[worker], c.s.raw[x.off..][0..x.len], x.class, x.row, c.o);
        c.blocks[i] = r.block;
        @memcpy(c.slots[x.off..][0..r.payload.len], r.payload);
    }

    fn decodeOne(c: *Coded, i: usize, worker: usize) void {
        const x = c.cuts[i];
        const b = c.blocks[i];
        codec.decodeBlock(&c.scratch[worker], b, c.slots[x.off..][0..b.len], c.back[x.off..][0..x.len]) catch |err| {
            c.failed = err;
        };
    }

    fn encode(c: *Coded) anyerror!void {
        codec.each(c.threads, c.cuts.len, c, encodeOne);
    }

    fn decode(c: *Coded) anyerror!void {
        codec.each(c.threads, c.cuts.len, c, decodeOne);
        if (c.failed) |err| return err;
    }

    fn stored(c: *const Coded) usize {
        var n: usize = 0;
        for (c.blocks) |b| n += b.len;
        return n;
    }
};

const Memcpy = struct {
    src: []const u8,
    dst: []u8,
    fn run(m: *Memcpy) anyerror!void {
        @memcpy(m.dst, m.src);
        std.mem.doNotOptimizeAway(m.dst.ptr);
    }
};

/// Deflate (std.compress.flate, raw container) over the same blocks, one thread: the general-purpose baseline.
const Deflate = struct {
    s: *const Stream,
    cuts: []const codec.Cut,
    level: flate.Compress.Options,
    window: []u8,
    out: []u8, // one block's worst case
    slots: []u8,
    lens: []usize, // 0: stored raw
    back: []u8,
    c: *flate.Compress,

    /// One block deflated into `out`, or null when it does not fit or does not pay (the block is then stored raw).
    fn one(d: *Deflate, raw: []const u8) ?usize {
        var w: std.Io.Writer = .fixed(d.out);
        d.c.* = flate.Compress.init(&w, d.window, .raw, d.level) catch return null;
        d.c.writer.writeAll(raw) catch return null;
        d.c.finish() catch return null;
        return if (w.end < raw.len) w.end else null;
    }

    fn encode(d: *Deflate) anyerror!void {
        for (d.cuts, d.lens) |x, *len| {
            const raw = d.s.raw[x.off..][0..x.len];
            if (d.one(raw)) |n| {
                len.* = n;
                @memcpy(d.slots[x.off..][0..n], d.out[0..n]);
            } else {
                len.* = 0;
                @memcpy(d.slots[x.off..][0..x.len], raw);
            }
        }
    }

    fn decode(d: *Deflate) anyerror!void {
        for (d.cuts, d.lens) |x, len| {
            const out = d.back[x.off..][0..x.len];
            if (len == 0) {
                @memcpy(out, d.slots[x.off..][0..x.len]);
                continue;
            }
            var r: std.Io.Reader = .fixed(d.slots[x.off..][0..len]);
            var z: flate.Decompress = .init(&r, .raw, d.window);
            try z.reader.readSliceAll(out);
        }
    }

    fn stored(d: *const Deflate) usize {
        var n: usize = 0;
        for (d.cuts, d.lens) |x, len| n += if (len == 0) x.len else len;
        return n;
    }
};

fn report(name: []const u8, codec_name: []const u8, threads: usize, raw: usize, stored: usize, enc: u64, dec: u64, states: []const usize) void {
    const ratio = @as(f64, @floatFromInt(stored)) / @as(f64, @floatFromInt(raw));
    std.debug.print("KVBENCH stream={s} codec={s} threads={d} raw={d} stored={d} ratio={d:.4} encode_gbs={d:.3} decode_gbs={d:.3}", .{ name, codec_name, threads, raw, stored, ratio, gbs(raw, enc), gbs(raw, dec) });
    for (states) |mib| {
        const b: f64 = @floatFromInt(mib << 20);
        std.debug.print(" keep{d}M_ms={d:.1} restore{d}M_ms={d:.1}", .{ mib, b / gbs(raw, enc) / 1e6, mib, b / gbs(raw, dec) / 1e6 });
    }
    std.debug.print("\n", .{});
}

fn engagement(name: []const u8, codec_name: []const u8, blocks: []const codec.Block) void {
    var by: [3][4]usize = @splat(@splat(0));
    for (blocks) |b| by[@min(b.kind, 2)][@min(b.xform, 3)] += 1;
    std.debug.print("KVBENCH stream={s} codec={s} engagement raw={d} planes={d} planes_xor={d} planes_delta={d} planes_delta2={d} t64={d} t64_xor={d} t64_delta={d} t64_delta2={d}\n", .{
        name, codec_name, by[0][0], by[1][0], by[1][1], by[1][2], by[1][3], by[2][0], by[2][1], by[2][2], by[2][3],
    });
}

fn bench(gpa: std.mem.Allocator, io: std.Io, cfg: Config, s: *const Stream) !void {
    const back = try gpa.alloc(u8, s.raw.len);
    defer gpa.free(back);
    var m: Memcpy = .{ .src = s.raw, .dst = back };
    const copy = try time(io, cfg.reps, &m, Memcpy.run);
    report(s.name, "memcpy", 1, s.raw.len, s.raw.len, copy, copy, cfg.states);

    const cuts = try gpa.alloc(codec.Cut, codec.cutCount(s.spans));
    defer gpa.free(cuts);
    codec.cut(s.spans, cuts);
    const slots = try gpa.alloc(u8, s.raw.len);
    defer gpa.free(slots);

    if (cfg.suite != .clickhouse) {
        const window = try gpa.alloc(u8, flate.max_window_len);
        defer gpa.free(window);
        const out = try gpa.alloc(u8, codec.block_size + (codec.block_size >> 4) + 1024);
        defer gpa.free(out);
        const lens = try gpa.alloc(usize, cuts.len);
        defer gpa.free(lens);
        const c = try gpa.create(flate.Compress);
        defer gpa.destroy(c);
        for ([_]struct { []const u8, flate.Compress.Options }{ .{ "deflate1", .level_1 }, .{ "deflate6", .level_6 } }) |lv| {
            var d: Deflate = .{ .s = s, .cuts = cuts, .level = lv[1], .window = window, .out = out, .slots = slots, .lens = lens, .back = back, .c = c };
            try d.encode();
            @memset(back, 0);
            try d.decode();
            if (!std.mem.eql(u8, s.raw, back)) return error.DeflateRoundTrip;
            const enc = try time(io, cfg.reps, &d, Deflate.encode);
            const dec = try time(io, cfg.reps, &d, Deflate.decode);
            report(s.name, lv[0], 1, s.raw.len, d.stored(), enc, dec, cfg.states);
        }
    }

    const scratch = try gpa.alloc(codec.Scratch, cfg.threads);
    defer gpa.free(scratch);
    for (scratch) |*x| x.* = try codec.Scratch.init(gpa);
    defer for (scratch) |*x| x.deinit(gpa);
    const blocks = try gpa.alloc(codec.Block, cuts.len);
    defer gpa.free(blocks);
    for ([_]struct { []const u8, codec.Options }{ .{ "plain", .plain }, .{ "atlas", .atlas }, .{ "atlas-exhaustive", .atlas_exhaustive } }) |o| {
        if (cfg.suite == .clickhouse and o[1].xforms.len > 1) continue; // plain only, as the reference
        for ([_]usize{ 1, cfg.threads }) |threads| {
            var c: Coded = .{ .s = s, .cuts = cuts, .blocks = blocks, .slots = slots, .back = back, .scratch = scratch, .threads = threads, .o = o[1] };
            try c.encode();
            @memset(back, 0);
            try c.decode();
            if (!std.mem.eql(u8, s.raw, back)) return error.CodecRoundTrip; // byte identity before any time
            const enc = try time(io, cfg.reps, &c, Coded.encode);
            const dec = try time(io, cfg.reps, &c, Coded.decode);
            report(s.name, o[0], threads, s.raw.len, c.stored(), enc, dec, cfg.states);
            if (threads == 1) engagement(s.name, o[0], blocks);
            if (cfg.threads == 1) break;
        }
    }

    if (cfg.suite != .codec) try clickhouse(gpa, io, cfg, s, cuts, back);

    var l: kv.ledger.Ledger = .{};
    for (cuts) |x| l.observe(@enumFromInt(@min(x.class, kv.ledger.classes - 1)), s.raw[x.off..][0..x.len]);
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try l.print(&w, s.name);
    std.debug.print("{s}", .{w.buffered()});
}

/// ClickHouse's common codecs over the same blocks, one thread, each round trip byte-identical before timing.
fn clickhouse(gpa: std.mem.Allocator, io: std.Io, cfg: Config, s: *const Stream, cuts: []const codec.Cut, back: []u8) !void {
    const mid = try gpa.alloc(u8, codec.block_size + (64 << 10));
    defer gpa.free(mid);
    const store = try gpa.alloc(u8, 2 * s.raw.len + cuts.len * (64 << 10));
    defer gpa.free(store);
    const lens = try gpa.alloc(usize, cuts.len);
    defer gpa.free(lens);
    const mids = try gpa.alloc(usize, cuts.len);
    defer gpa.free(mids);
    const tables = try gpa.alloc(u32, 2 << 12);
    defer gpa.free(tables);
    const counts = try gpa.alloc(u32, 1 << 16);
    defer gpa.free(counts);
    inline for (ch.pipes) |p| {
        var r: ch.Run = .{ .p = p, .w = s.width, .raw = s.raw, .cuts = cuts, .mid = mid, .store = store, .lens = lens, .mids = mids, .back = back, .tables = tables, .counts = counts };
        try r.encode();
        @memset(back, 0);
        try r.decode();
        if (!std.mem.eql(u8, s.raw, back)) {
            std.debug.print("KVBENCH stream={s} codec={s} ROUND TRIP FAILED\n", .{ s.name, p.name });
            return error.ClickHouseRoundTrip;
        }
        const enc = try time(io, cfg.reps, &r, ch.Run.encode);
        const dec = try time(io, cfg.reps, &r, ch.Run.decode);
        report(s.name, p.name, 1, s.raw.len, r.stored(), enc, dec, cfg.states);
        std.debug.print("KVBENCH stream={s} codec={s} width={d} raw_blocks={d}/{d}\n", .{ s.name, p.name, s.width, r.rawBlocks(), cuts.len });
    }
}

/// A dump's classes from its .parts lines ("class length"), its rows unknown.
fn readDump(a: std.mem.Allocator, io: std.Io, path: []const u8) !Stream {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .unlimited);
    const parts_path = try std.fmt.allocPrint(a, "{s}", .{path});
    const stem = if (std.mem.endsWith(u8, parts_path, ".bin")) parts_path[0 .. parts_path.len - 4] else parts_path;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}.parts", .{stem}), a, .limited(64 << 20));
    var spans: std.ArrayList(codec.Span) = .empty;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var total: usize = 0;
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const class = std.meta.stringToEnum(kv.ledger.Class, f.next() orelse return error.BadParts) orelse return error.BadParts;
        const len = try std.fmt.parseInt(usize, f.next() orelse return error.BadParts, 10);
        const row: u32 = if (f.next()) |r| try std.fmt.parseInt(u32, r, 10) else 0;
        try spans.append(a, .{ .len = len, .class = @intFromEnum(class), .row = row });
        total += len;
    }
    if (total != raw.len) return error.BadParts; // the manifest must cover the dump exactly
    return .{ .name = path, .raw = raw, .spans = spans.items };
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    const cfg = parse(a, argv) catch {
        std.debug.print("{s}", .{usage});
        return error.Usage;
    };
    std.debug.print("KVBENCH config mib={d} threads={d} reps={d} block={d}\n", .{ cfg.mib, cfg.threads, cfg.reps, codec.block_size });
    if (cfg.file) |path| {
        var s = try readDump(a, io, path);
        s.width = cfg.width;
        return bench(init.gpa, io, cfg, &s);
    }
    const raw = try a.alloc(u8, cfg.mib << 20);
    for (std.enums.values(kv.synth.Kind)) |kind| {
        kv.synth.fill(kind, raw, 1);
        // latent-shaped rows (512 BF16 channels) so the column candidates have a token axis
        const row: u32 = switch (kind) {
            .bf16_gauss, .bf16_rows, .state_mix => 1024,
            .e4m3 => 512,
            else => 0,
        };
        const spans = [_]codec.Span{.{ .len = raw.len, .class = @intFromEnum(kv.ledger.Class.latent), .row = row }};
        // the column type ClickHouse would be given: FP32 words, FP8 bytes, else BF16 (16-bit) values
        const width: u8 = switch (kind) {
            .f32_from_bf16, .f32_gauss => 4,
            .e4m3 => 1,
            else => 2,
        };
        const s: Stream = .{ .name = @tagName(kind), .raw = raw, .spans = &spans, .width = width };
        try bench(init.gpa, io, cfg, &s); // the arena keeps only the stream; each bench frees its own buffers
    }
}
