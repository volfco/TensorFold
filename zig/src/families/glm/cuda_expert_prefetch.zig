//! TF_GLM_EXPERT_PREFETCH: after a decode window's top-k pick, a side stream asks L2 for the head of every column-block
//! run of each picked NVFP4 expert (glm_expert_prefetch.cu), so the decode expert kernel's first loads may hit L2.
//! Loads only: the kernel writes nothing the model reads, so no output bit can change. Unbuilt prototype: its image is
//! not in the kernel list; TF_GLM_EXPERT_PREFETCH_IMAGE names a fatbin built by hand (tools/kvprobe/STATUS-track4.md).

const std = @import("std");
const cuda = @import("cuda");

/// nvfp4_experts.cu: a (32 columns, 32 inputs) block is 144 int32, 576 bytes; a k32 group of a run holds M of them.
pub const block_bytes: usize = 144 * 4;
pub const cols: usize = 32;
/// glm_expert_prefetch.cu's block size and its most blocks (one a pick, grid-stride past it).
pub const threads: u32 = 128;
pub const max_grid: usize = 1024;

/// TF_GLM_EXPERT_PREFETCH: unset or 0 (nothing loads or launches), bulk (cp.async.bulk.prefetch.L2, sm_90+), lines
/// (prefetch.global.L2 a 128-byte line). Anything else is refused by name.
pub const Mode = enum {
    off,
    bulk,
    lines,

    pub fn parse(text: ?[]const u8) !Mode {
        const v = text orelse return .off;
        if (v.len == 0 or std.mem.eql(u8, v, "0")) return .off;
        if (std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "bulk")) return .bulk;
        if (std.mem.eql(u8, v, "lines")) return .lines;
        return error.BadExpertPrefetchMode;
    }
};

/// TF_GLM_EXPERT_PREFETCH_GROUPS: k32 groups a run's head holds (1 to 64; default 2, the decode kernel's two stages in
/// flight). TF_GLM_EXPERT_PREFETCH_ROWS: the widest window (rows) that prefetches (1 to 2048; default 64).
pub const Settings = struct {
    mode: Mode = .off,
    groups: usize = 2,
    rows: usize = 64,

    pub fn parse(mode: ?[]const u8, groups: ?[]const u8, rows: ?[]const u8) !Settings {
        var s: Settings = .{ .mode = try Mode.parse(mode) };
        if (groups) |g| s.groups = std.fmt.parseInt(usize, g, 10) catch return error.BadExpertPrefetchGroups;
        if (s.groups < 1 or s.groups > 64) return error.BadExpertPrefetchGroups;
        if (rows) |r| s.rows = std.fmt.parseInt(usize, r, 10) catch return error.BadExpertPrefetchRows;
        if (s.rows < 1 or s.rows > 2048) return error.BadExpertPrefetchRows;
        return s;
    }

    pub fn read() !Settings {
        return parse(env("TF_GLM_EXPERT_PREFETCH"), env("TF_GLM_EXPERT_PREFETCH_GROUPS"), env("TF_GLM_EXPERT_PREFETCH_ROWS"));
    }
};

fn env(name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

/// One matrix's runs as the decode kernel reads them: expert e's column block cb starts at
/// base + (e * nb + cb) * run, and its kernel walks the run from the front, a group of `parts` blocks at a time.
pub const Matrix = struct {
    base: u64,
    nb: usize, // column blocks (outputs / 32)
    run: usize, // bytes of one column block's run (inputs / 32 groups)
    head: usize, // bytes of each run to prefetch

    pub fn stride(m: Matrix) usize {
        return m.nb * m.run;
    }
};

pub const Geometry = struct { up: Matrix, down: Matrix, experts: usize };

/// The NVFP4 layout of cuda.experts.Layer (up [E][NI/32][D/32][2][144], down [E][D/32][NI/32][1][144] int32), the head
/// of each run `groups` k32 groups (at most the run). Refuses a layer the decode kernel would not take.
pub fn geometry(l: anytype, groups: usize) !Geometry {
    if (l.format != .nvfp4) return error.ExpertPrefetchNeedsNvfp4;
    if (l.relu2) return error.ExpertPrefetchNeedsSwiglu; // GLM's experts are SwiGLU gate-up pairs
    if (l.width == 0 or l.dims == 0 or l.width % cols != 0 or l.dims % cols != 0 or l.experts == 0) return error.ExpertPrefetchShape;
    if (l.up % 16 != 0 or l.down % 16 != 0) return error.ExpertPrefetchAlignment; // the bulk form's rule
    const up_group = 2 * block_bytes;
    const down_group = block_bytes;
    const up_run = (l.dims / 32) * up_group;
    const down_run = (l.width / 32) * down_group;
    return .{
        .up = .{ .base = l.up, .nb = l.width / cols, .run = up_run, .head = @min(up_run, groups * up_group) },
        .down = .{ .base = l.down, .nb = l.dims / cols, .run = down_run, .head = @min(down_run, groups * down_group) },
        .experts = l.experts,
    };
}

/// Bytes one pick asks L2 for (both matrices); the kernel never reads past an expert's own runs.
pub fn bytesPerPick(g: Geometry) usize {
    return g.up.nb * g.up.head + g.down.nb * g.down.head;
}

/// A launch over `picks` picks: one block a pick up to max_grid, then grid-stride.
pub fn grid(picks: usize) u32 {
    return @intCast(@max(1, @min(picks, max_grid)));
}

/// The side stream, its fork and join events, and the two kernels of the hand-built image.
pub const Prefetcher = struct {
    settings: Settings,
    module: cuda.Module,
    bulk_fn: cuda.Function,
    lines_fn: cuda.Function,
    side: cuda.Stream,
    fork: cuda.Event,
    joined: cuda.Event,

    /// null when TF_GLM_EXPERT_PREFETCH is off; with it on, a missing or unreadable image is refused by name.
    pub fn fromEnv(d: *const cuda.Driver) !?Prefetcher {
        const gpa = std.heap.page_allocator;
        const s = try Settings.read();
        if (s.mode == .off) return null;
        const path = std.c.getenv("TF_GLM_EXPERT_PREFETCH_IMAGE") orelse return error.ExpertPrefetchNeedsImage;
        const image = try readImageAt(gpa, path);
        defer gpa.free(image);
        return try load(d, image, s);
    }

    pub fn load(d: *const cuda.Driver, image: []const u8, s: Settings) !Prefetcher {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        var side = try cuda.Stream.init(d, true);
        errdefer side.deinit();
        var fork = try cuda.Event.init(d, false);
        errdefer fork.deinit();
        return .{ .settings = s, .module = m, .bulk_fn = try m.function("tf_glm_expert_prefetch_bulk"), .lines_fn = try m.function("tf_glm_expert_prefetch_lines"), .side = side, .fork = fork, .joined = try cuda.Event.init(d, false) };
    }

    pub fn unload(p: *Prefetcher) void {
        p.joined.deinit();
        p.fork.deinit();
        p.side.deinit();
        p.module.unload();
    }

    /// After the pick on `main`: the side stream waits for it, then prefetches; a window wider than the rows setting
    /// is skipped. `picks`: i32 [rows][slots] (the shared slot's id is `experts` and is skipped on the device).
    pub fn issue(p: *const Prefetcher, main: cuda.Stream, picks: u64, rows: usize, slots: usize, l: anytype) !void {
        if (rows > p.settings.rows) return;
        const g = try geometry(l, p.settings.groups);
        try p.fork.record(main);
        try p.side.wait(p.fork);
        var a: cuda.Args = .{};
        a.add(picks);
        a.add(@as(c_int, @intCast(rows * slots)));
        a.add(@as(c_int, @intCast(g.experts)));
        inline for (.{ g.up, g.down }) |m| {
            a.add(m.base);
            a.add(@as(i64, @intCast(m.nb)));
            a.add(@as(i64, @intCast(m.run)));
            a.add(@as(i64, @intCast(m.head)));
        }
        const f = if (p.settings.mode == .bulk) p.bulk_fn else p.lines_fn;
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid(rows * slots) }, .block = .{ .x = threads } }, p.side, &a);
        try p.joined.record(p.side);
    }

    /// Before `main` next writes the picks (the following layer's select): wait for the side's last launch. A wait on
    /// an event never recorded (no issue yet, or a skipped window) returns at once, so it needs no flag.
    pub fn join(p: *const Prefetcher, main: cuda.Stream) !void {
        try main.wait(p.joined);
    }
};

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fread(ptr: [*]u8, size: usize, n: usize, f: *anyopaque) usize;
extern "c" fn fclose(f: *anyopaque) c_int;

/// The whole file, 16-byte aligned (a fatbin's loader wants its header aligned), at most 64 MiB.
pub fn readImageAt(gpa: std.mem.Allocator, path: [*:0]const u8) ![]align(16) u8 {
    const f = fopen(path, "rb") orelse return error.ExpertPrefetchImageMissing;
    defer _ = fclose(f);
    var buf = try gpa.alignedAlloc(u8, .@"16", 64 << 20);
    errdefer gpa.free(buf);
    const n = fread(buf.ptr, 1, buf.len, f);
    if (n == 0 or n == buf.len) return error.ExpertPrefetchImageSize;
    if (gpa.resize(buf, n)) return buf[0..n];
    const out = try gpa.alignedAlloc(u8, .@"16", n);
    @memcpy(out, buf[0..n]);
    gpa.free(buf);
    return out;
}

const t = std.testing;

const FakeLayer = struct { format: enum { nvfp4, exl3 } = .nvfp4, relu2: bool = false, up: u64 = 0x10000, down: u64 = 0x80000, width: usize = 256, dims: usize = 512, experts: usize = 8 };

test "GLM expert prefetch: TF_GLM_EXPERT_PREFETCH is off unless 1, bulk or lines, and a typo is refused" {
    try t.expectEqual(Mode.off, try Mode.parse(null));
    try t.expectEqual(Mode.off, try Mode.parse(""));
    try t.expectEqual(Mode.off, try Mode.parse("0"));
    try t.expectEqual(Mode.bulk, try Mode.parse("1"));
    try t.expectEqual(Mode.bulk, try Mode.parse("bulk"));
    try t.expectEqual(Mode.lines, try Mode.parse("lines"));
    for ([_][]const u8{ "on", "true", "2", "Bulk", " lines" }) |v| try t.expectError(error.BadExpertPrefetchMode, Mode.parse(v));
}

test "GLM expert prefetch: groups and rows are bounded and refused by name outside them" {
    const s = try Settings.parse("lines", "4", "8");
    try t.expectEqual(Settings{ .mode = .lines, .groups = 4, .rows = 8 }, s);
    try t.expectEqual(Settings{}, try Settings.parse(null, null, null));
    for ([_][]const u8{ "0", "65", "x", "-1" }) |v| try t.expectError(error.BadExpertPrefetchGroups, Settings.parse("1", v, null));
    for ([_][]const u8{ "0", "2049", "y" }) |v| try t.expectError(error.BadExpertPrefetchRows, Settings.parse("1", null, v));
    // a bad count is refused even with the mode off: a typo never waits for the day the flag is turned on
    try t.expectError(error.BadExpertPrefetchGroups, Settings.parse(null, "0", null));
}

test "GLM expert prefetch: a run starts where the decode kernel's blk pointer starts" {
    // nvfp4_experts.cu unit_full: blk = W + (e * NB + cb) * KG * (M * BLOCK4) uint4, BLOCK4 = 36 uint4 = 576 bytes
    const l: FakeLayer = .{};
    const g = try geometry(l, 2);
    const kg_up = l.dims / 32;
    const kg_down = l.width / 32;
    try t.expectEqual(@as(usize, kg_up * 2 * 36 * 16), g.up.run);
    try t.expectEqual(@as(usize, kg_down * 1 * 36 * 16), g.down.run);
    try t.expectEqual(l.width / 32, g.up.nb);
    try t.expectEqual(l.dims / 32, g.down.nb);
    // expert e's column block cb: the byte address the kernel's first stage loads from
    const e: usize = 5;
    const cb: usize = 3;
    try t.expectEqual(l.up + ((e * g.up.nb + cb) * kg_up * 2 * 36) * 16, g.up.base + e * g.up.stride() + cb * g.up.run);
    // the whole expert tensor: Nvfp4Experts.bytes(count, n, k, parts)
    try t.expectEqual(@as(usize, l.experts * (l.width / 32) * (l.dims / 32) * 2 * 144 * 4), l.experts * g.up.stride());
    try t.expectEqual(@as(usize, l.experts * (l.dims / 32) * (l.width / 32) * 1 * 144 * 4), l.experts * g.down.stride());
    // two groups: the two stages the kernel loads before its first compute
    try t.expectEqual(@as(usize, 2 * 2 * 576), g.up.head);
    try t.expectEqual(@as(usize, 2 * 576), g.down.head);
    try t.expectEqual(g.up.nb * 2304 + g.down.nb * 1152, bytesPerPick(g));
}

test "GLM expert prefetch: a head never passes its run, and every span is whole 16-byte units" {
    const l: FakeLayer = .{ .width = 64, .dims = 64 };
    const g = try geometry(l, 64);
    try t.expectEqual(g.up.run, g.up.head);
    try t.expectEqual(g.down.run, g.down.head);
    for ([_]usize{ 1, 2, 3, 7, 64 }) |n| {
        const h = try geometry(FakeLayer{}, n);
        try t.expect(h.up.head % 16 == 0 and h.down.head % 16 == 0 and h.up.run % 16 == 0);
        try t.expect(h.up.head <= h.up.run and h.down.head <= h.down.run);
    }
}

test "GLM expert prefetch: layers the decode kernel does not take are refused by name" {
    try t.expectError(error.ExpertPrefetchNeedsNvfp4, geometry(FakeLayer{ .format = .exl3 }, 2));
    try t.expectError(error.ExpertPrefetchNeedsSwiglu, geometry(FakeLayer{ .relu2 = true }, 2));
    try t.expectError(error.ExpertPrefetchShape, geometry(FakeLayer{ .width = 48 }, 2));
    try t.expectError(error.ExpertPrefetchShape, geometry(FakeLayer{ .experts = 0 }, 2));
    try t.expectError(error.ExpertPrefetchAlignment, geometry(FakeLayer{ .up = 0x10008 }, 2));
}

test "GLM expert prefetch: one block a pick up to the cap" {
    try t.expectEqual(@as(u32, 1), grid(0));
    try t.expectEqual(@as(u32, 9), grid(9)); // one decode row: top-8 plus the shared slot
    try t.expectEqual(@as(u32, 72), grid(8 * 9));
    try t.expectEqual(@as(u32, max_grid), grid(2048 * 9));
}

test "GLM expert prefetch: the flagged paths are analyzed, not only the pure helpers" {
    // lazily analyzed functions compile only when referenced: reference every one the flags reach
    _ = &Prefetcher.fromEnv;
    _ = &Prefetcher.issue;
    _ = &Prefetcher.join;
    _ = &Prefetcher.unload;
    _ = &@import("cuda_expert_ring.zig").Ring.fromEnv;
    _ = &@import("cuda_expert_ring.zig").Ring.run;
    _ = &@import("cuda_moe.zig").body;
    _ = &@import("cuda_experts.zig").run;
    _ = &@import("cuda_experts.zig").Kernels.load;
    _ = &@import("cuda_forward.zig").Modules.load;
    _ = &@import("cuda_forward.zig").Modules.deinit;
}
