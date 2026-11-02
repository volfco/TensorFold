//! Common EXL3 decode epilogues fused behind the same grouped projections and scratch-owned arrival counters.
const std = @import("std");
const ex = @import("exl3_experts.zig");
const launch = @import("launch.zig");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;

pub const group_shared_bytes = 64 * 1024;
pub const group_max_items = group_shared_bytes / @sizeOf(i32);

pub fn parse(raw: ?[]const u8) !bool {
    const value = raw orelse "0";
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.mem.eql(u8, value, "1")) return true;
    return error.BadExl3Epilogues;
}

pub fn fromEnv() !bool {
    const raw = std.c.getenv("TF_EXL3_FUSED_EPILOGUES");
    return parse(if (raw) |v| std.mem.span(v) else null);
}

pub fn ticketWords(l: ex.Layer, rows: usize, slots: usize) usize {
    return @min(rows * slots, l.experts) * ((rows + 15) / 16) * (l.width / 128);
}

pub fn eligible(l: ex.Layer, sc: ex.Scratch, rows: usize) bool {
    return rows > 0 and rows <= sc.rows and sc.ticket != 0 and sc.slots > 0 and sc.slots <= 32 and
        rows <= group_max_items / sc.slots and
        l.experts > 0 and l.experts <= 4096 and l.gu_stride == 0 and l.dims > 0 and l.dims % 128 == 0 and
        l.width > 0 and l.width % 64 == 0 and l.width % 128 == 0 and (l.act_mode == 0 or l.act_mode == 1);
}

pub const Grouped = extern struct {
    x0: u64,
    x1: u64,
    tp0: u64,
    tp1: u64,
    k2_0: u64,
    k2_1: u64,
    uids: u64,
    ucount: u64,
    members: u64,
    z: u64,
    k: i32,
    n: i32,
    p: i32,
    sk: i32,
    maxm: i32,
    slots: i32,
    nexp_max: i32,
    mats: i32,
    ns: i32 = 0,
    standard: i32 = 0,
    skip: i32 = -1,
};

pub const Epi = extern struct {
    pick: u64,
    sv0: u64,
    sv1: u64 = 0,
    sd: u64 = 0,
    xd: u64 = 0,
    y: u64 = 0,
    ticket: u64 = 0,
    experts: i32,
    limit: f32 = 0,
    act_mode: i32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(Grouped) == 128 and @offsetOf(Grouped, "skip") == 120);
    std.debug.assert(@sizeOf(Epi) == 72 and @offsetOf(Epi, "experts") == 56);
}

fn i(v: usize) i32 {
    return @intCast(v);
}

pub const Kernels = struct {
    group: Function,
    gateup: [3]Function,
    down: [3]Function,

    pub fn resolve(mod: Module) !Kernels {
        var k: Kernels = .{ .group = try mod.function("tf_exl3_group_epi"), .gateup = undefined, .down = undefined };
        try k.group.allowDynamicShared(group_shared_bytes);
        inline for (0..3) |cb| {
            k.gateup[cb] = try mod.function(std.fmt.comptimePrint("tf_exl3_fused_gu_{d}", .{cb}));
            k.down[cb] = try mod.function(std.fmt.comptimePrint("tf_exl3_fused_down_{d}", .{cb}));
        }
        return k;
    }

    pub fn project(k: Kernels, s: Stream, cb: ex.Codebook, gu: bool, g: Grouped, ep: Epi) !void {
        if (g.standard != 0 or g.k <= 0 or g.n <= 0 or g.p <= 0 or g.slots <= 0 or g.nexp_max <= 0 or
            @mod(g.n, 128) != 0 or g.maxm <= 0 or g.sk <= 0 or
            @mod(g.k, 64 * g.sk) != 0 or (gu and g.mats != 2) or (!gu and (g.mats != 1 or g.sk != 1)))
            return error.BadExl3EpilogueShape;
        var args: launch.Args = .{};
        args.add(g);
        args.add(ep);
        const mt: u32 = @intCast(@divTrunc(g.maxm + 15, 16));
        try launch.launch((if (gu) k.gateup else k.down)[@backingInt(cb)], .{
            .grid = .{ .x = @intCast(g.nexp_max), .y = @intCast(@divTrunc(g.n, 128)), .z = @intCast(@as(i32, @intCast(mt)) * g.mats * g.sk) },
            .block = .{ .x = 128 },
        }, s, &args);
    }

    pub fn run(k: Kernels, owner: *const ex.Experts, s: Stream, l: ex.Layer, x: u64, stride: usize, dtype: ex.DType, pick: u64, rows: usize, sc: ex.Scratch) !void {
        if (!eligible(l, sc, rows)) return error.BadExl3EpilogueShape;
        const p = rows * sc.slots;
        const maxu = @min(p, l.experts);
        var group: launch.Args = .{};
        inline for (.{ pick, sc.ids, sc.count, sc.members, i(rows), i(sc.slots), i(l.experts), i(rows), sc.ticket, i(ticketWords(l, rows, sc.slots)) }) |v| group.add(v);
        try launch.launch(k.group, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 }, .shared = @intCast(p * 4) }, s, &group);
        var rot: launch.Args = .{};
        inline for (.{ x, i(stride), @as(i32, @backingInt(dtype)), pick, l.suh_g, l.suh_u, sc.xg, sc.xu, i(l.dims), i(sc.slots), i(l.experts) }) |v| rot.add(v);
        try launch.launch(owner.rot_in, .{ .grid = .{ .x = @intCast(p), .y = @intCast(l.dims / 128), .z = 2 }, .block = .{ .x = 32 } }, s, &rot);
        const gu: Grouped = .{ .x0 = sc.xg, .x1 = sc.xu, .tp0 = l.gate, .tp1 = l.up, .k2_0 = l.gate_k2, .k2_1 = l.up_k2, .uids = sc.ids, .ucount = sc.count, .members = sc.members, .z = sc.z, .k = i(l.dims), .n = i(l.width), .p = i(p), .sk = @intCast(try ex.plan(l.dims, l.width, true)), .maxm = i(rows), .slots = i(sc.slots), .nexp_max = i(maxu), .mats = 2 };
        try k.project(s, l.codebook, true, gu, .{ .pick = pick, .sv0 = l.svh_g, .sv1 = l.svh_u, .sd = l.suh_d, .xd = sc.xd, .ticket = sc.ticket, .experts = i(l.experts), .limit = l.limit, .act_mode = l.act_mode });
        const down: Grouped = .{ .x0 = sc.xd, .x1 = sc.xd, .tp0 = l.down, .tp1 = l.down, .k2_0 = l.down_k2, .k2_1 = l.down_k2, .uids = sc.ids, .ucount = sc.count, .members = sc.members, .z = sc.z, .k = i(l.width), .n = i(l.dims), .p = i(p), .sk = 1, .maxm = i(rows), .slots = i(sc.slots), .nexp_max = i(maxu), .mats = 1 };
        try k.project(s, l.codebook, false, down, .{ .pick = pick, .sv0 = l.svh_d, .y = sc.y, .experts = i(l.experts) });
    }
};

test "EXL3 fused epilogues: strict opt-in" {
    try std.testing.expect(!try parse(null));
    try std.testing.expect(!try parse("0"));
    try std.testing.expect(try parse("1"));
    try std.testing.expectError(error.BadExl3Epilogues, parse(" 1 "));
    try std.testing.expectError(error.BadExl3Epilogues, parse(""));
    try std.testing.expectError(error.BadExl3Epilogues, parse("false"));
    try std.testing.expectError(error.BadExl3Epilogues, parse("2"));
}

test "EXL3 fused epilogues: grouping refuses above its opt-in shared-memory budget" {
    var l: ex.Layer = undefined;
    l.experts = 1;
    l.gu_stride = 0;
    l.dims = 128;
    l.width = 128;
    l.act_mode = 0;
    var sc: ex.Scratch = undefined;
    sc.rows = std.math.maxInt(usize);
    sc.ticket = 1;
    for ([_]usize{ 1, 8, 32 }) |slots| {
        sc.slots = slots;
        const most = group_max_items / slots;
        try std.testing.expect(eligible(l, sc, most - 1));
        try std.testing.expect(eligible(l, sc, most));
        try std.testing.expect(!eligible(l, sc, most + 1));
        try std.testing.expect(!eligible(l, sc, std.math.maxInt(usize)));
    }
    sc.slots = 0;
    try std.testing.expect(!eligible(l, sc, 1));
    sc.slots = 33;
    try std.testing.expect(!eligible(l, sc, 1));
}
