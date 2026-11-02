//! TF_NVFP4_DECODE_RING: GLM's routed NVFP4 decode experts through nvfp4_expert_ring.cu, D k32 groups of loads in
//! flight in registers instead of the stock kernel's two. Same bits by construction (the groups are computed in the
//! same order); only load timing changes. Unbuilt prototype: TF_NVFP4_DECODE_RING_IMAGE names a hand-built fatbin.

const std = @import("std");
const cuda = @import("cuda");

/// TF_NVFP4_DECODE_RING: unset (the stock kernel), 2 (the ring kernel at the stock depth: the control that must equal
/// the stock kernel's bits and time), 3 or 4. Anything else is refused by name.
pub fn parse(text: ?[]const u8) !?u8 {
    const v = text orelse return null;
    if (v.len == 0) return null;
    if (v.len == 1 and v[0] >= '2' and v[0] <= '4') return v[0] - '0';
    return error.BadDecodeRing;
}

/// The entry names nvfp4_expert_ring.cu's TF_RING lines define, by depth: gate-up (SwiGLU), then fp32 down.
pub fn symbols(depth: u8) ![2][:0]const u8 {
    return switch (depth) {
        2 => .{ "tf_nvfp4_ring_gate_up_d2", "tf_nvfp4_ring_down_f32_d2" },
        3 => .{ "tf_nvfp4_ring_gate_up_d3", "tf_nvfp4_ring_down_f32_d3" },
        4 => .{ "tf_nvfp4_ring_gate_up_d4", "tf_nvfp4_ring_down_f32_d4" },
        else => error.BadDecodeRing,
    };
}

pub const warps = 4; // the ring entries are built at the stock decode's 4 warps a block only

/// The epilogues the ring entries carry; the others keep the stock kernels (the caller asks `takes` first).
pub fn takes(which: enum { gate_up, down_f32, down }, relu2: bool) bool {
    return switch (which) {
        .gate_up => !relu2,
        .down_f32 => true,
        .down => false,
    };
}

pub const Ring = struct {
    module: cuda.Module,
    depth: u8,
    fns: [2]cuda.Function,
    resident: [2]usize,

    /// null when TF_NVFP4_DECODE_RING is unset; with it set, a missing image is refused by name.
    pub fn fromEnv(d: *const cuda.Driver, sms: usize) !?Ring {
        const gpa = std.heap.page_allocator;
        const depth = (try parse(if (std.c.getenv("TF_NVFP4_DECODE_RING")) |v| std.mem.span(v) else null)) orelse return null;
        if (std.c.getenv("TF_NVFP4_DECODE_WARPS")) |v| if (!std.mem.eql(u8, std.mem.span(v), "4")) return error.DecodeRingNeedsFourWarps;
        const path = std.c.getenv("TF_NVFP4_DECODE_RING_IMAGE") orelse return error.DecodeRingNeedsImage;
        const image = try @import("cuda_expert_prefetch.zig").readImageAt(gpa, path);
        defer gpa.free(image);
        return try load(d, image, depth, sms);
    }

    pub fn load(d: *const cuda.Driver, image: []const u8, depth: u8, sms: usize) !Ring {
        return loadNamed(d, image, try symbols(depth), depth, sms);
    }

    /// Any two entries with the stock decode's arguments: the bench takes the stock kernels themselves from the ring
    /// image (it includes nvfp4_experts.cu, so it carries nvfp4_expert_kernel<M, EPI, 4> as built beside the ring).
    pub fn loadNamed(d: *const cuda.Driver, image: []const u8, names: [2][:0]const u8, depth: u8, sms: usize) !Ring {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        var r: Ring = .{ .module = m, .depth = depth, .fns = undefined, .resident = undefined };
        for (names, &r.fns, &r.resident) |name, *f, *res| {
            f.* = try m.function(name);
            res.* = @max(1, try f.occupancy(warps * 32, 0)) * sms;
        }
        return r;
    }

    pub fn unload(r: *Ring) void {
        r.module.unload();
    }

    /// Nvfp4Experts.run's launch with the ring entry: the same arguments in the same order, the same grid rule.
    pub fn run(r: *const Ring, s: cuda.Stream, gate: bool, in: cuda.experts.Rows, l: cuda.experts.Layer, p: cuda.grouped.Plan, out: u64, items: usize, skip: i32) !void {
        const kg = (if (gate) l.dims else l.width) / 32;
        const nb = (if (gate) l.width else l.dims) / 32;
        const i: usize = if (gate) 0 else 1;
        const g = grid(items, nb, r.resident[i]);
        if (g < 1) return;
        var a: cuda.Args = .{};
        a.add(in.x);
        a.add(@as(i32, @intCast(in.stride)));
        a.add(@as(i32, @intCast(if (gate) in.slots else 0)));
        a.add(if (gate) l.up else l.down);
        a.add(if (gate) l.up_scale else l.down_scale);
        a.add(@as(i32, @intCast(kg)));
        a.add(@as(i32, @intCast(nb)));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(if (gate) l.width else l.dims)));
        a.add(if (gate) l.limit else @as(f32, 0.0));
        a.add(skip);
        try cuda.launch.launch(r.fns[i], .{ .grid = .{ .x = @intCast(g) }, .block = .{ .x = warps * 32 } }, s, &a);
    }
};

/// Nvfp4Experts.run's grid: a warp a (item, column block) unit, capped at the blocks resident at once.
pub fn grid(items: usize, nb: usize, resident: usize) usize {
    return @min((items * nb + warps - 1) / warps, resident);
}

const t = std.testing;

test "GLM expert ring: TF_NVFP4_DECODE_RING takes 2, 3 or 4 and refuses anything else by name" {
    try t.expectEqual(@as(?u8, null), try parse(null));
    try t.expectEqual(@as(?u8, null), try parse(""));
    try t.expectEqual(@as(?u8, 2), try parse("2"));
    try t.expectEqual(@as(?u8, 3), try parse("3"));
    try t.expectEqual(@as(?u8, 4), try parse("4"));
    for ([_][]const u8{ "1", "5", "0", "33", "on", "d3" }) |v| try t.expectError(error.BadDecodeRing, parse(v));
    try t.expectError(error.BadDecodeRing, symbols(5));
}

test "GLM expert ring: each depth names its two entries, gate-up first" {
    for ([_]u8{ 2, 3, 4 }) |depth| {
        const n = try symbols(depth);
        try t.expect(std.mem.startsWith(u8, n[0], "tf_nvfp4_ring_gate_up_d"));
        try t.expect(std.mem.startsWith(u8, n[1], "tf_nvfp4_ring_down_f32_d"));
        try t.expectEqual(depth, n[0][n[0].len - 1] - '0');
        try t.expectEqual(depth, n[1][n[1].len - 1] - '0');
    }
}

test "GLM expert ring: only SwiGLU gate-up and fp32 down take the ring; the rest keep the stock kernels" {
    try t.expect(takes(.gate_up, false));
    try t.expect(!takes(.gate_up, true));
    try t.expect(takes(.down_f32, false));
    try t.expect(!takes(.down, false));
}

test "GLM expert ring: the grid is the stock decode's (a warp a unit, capped at the resident blocks)" {
    try t.expectEqual(@as(usize, 0), grid(0, 64, 192));
    try t.expectEqual(@as(usize, 16), grid(1, 64, 192));
    try t.expectEqual(@as(usize, 1), grid(1, 1, 192));
    try t.expectEqual(@as(usize, 192), grid(72, 64, 192));
}

test "GLM expert ring: k_loop_ring's slots compute every group once, in order, each from its own load" {
    // a model of nvfp4_expert_ring.cu's k_loop_ring: slot d holds the group last loaded into it; a compute reads the
    // slot, and the reload of group g + D into it only follows that compute (the stock kernel is D = 2)
    for ([_]usize{ 2, 3, 4 }) |depth| for (1..70) |kg| {
        var slot: [4]?usize = @splat(null);
        for (0..@min(depth, kg)) |dd| slot[dd] = dd;
        var next: usize = 0;
        var g0: usize = 0;
        while (g0 < kg) : (g0 += depth) for (0..depth) |dd| {
            const g = g0 + dd;
            if (g >= kg) continue;
            try t.expectEqual(@as(?usize, g), slot[dd]); // the slot holds exactly the group computed now
            try t.expectEqual(next, g); // groups in order: the same fp32 accumulation chain as the stock kernel
            next += 1;
            slot[dd] = if (g + depth < kg) g + depth else null;
        };
        try t.expectEqual(kg, next);
        for (slot) |s| try t.expectEqual(@as(?usize, null), s); // no load left unconsumed
    };
}
