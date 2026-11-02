//! KDA vector staging: every state/readout byte equals the shipped step; separate cancellation probes prove FP32 sums.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ct = core.contract;
const c = ct.spec;
const g = ct.gates;
const width = 128;
const Kind = enum { bytes, state, readout };

const served = [_]c.Shape{
    .{ .rows = 256, .n = 32, .k = width },  .{ .rows = 2048, .n = 32, .k = width },
    .{ .rows = 257, .n = 21, .k = width },  .{ .rows = 1024, .n = 22, .k = width },
    .{ .rows = 2048, .n = 64, .k = width }, .{ .rows = 3, .n = 32, .k = width },
};

fn depth(s: c.Shape) usize {
    return s.rows * (width + 8);
}

fn spec(kind: Kind, tile: u32, tail: bool) c.Spec {
    const name = switch (kind) {
        .bytes => if (tail) "control: KDA vector skips the last prompt row" else if (tile == 8) "cuda glm KDA vector8 state and readout bytes" else "cuda glm KDA vector16 state and readout bytes",
        .state => if (tile == 8) "cuda glm KDA vector8 update FP32 sums" else "cuda glm KDA vector16 update FP32 sums",
        .readout => if (tile == 8) "cuda glm KDA vector8 readout FP32 sums" else "cuda glm KDA vector16 readout FP32 sums",
    };
    return .{
        .name = name,
        .accum = if (kind == .bytes) .none else .f32,
        .depth = depth,
        .ops = if (kind == .state) 2 else 0,
        .out = switch (kind) {
            .bytes => .raw,
            .state => .f32,
            .readout => .bf16,
        },
        .replaces = "tf_glm_kda_step (unchanged W8/TR16 recurrence)",
        .nan = .{ .stated = "every readout and FP32 state byte equals the shipped step, including NaN and both infinities" },
        .tiles = if (kind == .bytes) .{ .{ .size = tile }, .{}, .{} } else .{ .{}, .{}, .{} },
        .served = if (kind == .bytes) &served else &.{.{ .rows = 1, .n = 32, .k = width }},
    };
}

pub const Case = struct {
    rig: rig_.Rig,
    arena: std.heap.ArenaAllocator,
    base: cuda.Function,
    candidate: cuda.Function,
    kind: Kind = .bytes,
    tile: u32 = 8,
    tail: bool = false,
    shape: c.Shape = undefined,
    state: []f32 = &.{},
    q: []f32 = &.{},
    k: []f32 = &.{},
    v: []u16 = &.{},
    decay: []f32 = &.{},
    beta: []f32 = &.{},
    in: [6]u64 = undefined,
    y: u64 = 0,
    out: u64 = 0,
    base_y: u64 = 0,
    base_state: u64 = 0,
    baseline: std.ArrayList(u8) = .empty,

    const vt = g.hooks(Case);

    pub fn deinit(self: *Case) void {
        self.baseline.deinit(self.rig.gpu.gpa);
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        return .{ .spec = spec(self.kind, self.tile, self.tail), .ctx = self, .vt = &vt, .control = if (self.tail) .shapes else null };
    }

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != width or shape.rows == 0 or shape.n == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        const nr = shape.rows * shape.n * width;
        const ns = shape.n * width * width;
        self.state = try a.alloc(f32, ns);
        self.q = try a.alloc(f32, nr);
        self.k = try a.alloc(f32, nr);
        self.v = try a.alloc(u16, nr);
        self.decay = try a.alloc(f32, nr);
        self.beta = try a.alloc(f32, shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.state) |*x| x.* = (rng.float(f32) * 2 - 1) * 0.01;
        for (self.q, self.k, self.v, self.decay) |*q, *k, *v, *d| {
            q.* = (rng.float(f32) * 2 - 1) * 0.15;
            k.* = (rng.float(f32) * 2 - 1) * 0.15;
            v.* = c.toBf16(rng.float(f32) * 2 - 1);
            d.* = 0.95 + rng.float(f32) * 0.05;
        }
        for (self.beta) |*x| x.* = 0.1 + rng.float(f32) * 0.8;
        if (fill == .stress) self.stress();
        if (fill == .nonfinite) {
            self.q[0] = std.math.nan(f32);
            self.q[width] = std.math.inf(f32);
            self.q[2 * width] = -std.math.inf(f32);
            self.decay[3 * width] = std.math.inf(f32);
        }
        const data = [_][]const u8{ std.mem.sliceAsBytes(self.state), std.mem.sliceAsBytes(self.q), std.mem.sliceAsBytes(self.k), std.mem.sliceAsBytes(self.v), std.mem.sliceAsBytes(self.decay), std.mem.sliceAsBytes(self.beta) };
        for (data, &self.in) |bytes, *p| p.* = try self.rig.input(bytes);
        self.y = if (self.kind == .state) try self.rig.scratch(nr * 2) else try self.rig.output(nr * 2);
        self.out = if (self.kind == .readout) try self.rig.scratch(ns * 4) else try self.rig.output(ns * 4);
        self.base_y = try self.rig.scratch(nr * 2);
        self.base_state = try self.rig.scratch(ns * 4);
        return true;
    }

    fn stress(self: *Case) void {
        for (self.state, 0..) |*x, i| x.* = stressTerm(i % width);
        @memset(self.decay, 1);
        @memset(self.v, 0);
        @memset(self.beta, if (self.kind == .state) 1 else 0);
        @memset(self.k, if (self.kind == .state) 1 else 0);
        @memset(self.q, if (self.kind == .readout) 1 else 0);
    }

    fn launch(self: *Case, f: cuda.Function, y: u64, state: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(@as(c_int, @intCast(self.shape.n)));
        for (self.in) |p| a.add(p);
        a.add(@as(c_int, @intCast(rows)));
        a.add(y);
        a.add(state);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(self.shape.n), .y = 4 }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Case) !void {
        try self.launch(self.candidate, self.y, self.out, self.shape.rows - @as(usize, @intFromBool(self.tail)));
    }

    pub fn replaced(self: *Case) ![]const u8 {
        try self.launch(self.base, self.base_y, self.base_state, self.shape.rows);
        self.baseline.clearRetainingCapacity();
        if (self.kind != .state) try self.baseline.appendSlice(self.rig.gpu.gpa, try self.rig.bytes(self.base_y, self.shape.rows * self.shape.n * width * 2));
        if (self.kind != .readout) try self.baseline.appendSlice(self.rig.gpu.gpa, try self.rig.bytes(self.base_state, self.shape.n * width * width * 4));
        return self.baseline.items;
    }

    pub fn count(self: *Case) usize {
        return switch (self.kind) {
            .bytes => self.shape.rows * self.shape.n * width * 2 + self.shape.n * width * width * 4,
            .state => self.shape.n * width * width,
            .readout => self.shape.rows * self.shape.n * width,
        };
    }

    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const got = try self.rig.outputBytes();
        const want = try self.replaced();
        if (!std.mem.eql(u8, got, want)) return "candidate state or readout differs from the shipped step";
        var nonfinite: usize = 0;
        var finite: usize = 0;
        const raw = try self.rig.bytes(self.y, self.shape.rows * self.shape.n * width * 2);
        for (0..self.shape.rows * self.shape.n * width) |i| {
            const y = c.fromBf16(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little));
            if (std.math.isFinite(y)) finite += 1 else nonfinite += 1;
        }
        if (finite == 0 or nonfinite == 0) return "non-finite plants did not leave both finite and non-finite readouts";
        return null;
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            w.* = if (self.kind == .state) @as(f64, self.state[i]) - 1 else 1;
            m.* = 4194303;
        }
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        var terms: [width]f64 = undefined;
        for (&terms, 0..) |*x, i| x.* = stressTerm(i);
        const sum = ct.rows.sumIn(acc, order, &terms);
        for (out, 0..) |*x, i| x.* = if (self.kind == .state) @as(f32, @floatCast(@as(f64, self.state[i]) - sum)) else c.fromBf16(c.toBf16(@floatCast(sum)));
    }
};

fn stressTerm(i: usize) f32 {
    return if (i < width / 2) 32768 else if (i == width / 2) -32767 else -32768;
}

pub const Registrations = struct {
    module: cuda.Module,
    cases: [7]Case,

    pub fn init(gpu: check_.Gpu, s: cuda.Stream) !Registrations {
        var module = try cuda.Module.load(gpu.d, cuda.kernels.glm_kda);
        errdefer module.unload();
        const base = try module.function("tf_glm_kda_step");
        const fast = [_]cuda.Function{ try module.function("tf_glm_kda_step_vector8"), try module.function("tf_glm_kda_step_vector16") };
        var out: Registrations = .{ .module = module, .cases = undefined };
        for (&out.cases, 0..) |*item, i| item.* = .{
            .rig = .{ .gpu = gpu, .s = s },
            .arena = .init(gpu.gpa),
            .base = base,
            .candidate = fast[if (i < 3 or i == 6) 0 else 1],
            .kind = @fromBackingInt(@intCast(i % 3)),
            .tile = if (i < 3 or i == 6) 8 else 16,
            .tail = i == 6,
        };
        return out;
    }

    pub fn append(self: *Registrations, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        for (&self.cases) |*item| try list.append(a, item.kernel());
    }

    pub fn deinit(self: *Registrations) void {
        for (&self.cases) |*item| item.deinit();
        self.module.unload();
    }
};

pub fn run(gpu: check_.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var registrations = try Registrations.init(gpu, s);
    defer registrations.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try registrations.append(gpu.gpa, &kernels);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x6b6461, &tally);
    std.debug.print("SUMMARY KDA diagnostic: {d} kernels, {d} pass, {d} fail; {d} controls, {d} red, {d} missed; limits not run\n", .{ tally.kernels, tally.passed, tally.failed, tally.controls, tally.caught, tally.missed });
    try check_.expect(tally.problems() == 0, "KDA vector diagnostic: {d} gate failures and {d} controls missed", .{ tally.failed, tally.missed });
}
