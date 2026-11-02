//! EXL3's N256 MCg4 product: identical K partitions and warp sums to the N128 product.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ct = core.contract;
const c = ct.spec;
const g = ct.gates;
const fmt = core.exl3_format;

pub const Args = extern struct {
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
    ns: i32,
    standard: i32,
    skip: i32,
};

fn depth(s: c.Shape) usize {
    return s.k + 4;
}

const served = [_]c.Shape{
    .{ .rows = 8, .n = 1024, .k = 4096 },
    .{ .rows = 1, .n = 1024, .k = 4096 },
    .{ .rows = 4, .n = 4096, .k = 1024 },
    .{ .rows = 16, .n = 512, .k = 512 },
    .{ .rows = 17, .n = 256, .k = 256 },
};

pub fn spec(sk: usize, fault: bool) c.Spec {
    return .{
        .name = if (fault) "control: EXL3 N256 aliases token rows" else if (sk == 4) "cuda EXL3 N256 MCg4 gate-up product" else "cuda EXL3 N256 MCg4 down product",
        .accum = .f32,
        .depth = depth,
        .out = .f32,
        .replaces = "tf_exl3_grouped_1 N128 product",
        .nan = .{ .stated = "the N128 product's bytes, including NaN and both infinity signs" },
        .tiles = .{ .{ .size = 16 }, .{ .size = 256, .step = 256, .min = 256 }, .{ .size = 256, .step = 256, .min = 256 } },
        .served = &served,
    };
}

const small_served = [_]c.Shape{
    .{ .rows = 8, .n = 4096, .k = 1024 }, .{ .rows = 1, .n = 4096, .k = 1024 },
    .{ .rows = 2, .n = 4096, .k = 1024 }, .{ .rows = 3, .n = 4096, .k = 1024 },
    .{ .rows = 4, .n = 4096, .k = 1024 }, .{ .rows = 5, .n = 4096, .k = 1024 },
    .{ .rows = 6, .n = 4096, .k = 1024 }, .{ .rows = 7, .n = 4096, .k = 1024 },
    .{ .rows = 4, .n = 4096, .k = 640 },  .{ .rows = 4, .n = 4096, .k = 768 },
};

pub fn smallSpec(fault: bool) c.Spec {
    var s = spec(1, fault);
    s.name = if (fault) "control: EXL3 N256 small down aliases token rows" else "cuda EXL3 N256 small MCg4 down product";
    s.tiles = .{ .{}, .{ .size = 256, .step = 256, .min = 256 }, .{ .size = 64, .step = 64, .min = 64 } };
    s.served = &small_served;
    return s;
}

pub const Loaded = struct {
    module: cuda.Module,
    base: cuda.Function,
    candidate: cuda.Function,
    small: cuda.Function,

    pub fn init(gpu: check.Gpu) !Loaded {
        var m = try cuda.Module.load(gpu.d, cuda.kernels.exl3_experts);
        errdefer m.unload();
        return .{ .module = m, .base = try m.function("tf_exl3_grouped_1"), .candidate = try m.function("tf_exl3_grouped_256_1"), .small = try m.function("tf_exl3_grouped_256_small_1") };
    }

    pub fn deinit(self: *Loaded) void {
        self.module.unload();
    }
};

/// The public product fixture also qualifies fused epilogues' unchanged projection partials.
pub const Product = struct {
    rig: rig_.Rig,
    loaded: *const Loaded,
    sk: usize = 4,
    mats: usize = 2,
    fault: bool = false,
    small: bool = false,
    partials_output: bool = true,
    run_hook: ?*const fn (*Product) anyerror!void = null,
    shape: c.Shape = undefined,
    args: Args = undefined,
    x: []f16 = &.{},
    w: []f16 = &.{},
    out: u64 = 0,
    base: u64 = 0,
    wrong_x: u64 = 0,
    wrong_x1: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Product) void {
        self.drop();
        self.rig.deinit();
    }

    fn drop(self: *Product) void {
        self.rig.gpu.gpa.free(self.x);
        self.rig.gpu.gpa.free(self.w);
        self.x = &.{};
        self.w = &.{};
        self.rig.reset();
    }

    pub fn kernel(self: *Product) g.Kernel {
        return .{ .spec = if (self.small) smallSpec(self.fault) else spec(self.sk, self.fault), .ctx = self, .vt = &vt, .control = if (self.fault) .shapes else null };
    }

    const vt = g.hooks(Product);

    pub fn prepare(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 256 != 0 or shape.k == 0 or shape.k % (self.sk * 64) != 0) return false;
        if (self.small and (shape.rows > 8 or self.mats != 1 or self.sk != 1)) return false;
        self.drop();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        self.x = try a.alloc(f16, shape.rows * shape.k);
        self.w = try a.alloc(f16, shape.k * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const trellis = try a.alloc(i16, shape.k / 16 * (shape.n / 16) * 64);
        defer a.free(trellis);
        for (trellis) |*v| v.* = if (fill == .stress) 0 else rng.int(i16);
        fmt.unpack(.{ .halves = 8 }, .mcg, trellis, shape.k / 16, shape.n / 16, self.w);
        for (self.x) |*v| v.* = @floatCast(rng.float(f32) * 4 - 2);
        if (fill == .stress) {
            const per = shape.k / self.sk;
            for (0..shape.rows) |row| for (0..self.sk) |s| for (0..per / 2) |i| {
                const value: f16 = @floatFromInt(32 * (1 + rng.uintLessThan(u16, 64)));
                self.x[row * shape.k + s * per + i] = value;
                self.x[row * shape.k + s * per + per / 2 + i] = -value + @as(f16, if (i < 8) 32 else 0);
            };
        } else if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, shape.k / 3 }, .{ 1, shape.k / 2 }, .{ 2, shape.k - 1 }, .{ 3, 0 }, .{ 3, shape.k - 1 } };
            const values = [_]f16{ std.math.nan(f16), std.math.inf(f16), -std.math.inf(f16), std.math.inf(f16), -std.math.inf(f16) };
            for (at, values) |p, value| if (p[0] < shape.rows) {
                self.x[p[0] * shape.k + p[1]] = value;
            };
        }
        const x = try self.rig.input(std.mem.sliceAsBytes(self.x));
        const other = try a.dupe(f16, self.x);
        defer a.free(other);
        for (other) |*v| v.* = -v.*;
        const x1 = if (self.mats == 2) try self.rig.input(std.mem.sliceAsBytes(other)) else x;
        const tr = try self.rig.input(std.mem.sliceAsBytes(trellis));
        const ptrs = try self.rig.input(std.mem.asBytes(&tr));
        const k2 = try self.rig.input(std.mem.asBytes(&@as(i32, 8)));
        const uid = try self.rig.input(std.mem.asBytes(&@as(i32, 0)));
        const ucount = try self.rig.input(std.mem.asBytes(&@as(i32, 1)));
        const members = try a.alloc(i32, shape.rows);
        defer a.free(members);
        for (members, 0..) |*v, row| v.* = @intCast(row * 32);
        const member_ptr = try self.rig.input(std.mem.sliceAsBytes(members));
        self.out = if (self.partials_output) try self.rig.output(self.count() * 4) else try self.rig.scratch(self.count() * 4);
        self.base = try self.rig.scratch(self.count() * 4);
        self.args = .{ .x0 = x, .x1 = x1, .tp0 = ptrs, .tp1 = ptrs, .k2_0 = k2, .k2_1 = k2, .uids = uid, .ucount = ucount, .members = member_ptr, .z = self.out, .k = @intCast(shape.k), .n = @intCast(shape.n), .p = @intCast(shape.rows), .sk = @intCast(self.sk), .maxm = @intCast(shape.rows), .slots = 1, .nexp_max = 1, .mats = @intCast(self.mats), .ns = 0, .standard = 0, .skip = -1 };
        if (self.fault) {
            const wrong = try a.dupe(f16, self.x);
            defer a.free(wrong);
            for (1..shape.rows) |row| @memcpy(wrong[row * shape.k ..][0..shape.k], self.x[0..shape.k]);
            self.wrong_x = try self.rig.input(std.mem.sliceAsBytes(wrong));
            for (wrong) |*v| v.* = -v.*;
            self.wrong_x1 = if (self.mats == 2) try self.rig.input(std.mem.sliceAsBytes(wrong)) else self.wrong_x;
        }
        return true;
    }

    pub fn launch(self: *Product, candidate: bool, output: u64) !void {
        var args = self.args;
        args.z = output;
        if (candidate and self.fault) {
            args.x0 = self.wrong_x;
            args.x1 = self.wrong_x1;
        }
        var launch_args: cuda.Args = .{};
        launch_args.add(args);
        const f = if (!candidate) self.loaded.base else if (self.small) self.loaded.small else self.loaded.candidate;
        try cuda.launch.launch(f, .{ .grid = .{ .x = 1, .y = @intCast(self.shape.n / @as(usize, if (candidate) 256 else 128)), .z = @intCast(self.mats * self.sk * ((self.shape.rows + 15) / 16)) }, .block = .{ .x = 128 } }, self.rig.s, &launch_args);
    }

    pub fn run(self: *Product) !void {
        if (self.run_hook) |hook| try hook(self) else try self.launch(true, self.out);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Product) ![]const u8 {
        try self.launch(false, self.base);
        return self.rig.bytes(self.base, self.count() * 4);
    }

    pub fn check(self: *Product, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} differs from N128", .{at});
        var planted = false;
        var finite = false;
        for (0..got.len / 4) |i| {
            if (c.classOf(c.read(.f32, got, i)) == .finite) finite = true else planted = true;
        }
        return if (planted and finite) null else "non-finite plants did not leave both finite and non-finite outputs";
    }

    pub fn count(self: *Product) usize {
        return self.mats * self.sk * self.shape.rows * self.shape.n;
    }

    fn terms(self: *const Product, matrix: usize, s: usize, row: usize, col: usize, out: []f64) void {
        const begin = s * (self.shape.k / self.sk);
        for (out, begin..) |*v, k| {
            const x: f64 = self.x[row * self.shape.k + k];
            v.* = (if (matrix == 1) -x else x) * @as(f64, self.w[k * self.shape.n + col]);
        }
    }

    pub fn reference(self: *Product, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        const t = try a.alloc(f64, self.shape.k / self.sk);
        defer a.free(t);
        for (0..self.mats) |m| for (0..self.sk) |s| for (0..self.shape.rows) |row| for (0..self.shape.n) |col| {
            self.terms(m, s, row, col, t);
            var value: f64 = 0;
            var magnitude: f64 = 0;
            for (t) |v| {
                value += v;
                magnitude += @abs(v);
            }
            const at = ((m * self.sk + s) * self.shape.rows + row) * self.shape.n + col;
            want[at] = value;
            mag[at] = magnitude;
        };
    }

    pub fn simulate(self: *Product, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const t = try a.alloc(f64, self.shape.k / self.sk);
        defer a.free(t);
        for (0..self.mats) |m| for (0..self.sk) |s| for (0..self.shape.rows) |row| for (0..self.shape.n) |col| {
            self.terms(m, s, row, col, t);
            out[((m * self.sk + s) * self.shape.rows + row) * self.shape.n + col] = ct.rows.sumIn(acc, order, t);
        };
    }
};

test "EXL3 grouped argument ABI includes the final padding word" {
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(Args));
    try std.testing.expectEqual(@as(usize, 120), @offsetOf(Args, "skip"));
}
