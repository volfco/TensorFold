//! Common EXL3 fused projections and final tails retain the standalone kernels' arithmetic and bytes.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const base = @import("contract_exl3_n256.zig");
const rig_ = @import("contract_rig.zig");
const h = @import("quant_ref_tests.zig").helpers;
const ct = core.contract;
const c = ct.spec;
const g = ct.gates;
const epi = cuda.exl3_experts.epilogues;

pub const Product = struct {
    product: base.Product,
    kernels: epi.Kernels,
    rig: *rig_.Rig = undefined,
    ep: epi.Epi = undefined,
    fault: bool = false,
    tail: bool = false,
    baseline_tail: u64 = 0,
    fault_sd: u64 = 0,
    gateup_fn: cuda.Function = undefined,
    down_fn: cuda.Function = undefined,
    message: [160]u8 = undefined,

    pub fn deinit(self: *Product) void {
        self.product.deinit();
    }

    pub fn kernel(self: *Product) g.Kernel {
        self.rig = &self.product.rig;
        var spec = base.spec(self.product.sk, false);
        spec.name = if (self.fault) "control: fused EXL3 gate-up aliases token rows" else if (self.product.mats == 2) "cuda EXL3 fused gate-up MCg4 projection partials" else "cuda EXL3 fused down MCg4 projection partials";
        if (self.tail) {
            spec.name = if (self.fault) "control: fused EXL3 gate-up tail doubles down-input scales" else if (self.product.mats == 2) "cuda EXL3 fused gate-up FP16 tail" else "cuda EXL3 fused down FP32 tail";
            spec.out = if (self.product.mats == 2) .f16 else .f32;
            spec.replaces = "the N128 product followed by its standalone epilogue";
            spec.nan = .{ .stated = "the standalone epilogue bytes for NaN and both signed infinity inputs" };
            spec.sums = .{
                .f16 = 0,
                .bf16 = 0,
                .exact = false,
                .why = "activation and output rounding can hide reduced sums; partial registrations prove projection FP32 accumulation",
            };
            spec.ops = 64;
            spec.mids = if (self.product.mats == 2) 4 else 0;
        }
        return .{ .ctx = self, .spec = spec, .vt = &vt, .control = if (self.fault) .shapes else null };
    }

    const vt = g.hooks(Product);

    pub fn prepare(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.product.fault = self.fault and !self.tail;
        self.product.partials_output = !self.tail;
        if (!try self.product.prepare(shape, fill, seed)) return false;
        const a = self.rig.gpu.gpa;
        const picks = try a.alloc(i32, shape.rows);
        defer a.free(picks);
        @memset(picks, 0);
        const scales = try a.alloc(f16, shape.n);
        defer a.free(scales);
        @memset(scales, @as(f16, 1.0 / 1024.0));
        const sv = try self.rig.input(std.mem.sliceAsBytes(scales));
        self.ep = .{
            .pick = try self.rig.input(std.mem.sliceAsBytes(picks)),
            .sv0 = sv,
            .sv1 = sv,
            .sd = sv,
            .xd = if (self.tail and self.product.mats == 2) try self.rig.output(shape.rows * shape.n * 2) else try self.rig.scratch(shape.rows * shape.n * 2),
            .y = if (self.tail and self.product.mats == 1) try self.rig.output(shape.rows * shape.n * 4) else try self.rig.scratch(shape.rows * shape.n * 4),
            .ticket = try self.rig.scratch(((shape.rows + 15) / 16) * (shape.n / 128) * 4),
            .experts = 1,
        };
        if (self.tail) {
            self.baseline_tail = try self.rig.scratch(self.count() * @as(usize, if (self.product.mats == 2) 2 else 4));
            self.gateup_fn = try self.product.loaded.module.function("tf_exl3_gateup_epilogue");
            self.down_fn = try self.product.loaded.module.function("tf_exl3_down_combine");
            @memset(scales, @as(f16, 2.0 / 1024.0));
            self.fault_sd = try self.rig.input(std.mem.sliceAsBytes(scales));
        }
        return true;
    }

    pub fn run(self: *Product) !void {
        if (self.tail) {
            const output: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = if (self.product.mats == 2) self.ep.xd else self.ep.y, .len = self.count() * @as(usize, if (self.product.mats == 2) 2 else 4) };
            try output.fill8(0xff, self.rig.s.handle);
        }
        const a = self.product.args;
        var args: epi.Grouped = .{
            .x0 = a.x0,
            .x1 = a.x1,
            .tp0 = a.tp0,
            .tp1 = a.tp1,
            .k2_0 = a.k2_0,
            .k2_1 = a.k2_1,
            .uids = a.uids,
            .ucount = a.ucount,
            .members = a.members,
            .z = a.z,
            .k = a.k,
            .n = a.n,
            .p = a.p,
            .sk = a.sk,
            .maxm = a.maxm,
            .slots = a.slots,
            .nexp_max = a.nexp_max,
            .mats = a.mats,
            .ns = a.ns,
            .standard = a.standard,
            .skip = a.skip,
        };
        if (self.fault and !self.tail) {
            args.x0 = self.product.wrong_x;
            args.x1 = self.product.wrong_x1;
        }
        var ep = self.ep;
        if (self.fault and self.tail) ep.sd = self.fault_sd;
        try self.kernels.project(self.rig.s, .mcg, self.product.mats == 2, args, ep);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Product) usize {
        return if (self.tail) self.product.shape.rows * self.product.shape.n else self.product.count();
    }

    pub fn reference(self: *Product, want: []f64, mag: []f64) !void {
        if (!self.tail) return self.product.reference(want, mag);
        const tmp = try self.rig.gpu.gpa.alloc(f64, self.product.count() * 2);
        defer self.rig.gpu.gpa.free(tmp);
        try self.product.reference(tmp[0..self.product.count()], tmp[self.product.count()..]);
        self.finish(tmp[0..self.product.count()], want);
        for (want, mag) |v, *m| m.* = @abs(v);
    }

    pub fn simulate(self: *Product, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!self.tail) return self.product.simulate(acc, order, out);
        const tmp = try self.rig.gpu.gpa.alloc(f64, self.product.count());
        defer self.rig.gpu.gpa.free(tmp);
        try self.product.simulate(acc, order, tmp);
        self.finish(tmp, out);
    }

    pub fn replaced(self: *Product) ![]const u8 {
        if (!self.tail) return self.product.replaced();
        try self.product.launch(false, self.product.base);
        const p: i32 = @intCast(self.product.shape.rows);
        const n: i32 = @intCast(self.product.shape.n);
        var args: cuda.Args = .{};
        if (self.product.mats == 2) {
            inline for (.{ self.product.base, self.ep.pick, self.ep.sv0, self.ep.sv1, self.ep.sd, self.baseline_tail, p, n, @as(i32, @intCast(self.product.sk)), @as(i32, 1), @as(f32, 0), @as(i32, 0) }) |v| args.add(v);
        } else {
            inline for (.{ self.product.base, self.ep.pick, self.ep.sv0, self.baseline_tail, @as(u64, 0), @as(u64, 0), p, n, @as(i32, 1), @as(i32, 1), @as(i32, 1) }) |v| args.add(v);
        }
        try cuda.launch.launch(if (self.product.mats == 2) self.gateup_fn else self.down_fn, .{
            .grid = .{ .x = @intCast(p), .y = @intCast(@divTrunc(n, 128)) },
            .block = .{ .x = 32 },
        }, self.rig.s, &args);
        return self.rig.bytes(self.baseline_tail, self.count() * @as(usize, if (self.product.mats == 2) 2 else 4));
    }

    pub fn check(self: *Product, at: g.Gate) !?[]const u8 {
        if (!self.tail) return self.product.check(at);
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        if (std.mem.indexOfDiff(u8, got, try self.replaced())) |i|
            return try std.fmt.bufPrint(&self.message, "final tail differs from standalone epilogue at byte {d}", .{i});
        return null;
    }

    fn finish(self: *Product, partials: []f64, out: []f64) void {
        const shape = self.product.shape;
        const e = shape.rows * shape.n;
        for (0..shape.rows) |row| for (0..shape.n / 128) |block| {
            var gv: [128]f64 = @splat(0);
            var uv: [128]f64 = @splat(0);
            const col0 = block * 128;
            for (0..self.product.sk) |s| for (0..128) |i| {
                gv[i] += partials[(s * shape.rows + row) * shape.n + col0 + i];
                if (self.product.mats == 2) uv[i] += partials[self.product.sk * e + (s * shape.rows + row) * shape.n + col0 + i];
            };
            rotate(&gv);
            if (self.product.mats == 2) {
                rotate(&uv);
                for (&gv, uv) |*gg, uu| {
                    const gval = bf(gg.* / 1024.0);
                    const uval = bf(uu / 1024.0);
                    gg.* = bf(bf(gval / (1 + @exp(-gval))) * uval) / 1024.0;
                }
                rotate(&gv);
                for (gv, out[row * shape.n + col0 ..][0..128]) |v, *o| o.* = @as(f16, @floatCast(v));
            } else for (gv, out[row * shape.n + col0 ..][0..128]) |v, *o| o.* = v / 1024.0;
        };
    }
};

fn bf(v: f64) f64 {
    return h.bfOf(h.bf16Of(@floatCast(v)));
}

fn rotate(block: *[128]f64) void {
    var stride: usize = 1;
    while (stride < 128) : (stride *= 2) {
        var first: usize = 0;
        while (first < 128) : (first += 2 * stride) for (0..stride) |i| {
            const a = block[first + i];
            const b = block[first + stride + i];
            block[first + i] = a + b;
            block[first + stride + i] = a - b;
        };
    }
    for (block) |*v| v.* /= @sqrt(@as(f64, 128));
}
