//! TF_GLM_INDEX_PRUNE's strips under the contract on CUDA: tf_glm_prune_bounds writes each page a write of pooled keys
//! completes (pages out of order) as its keys' per-dimension minimum, then maximum: exact bf16 values, NaN when the
//! column holds one, infinities kept. A swapped table is the planted control (the strips land on other pages).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const rig_ = @import("contract_rig.zig");
const pages = @import("contract_glm_pages.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const dk = pages.dk;
const pp = glm.index_prune.page_pools;
/// A page here: its 128 pooled keys, then the strip (as a striped cache's part of a page holds them).
const page_elems = pp * dk + 2 * dk;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A decode window's write and a prompt chunk's (k: the strip's two halves).
pub fn spec() c.Spec {
    return .{ .name = "cuda glm tf_glm_prune_bounds on pages out of order", .accum = .none, .depth = noSums, .out = .bf16, .served = &.{ .{ .rows = 8, .n = dk, .k = 2 }, .{ .rows = 2048, .n = dk, .k = 2 } } };
}

pub const Strips = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    wrong_table: bool = false,
    shape: c.Shape = undefined,
    keys: []u16 = &.{}, // [pools][dk] in pool order
    ids: []i32 = &.{},
    at: usize = 0,
    first: usize = 0, // the first page the write completes
    whole: usize = 0, // the pages it completes
    arena_in: u64 = 0,
    arena: u64 = 0,
    arena_bytes: usize = 0,
    pos: u64 = 0,
    pt: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Strips) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Strips) void {
        const a = self.rig.gpu.gpa;
        a.free(self.keys);
        a.free(self.ids);
        self.keys = &.{};
        self.ids = &.{};
    }

    pub fn kernel(self: *Strips) g.Kernel {
        var declared = spec();
        if (self.wrong_table) declared.name = "control: GLM page strips use a swapped table";
        return .{ .spec = declared, .ctx = self, .vt = &vt, .control = if (self.wrong_table) .shapes else null };
    }

    const vt = g.hooks(Strips);

    /// Keys in [-2, 2); non-finite: NaN, +inf and -inf in the first page the write completes. A decode window ends a
    /// page; a prompt chunk completes four.
    pub fn prepare(self: *Strips, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != dk or shape.k != 2 or shape.rows == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.at = if (shape.rows <= 8) 3 * 512 - shape.rows / 2 else 1000;
        const done = (self.at + shape.rows) / 4;
        self.first = self.at / 4 / pp;
        self.whole = (done / pp) -| self.first;
        if (self.whole == 0) return false;
        const n_pages = done / pp + 1;
        self.keys = try a.alloc(u16, n_pages * pp * dk);
        for (self.keys) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        if (fill == .nonfinite) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (put, 0..) |v, i| self.keys[(self.first * pp + 5 + 40 * i) * dk + 3 + 50 * i] = c.toBf16(v);
        }
        self.ids = try pages.shuffled(a, rng, n_pages);
        const arena = try a.alloc(u16, (n_pages + 3) * page_elems);
        defer a.free(arena);
        @memset(arena, 0); // strips nobody wrote read as 0
        for (0..n_pages * pp) |p| @memcpy(arena[@as(usize, @intCast(self.ids[p / pp])) * page_elems + (p % pp) * dk ..][0..dk], self.keys[p * dk ..][0..dk]);
        self.arena_bytes = arena.len * 2;
        self.arena_in = try self.rig.input(std.mem.sliceAsBytes(arena));
        self.arena = try self.rig.scratch(self.arena_bytes);
        const p: i32 = @intCast(self.at);
        self.pos = try self.rig.input(std.mem.asBytes(&p));
        self.pt = try pages.uploadTable(&self.rig, self.ids, self.wrong_table);
        self.out = try self.rig.output(self.whole * 2 * dk * 2);
        return true;
    }

    /// The keys onto the pages, the strips written, then each completed page's strip back in page order.
    pub fn run(self: *Strips) !void {
        try copy(&self.rig, self.arena, self.arena_in, self.arena_bytes);
        var a: cuda.Args = .{};
        inline for (.{ self.arena, self.pt, @as(i64, page_elems * 2), self.pos, @as(i64, @intCast(self.shape.rows)) }) |v| a.add(v);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(glm.index_prune.writeGrid(self.shape.rows)) }, .block = .{ .x = glm.index_prune.strip_threads } }, self.rig.s, &a);
        for (0..self.whole) |i| {
            const page: u64 = @intCast(self.ids[self.first + i]);
            try copy(&self.rig, self.out + i * 4 * dk, self.arena + (page * page_elems + pp * dk) * 2, 4 * dk);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Strips) usize {
        return self.whole * 2 * dk;
    }

    /// Each completed page's minima, then maxima, in float64 (a NaN in a column gives NaN at both).
    pub fn reference(self: *Strips, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        for (0..self.whole) |i| for (0..dk) |d| {
            const page = self.first + i;
            var lo = c.fromBf16(self.keys[page * pp * dk + d]);
            var hi = lo;
            for (1..pp) |q| {
                const v = c.fromBf16(self.keys[(page * pp + q) * dk + d]);
                lo = if (std.math.isNan(v) or v < lo) v else lo;
                hi = if (std.math.isNan(v) or v > hi) v else hi;
            }
            want[i * 2 * dk + d] = lo;
            want[i * 2 * dk + dk + d] = hi;
        };
    }
};

fn copy(r: *rig_.Rig, dst: u64, src: u64, len: usize) !void {
    try (cuda.DeviceBuffer{ .d = r.gpu.d, .ptr = dst, .len = len }).copyFrom(0, src, len, r.s.handle);
}
