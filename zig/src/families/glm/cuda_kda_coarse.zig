//! TF_GLM_KDA_REPLAY_INIT=fp8: a kept entry's KDA states and conv history as e4m3, one power-of-two fp32 scale a row.
//!
//! What is coarse: each KDA layer's current recurrent half ([heads][128][128] fp32, a row the 128 values of one key
//! channel) and the conv history ([3][q | k | v of heads x 128] bf16, a row one head's 128 channels of one block).
//! The arithmetic, exact on the host and the same on every run:
//!   amax  = the largest |x| over the row's finite values (0 when none);
//!   scale = 2^k, the smallest power of two with amax / 2^k <= 448 (k at least -126; 1 when amax is 0);
//!   code  = e4m3FromF32(x * 2^-k) (OCP E4M3, nearest even; `core.exlr8.codec`), the product exact;
//!   value = e4m3Value(code) * 2^k, exact in fp32 (and in bf16 for the conv history: at most 4 significant bits).
//! So the only error is e4m3's rounding: |value - x| <= max(|x| / 16, scale / 1024) a value (the second term where
//! x / scale is under 2^-6, e4m3's subnormals), and values on the e4m3 grid times the row's scale come back
//! bit-identical, as do zero rows. Not preserved: infinities (saturate to +-448 x scale) and NaN (stays NaN).
//! The copy is the state at the kept position; a replay of W tokens starts from it (see STATUS-replay.md).
const std = @import("std");
const cuda = @import("cuda");
const e4m3 = @import("core").exlr8.codec;

pub const Init = @import("cuda_snapshot_layout.zig").ReplayInit;
pub const row = 128;
const header = 32; // "tfkdafp8", then u32 layers, heads, conv_c, version 1, padding

/// A rank's coarse copy: its KDA layers, heads and conv channels a history row (3 x heads x 128).
pub const Shape = struct {
    layers: usize,
    heads: usize,
    conv_c: usize,

    pub fn of(st: anytype) !Shape {
        const s: Shape = .{ .layers = st.kda_layers, .heads = st.heads, .conv_c = st.conv_c };
        if (s.conv_c != 3 * s.heads * row or st.rec_layer != s.heads * row * row * 4) return error.KdaReplayCoarseShape;
        return s;
    }

    fn recRows(s: Shape) usize {
        return s.layers * s.heads * row;
    }

    fn convRows(s: Shape) usize {
        return s.layers * 3 * s.conv_c / row;
    }

    /// The coarse copy's bytes: the header, every row's scale (rec rows, then conv rows), every row's 128 codes.
    pub fn len(s: Shape) usize {
        return header + (s.recRows() + s.convRows()) * (4 + row);
    }

    /// One layer's recurrent half in fp32, or the conv history in bf16, whichever is larger: a staging tail's bytes.
    fn tail(s: Shape) usize {
        return @max(s.heads * row * row * 4, s.layers * 3 * s.conv_c * 2);
    }
};

/// One row's scale exponent k: amax / 2^k in (224, 448], k >= -126 (0 for a row of zeros or non-finite values).
fn exponent(xs: []const f32) i32 {
    var amax: f32 = 0;
    for (xs) |x| if (std.math.isFinite(x)) {
        amax = @max(amax, @abs(x));
    };
    if (amax == 0) return 0;
    const f = std.math.frexp(amax); // amax = f.significand x 2^f.exponent, significand in [0.5, 1)
    var k: i32 = f.exponent - 9; // amax / 2^k in [256, 512)
    if (amax > std.math.ldexp(@as(f32, 448), k)) k += 1;
    return @max(k, -126);
}

pub fn encodeRow(xs: []const f32, codes: []u8) f32 {
    const k = exponent(xs);
    const inv = std.math.ldexp(@as(f32, 1), -k);
    for (xs, codes) |x, *c| c.* = e4m3.e4m3FromF32(x * inv);
    return std.math.ldexp(@as(f32, 1), k);
}

pub fn decodeRow(codes: []const u8, scale: f32, out: []f32) void {
    for (codes, out) |c, *x| x.* = e4m3.e4m3Value(c) * scale;
}

fn scales(coarse: []u8) []align(1) f32 {
    return std.mem.bytesAsSlice(f32, coarse[header..]);
}

/// The coarse copy's view: its scales (rec rows first) and codes, the header checked against `s`.
const View = struct {
    scale: []align(1) const f32,
    code: []const u8,

    fn of(s: Shape, coarse: []const u8) !View {
        if (coarse.len != s.len() or !std.mem.eql(u8, coarse[0..8], "tfkdafp8")) return error.KdaReplayCoarseShape;
        const want = [_]u32{ @intCast(s.layers), @intCast(s.heads), @intCast(s.conv_c), 1 };
        for (want, 0..) |v, i| if (std.mem.readInt(u32, coarse[8 + 4 * i ..][0..4], .little) != v) return error.KdaReplayCoarseShape;
        const rows = s.recRows() + s.convRows();
        return .{ .scale = std.mem.bytesAsSlice(f32, coarse[header..][0 .. 4 * rows]), .code = coarse[header + 4 * rows ..] };
    }
};

pub fn writeHeader(s: Shape, coarse: []u8) void {
    @memcpy(coarse[0..8], "tfkdafp8");
    for ([_]u32{ @intCast(s.layers), @intCast(s.heads), @intCast(s.conv_c), 1 }, 0..) |v, i| std.mem.writeInt(u32, coarse[8 + 4 * i ..][0..4], v, .little);
    @memset(coarse[24..header], 0);
}

/// KDA layer `li`'s current half ([heads][128][128]) into the coarse copy.
pub fn encodeRec(s: Shape, coarse: []u8, li: usize, rec: []const f32) void {
    const rows = s.recRows() + s.convRows();
    const sc = scales(coarse);
    for (0..s.heads * row) |i| {
        const r = li * s.heads * row + i;
        sc[r] = encodeRow(rec[i * row ..][0..row], coarse[header + 4 * rows + r * row ..][0..row]);
    }
}

/// The conv history ([layers][3][conv_c] bf16) into the coarse copy.
pub fn encodeConv(s: Shape, coarse: []u8, conv: []const u16) void {
    const rows = s.recRows() + s.convRows();
    const sc = scales(coarse);
    var x: [row]f32 = undefined;
    for (0..s.convRows()) |i| {
        for (&x, conv[i * row ..][0..row]) |*v, b| v.* = @bitCast(@as(u32, b) << 16);
        const r = s.recRows() + i;
        sc[r] = encodeRow(&x, coarse[header + 4 * rows + r * row ..][0..row]);
    }
}

/// A whole copy: the header, every layer's half (`rec`: [layers][heads][128][128]) and the conv history.
pub fn encodeStates(s: Shape, coarse: []u8, rec: []const f32, conv: []const u16) void {
    writeHeader(s, coarse);
    const per = s.heads * row * row;
    for (0..s.layers) |li| encodeRec(s, coarse, li, rec[li * per ..][0..per]);
    encodeConv(s, coarse, conv);
}

/// Layer `li`'s current half from the coarse copy.
pub fn decodeRec(s: Shape, coarse: []const u8, li: usize, out: []f32) !void {
    const v = try View.of(s, coarse);
    for (0..s.heads * row) |i| {
        const r = li * s.heads * row + i;
        decodeRow(v.code[r * row ..][0..row], v.scale[r], out[i * row ..][0..row]);
    }
}

/// The conv history from the coarse copy, as bf16 (exact: a decoded value has at most 4 significant bits).
pub fn decodeConv(s: Shape, coarse: []const u8, out: []u16) !void {
    const v = try View.of(s, coarse);
    var x: [row]f32 = undefined;
    for (0..s.convRows()) |i| {
        const r = s.recRows() + i;
        decodeRow(v.code[r * row ..][0..row], v.scale[r], &x);
        for (x, out[i * row ..][0..row]) |f, *b| b.* = @intCast(@as(u32, @bitCast(f)) >> 16);
    }
}

/// Mapped pinned host memory, as the host page pool's (its device address valid under UVA): the coarse copy a keep
/// writes and the pages copy from (and a restore's pages copy into), then a tail the decoded rows go through to the
/// arena. One engine thread keeps and restores, never both at once; every use ends on a stream sync.
pub const Stage = struct {
    buf: cuda.HostBuffer,
    dev: u64,
    shape: Shape,

    pub fn open(d: *const cuda.Driver, s: Shape) !Stage {
        var buf = try cuda.HostBuffer.allocMapped(d, tailAt(s) + s.tail());
        errdefer buf.free();
        return .{ .buf = buf, .dev = try buf.device(), .shape = s };
    }

    pub fn deinit(g: *Stage) void {
        g.buf.free();
    }

    fn tailAt(s: Shape) usize {
        return std.mem.alignForward(usize, s.len(), 256);
    }

    pub fn coarse(g: *const Stage) []u8 {
        return g.buf.bytes[0..g.shape.len()];
    }

    fn tail(g: *const Stage) []align(16) u8 {
        return @alignCast(g.buf.bytes[tailAt(g.shape)..][0..g.shape.tail()]);
    }
};

/// A keep: the live conv history and each layer's current half (`st.cur`) into the stage's coarse copy.
pub fn capture(g: *const Stage, st: anytype, stream: cuda.Stream) !void {
    const s = g.shape;
    try stream.synchronize(); // the pass that made the state may still be running on a non-blocking stream
    const out = g.coarse();
    writeHeader(s, out);
    const base = st.arena.buf.ptr;
    const conv = std.mem.bytesAsSlice(u16, g.tail()[0 .. s.layers * 3 * s.conv_c * 2]);
    try st.arena.buf.download(st.conv - base, std.mem.sliceAsBytes(conv));
    encodeConv(s, out, conv);
    const rec = std.mem.bytesAsSlice(f32, g.tail()[0..st.rec_layer]);
    for (0..s.layers) |li| {
        try st.arena.buf.download(st.recAt(st.cur[li], li) - base, std.mem.sliceAsBytes(rec));
        encodeRec(s, out, li, rec);
    }
}

/// A restore: `coarse` decoded into the conv history and each layer's half 0 (the caller zeroed the head and the
/// parity), through the stage's tail, one copy and one sync at a time.
pub fn apply(g: *const Stage, st: anytype, stream: cuda.Stream, coarse: []const u8) !void {
    const s = g.shape;
    const d = st.arena.buf.d;
    const tail_dev = g.dev + Stage.tailAt(s);
    const conv = std.mem.bytesAsSlice(u16, g.tail()[0 .. s.layers * 3 * s.conv_c * 2]);
    try decodeConv(s, coarse, conv);
    try d.check(d.api.cuMemcpyDtoDAsync_v2(st.conv, tail_dev, conv.len * 2, stream.handle), "cuMemcpyDtoDAsync");
    try stream.synchronize();
    const rec = std.mem.bytesAsSlice(f32, g.tail()[0..st.rec_layer]);
    for (0..s.layers) |li| {
        try decodeRec(s, coarse, li, rec);
        try d.check(d.api.cuMemcpyDtoDAsync_v2(st.recAt(0, li), tail_dev, st.rec_layer, stream.handle), "cuMemcpyDtoDAsync");
        try stream.synchronize();
    }
}

test {
    _ = @import("cuda_kda_coarse_test.zig");
}
