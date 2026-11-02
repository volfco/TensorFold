//! TF_GLM_KDA_REPLAY_INIT=fp8's host arithmetic: the row scale, the e4m3 bound, exact rows and the copy's layout.
const std = @import("std");
const t = std.testing;
const coarse = @import("cuda_kda_coarse.zig");
const e4m3 = @import("core").exlr8.codec;

fn bounded(x: []const f32, y: []const f32, scale: f32) !void {
    for (x, y) |a, b| try t.expect(@abs(b - a) <= @max(@abs(a) / 16, scale / 1024));
}

test "GLM KDA coarse: a row's scale is a power of two putting its max in (224, 448], every value within e4m3's step" {
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    var x: [128]f32 = undefined;
    var codes: [128]u8 = undefined;
    var y: [128]f32 = undefined;
    for ([_]f32{ 1e-30, 3e-4, 0.07, 1, 13.5, 448, 449, 6e20 }) |mag| {
        for (&x) |*v| v.* = (rnd.float(f32) * 2 - 1) * mag;
        const scale = coarse.encodeRow(&x, &codes);
        const f = std.math.frexp(scale);
        try t.expectEqual(@as(f32, 0.5), f.significand); // a power of two
        var amax: f32 = 0;
        for (x) |v| amax = @max(amax, @abs(v));
        try t.expect(amax / scale <= 448 and amax / scale > 224);
        coarse.decodeRow(&codes, scale, &y);
        try bounded(&x, &y, scale);
        var d2: f64 = 0;
        var r2: f64 = 0;
        for (x, y) |a, b| {
            d2 += (@as(f64, b) - a) * (@as(f64, b) - a);
            r2 += @as(f64, a) * a;
        }
        try t.expect(@sqrt(d2 / r2) < 0.0625); // relative L2 under e4m3's half step
        // deterministic: the same row gives the same codes and scale
        var again: [128]u8 = undefined;
        try t.expectEqual(scale, coarse.encodeRow(&x, &again));
        try t.expectEqualSlices(u8, &codes, &again);
    }
}

test "GLM KDA coarse: e4m3 grid values times the row's scale and zero rows come back bit-identical" {
    var x: [128]f32 = undefined;
    var codes: [128]u8 = undefined;
    var y: [128]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = e4m3.e4m3Value(@intCast(i % 0x7F)) * 0x1p-20; // codes 0 .. 0x7E, 448 the max
    for (x[64..]) |*v| v.* = -v.*; // both signs
    const scale = coarse.encodeRow(&x, &codes);
    try t.expectEqual(@as(f32, 0x1p-20), scale);
    coarse.decodeRow(&codes, scale, &y);
    for (x, y) |a, b| try t.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
    @memset(&x, 0);
    try t.expectEqual(@as(f32, 1), coarse.encodeRow(&x, &codes));
    coarse.decodeRow(&codes, 1, &y);
    for (y) |b| try t.expectEqual(@as(u32, 0), @as(u32, @bitCast(b)));
    // infinities saturate to 448 x scale and NaN stays NaN; the scale comes from the finite values
    x[3] = std.math.inf(f32);
    x[4] = std.math.nan(f32);
    x[5] = 2;
    const s2 = coarse.encodeRow(&x, &codes);
    coarse.decodeRow(&codes, s2, &y);
    try t.expectEqual(@as(f32, 448) * s2, y[3]);
    try t.expect(std.math.isNan(y[4]));
    try t.expectEqual(@as(f32, 2), y[5]);
}

test "GLM KDA coarse: a copy holds every layer's half and the conv history at a quarter of fp32, its shape checked" {
    const s: coarse.Shape = .{ .layers = 2, .heads = 2, .conv_c = 3 * 2 * 128 };
    try t.expectEqual(@as(usize, 32 + (2 * 2 * 128 + 2 * 3 * 6) * 132), s.len());
    const buf = try t.allocator.alloc(u8, s.len());
    defer t.allocator.free(buf);
    const per = 2 * 128 * 128;
    const rec = try t.allocator.alloc(f32, 2 * per);
    defer t.allocator.free(rec);
    for (rec, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 211)) - 105)) * 0.013;
    var conv: [2 * 3 * 768]u16 = undefined;
    for (&conv, 0..) |*v, i| v.* = (0x3f80 + @as(u16, @intCast(i % 97))) | (if (i % 5 == 0) @as(u16, 0x8000) else 0);
    coarse.writeHeader(s, buf);
    for (0..2) |li| coarse.encodeRec(s, buf, li, rec[li * per ..][0..per]);
    coarse.encodeConv(s, buf, &conv);
    const out = try t.allocator.alloc(f32, per);
    defer t.allocator.free(out);
    for (0..2) |li| {
        try coarse.decodeRec(s, buf, li, out);
        for (0..2 * 128) |r| {
            var codes: [128]u8 = undefined;
            const scale = coarse.encodeRow(rec[li * per + r * 128 ..][0..128], &codes);
            try bounded(rec[li * per + r * 128 ..][0..128], out[r * 128 ..][0..128], scale);
        }
    }
    var back: [conv.len]u16 = undefined;
    try coarse.decodeConv(s, buf, &back);
    for (conv, back) |a, b| {
        const x: f32 = @bitCast(@as(u32, a) << 16);
        const y: f32 = @bitCast(@as(u32, b) << 16);
        try t.expect(@abs(y - x) <= @abs(x) / 16);
    }
    // a copy of another shape, or a damaged header, is refused by name
    const other: coarse.Shape = .{ .layers = 2, .heads = 1, .conv_c = 3 * 128 };
    try t.expectError(error.KdaReplayCoarseShape, coarse.decodeRec(other, buf[0..other.len()], 0, out));
    buf[0] = 'x';
    try t.expectError(error.KdaReplayCoarseShape, coarse.decodeConv(s, buf, &back));
}
