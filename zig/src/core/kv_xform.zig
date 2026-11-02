//! The Atlas column-codec candidates (atlas-kv-column-codecs-plan.md, ClickHouse CODEC style), as prediction
//! transforms ahead of the plane coder and a bit-packing backend:
//!   xor    Gorilla: each element XOR the element `lag` before it
//!   delta  Delta: the wrapping difference from the element `lag` before, zigzagged
//!   delta2 DoubleDelta: the difference from the linear prediction 2 x[i-lag] - x[i-2 lag], zigzagged
//!   t64    T64 in its frame-of-reference form: 64 elements a group, each less the group minimum in the fewest bits
//! `lag` is 1 (neighbouring elements) or a row's elements (one channel across token positions: the column view).
//! Elements are little-endian u16 or u32; the len % width tail bytes pass through unchanged.
const std = @import("std");

pub const Xform = enum(u8) { none = 0, xor = 1, delta = 2, delta2 = 3 };

fn load(comptime T: type, b: []const u8, i: usize) T {
    return std.mem.readInt(T, b[i * @sizeOf(T) ..][0..@sizeOf(T)], .little);
}

fn store(comptime T: type, b: []u8, i: usize, v: T) void {
    std.mem.writeInt(T, b[i * @sizeOf(T) ..][0..@sizeOf(T)], v, .little);
}

fn zigzag(comptime T: type, d: T) T {
    const S = @Int(.signed, @bitSizeOf(T));
    const s: S = @bitCast(d);
    return (d << 1) ^ @as(T, @bitCast(s >> (@bitSizeOf(T) - 1)));
}

fn unzigzag(comptime T: type, z: T) T {
    return (z >> 1) ^ (0 -% (z & 1));
}

fn predict(comptime T: type, x: Xform, b: []const u8, i: usize, lag: usize) T {
    if (i < lag) return 0;
    const a = load(T, b, i - lag);
    if (x != .delta2 or i < 2 * lag) return a;
    return (a *% 2) -% load(T, b, i - 2 * lag);
}

fn forwardT(comptime T: type, x: Xform, lag: usize, src: []const u8, dst: []u8) void {
    const n = src.len / @sizeOf(T);
    for (0..n) |i| {
        const v = load(T, src, i);
        const p = predict(T, x, src, i, lag);
        store(T, dst, i, if (x == .xor) v ^ p else zigzag(T, v -% p));
    }
}

fn inverseT(comptime T: type, x: Xform, lag: usize, buf: []u8) void {
    const n = buf.len / @sizeOf(T);
    for (0..n) |i| {
        const r = load(T, buf, i);
        const p = predict(T, x, buf, i, lag); // elements before i are already restored
        store(T, buf, i, if (x == .xor) r ^ p else unzigzag(T, r) +% p);
    }
}

/// `dst` (as long as `src`) gets the residuals of `src` read as `w`-byte elements; `w` is 2 or 4, `lag` at least 1.
pub fn forward(x: Xform, w: u8, lag: usize, src: []const u8, dst: []u8) void {
    std.debug.assert(lag >= 1 and dst.len == src.len);
    switch (w) {
        2 => forwardT(u16, x, lag, src, dst),
        4 => forwardT(u32, x, lag, src, dst),
        else => unreachable,
    }
    const whole = src.len / w * w;
    @memcpy(dst[whole..], src[whole..]);
}

/// Undo `forward` in place.
pub fn inverse(x: Xform, w: u8, lag: usize, buf: []u8) void {
    std.debug.assert(lag >= 1);
    switch (w) {
        2 => inverseT(u16, x, lag, buf),
        4 => inverseT(u32, x, lag, buf),
        else => unreachable,
    }
}

pub const group = 64;

fn t64Bits(comptime T: type, src: []const u8, from: usize, n: usize) struct { lo: T, bits: u8 } {
    var lo: T = std.math.maxInt(T);
    var hi: T = 0;
    for (from..from + n) |i| {
        const v = load(T, src, i);
        lo = @min(lo, v);
        hi = @max(hi, v);
    }
    const range = hi - lo;
    return .{ .lo = lo, .bits = if (range == 0) 0 else @intCast(@bitSizeOf(T) - @clz(range)) };
}

fn t64SizeT(comptime T: type, src: []const u8) usize {
    const n = src.len / @sizeOf(T);
    var size: usize = src.len - n * @sizeOf(T);
    var i: usize = 0;
    while (i < n) : (i += group) {
        const k: usize = @min(group, n - i);
        size += 1 + @sizeOf(T) + (k * t64Bits(T, src, i, k).bits + 7) / 8;
    }
    return size;
}

/// The exact bytes `t64Encode` writes for `src`.
pub fn t64Size(w: u8, src: []const u8) usize {
    return switch (w) {
        2 => t64SizeT(u16, src),
        4 => t64SizeT(u32, src),
        else => unreachable,
    };
}

fn t64EncodeT(comptime T: type, src: []const u8, out: []u8) error{NoRoom}!usize {
    const n = src.len / @sizeOf(T);
    var at: usize = 0;
    var i: usize = 0;
    while (i < n) : (i += group) {
        const k: usize = @min(group, n - i);
        const g = t64Bits(T, src, i, k);
        const packed_len = (k * g.bits + 7) / 8;
        if (out.len - at < 1 + @sizeOf(T) + packed_len) return error.NoRoom;
        out[at] = g.bits;
        std.mem.writeInt(T, out[at + 1 ..][0..@sizeOf(T)], g.lo, .little);
        at += 1 + @sizeOf(T);
        var acc: u64 = 0;
        var held: u6 = 0;
        for (i..i + k) |e| {
            acc |= @as(u64, load(T, src, e) - g.lo) << held;
            var total: u8 = @as(u8, held) + g.bits;
            while (total >= 8) : (total -= 8) {
                out[at] = @truncate(acc);
                at += 1;
                acc >>= 8;
            }
            held = @intCast(total);
        }
        if (held > 0) {
            out[at] = @truncate(acc);
            at += 1;
        }
    }
    const tail = src[n * @sizeOf(T) ..];
    if (out.len - at < tail.len) return error.NoRoom;
    @memcpy(out[at..][0..tail.len], tail);
    return at + tail.len;
}

pub fn t64Encode(w: u8, src: []const u8, out: []u8) error{NoRoom}!usize {
    return switch (w) {
        2 => t64EncodeT(u16, src, out),
        4 => t64EncodeT(u32, src, out),
        else => unreachable,
    };
}

fn t64DecodeT(comptime T: type, in: []const u8, out: []u8) error{ Truncated, Corrupt }!void {
    const n = out.len / @sizeOf(T);
    var at: usize = 0;
    var i: usize = 0;
    while (i < n) : (i += group) {
        const k: usize = @min(group, n - i);
        if (in.len - at < 1 + @sizeOf(T)) return error.Truncated;
        const bits = in[at];
        if (bits > @bitSizeOf(T)) return error.Corrupt;
        const lo = std.mem.readInt(T, in[at + 1 ..][0..@sizeOf(T)], .little);
        at += 1 + @sizeOf(T);
        const packed_len = (k * bits + 7) / 8;
        if (in.len - at < packed_len) return error.Truncated;
        const mask: u64 = (@as(u64, 1) << @intCast(bits)) - 1;
        var acc: u64 = 0;
        var held: u8 = 0;
        var p = at;
        for (i..i + k) |e| {
            while (held < bits) : (held += 8) {
                acc |= @as(u64, in[p]) << @intCast(held);
                p += 1;
            }
            store(T, out, e, lo +% @as(T, @truncate(acc & mask)));
            acc >>= @intCast(bits);
            held -= bits;
        }
        at += packed_len;
    }
    const tail = out.len - n * @sizeOf(T);
    if (in.len - at != tail) return error.Corrupt;
    @memcpy(out[n * @sizeOf(T) ..], in[at..]);
}

/// Decode exactly all of `in` into exactly `out`.
pub fn t64Decode(w: u8, in: []const u8, out: []u8) error{ Truncated, Corrupt }!void {
    return switch (w) {
        2 => t64DecodeT(u16, in, out),
        4 => t64DecodeT(u32, in, out),
        else => error.Corrupt,
    };
}

test "kv xform: each transform inverts at both widths and lags, tails included" {
    var prng = std.Random.DefaultPrng.init(17);
    var src: [1027]u8 = undefined;
    var dst: [1027]u8 = undefined;
    prng.random().bytes(&src);
    for ([_]Xform{ .xor, .delta, .delta2 }) |x| for ([_]u8{ 2, 4 }) |w| for ([_]usize{ 1, 3, 64 }) |lag| {
        forward(x, w, lag, &src, &dst);
        inverse(x, w, lag, &dst);
        try std.testing.expectEqualSlices(u8, &src, &dst);
    };
}

test "kv xform: a constant stride is all zero residuals under delta2, and T64 packs a narrow range" {
    var src: [256]u8 = undefined;
    for (0..64) |i| std.mem.writeInt(u32, src[4 * i ..][0..4], @intCast(1000 + 7 * i), .little);
    var dst: [256]u8 = undefined;
    forward(.delta2, 4, 1, &src, &dst);
    for (2..64) |i| try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, dst[4 * i ..][0..4], .little));
    const size = t64Size(4, &src);
    try std.testing.expectEqual(@as(usize, 1 + 4 + (64 * 9 + 7) / 8), size); // range 441 takes 9 bits
    var out: [512]u8 = undefined;
    try std.testing.expectEqual(size, try t64Encode(4, &src, &out));
    var back: [256]u8 = undefined;
    try t64Decode(4, out[0..size], &back);
    try std.testing.expectEqualSlices(u8, &src, &back);
    try std.testing.expectError(error.Truncated, t64Decode(4, out[0 .. size - 1], &back));
}

test "kv xform: T64 round trips random groups, full-width ranges and odd tails" {
    var prng = std.Random.DefaultPrng.init(23);
    for ([_]usize{ 0, 1, 2, 3, 5, 129, 130, 4099 }) |len| for ([_]u8{ 2, 4 }) |w| {
        var src: [4099]u8 = undefined;
        prng.random().bytes(src[0..len]);
        var out: [4099 + 4099 / 64 * 5 + 16]u8 = undefined;
        const n = try t64Encode(w, src[0..len], &out);
        try std.testing.expectEqual(t64Size(w, src[0..len]), n);
        var back: [4099]u8 = undefined;
        try t64Decode(w, out[0..n], back[0..len]);
        try std.testing.expectEqualSlices(u8, src[0..len], back[0..len]);
    };
}
