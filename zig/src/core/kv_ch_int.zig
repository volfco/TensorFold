//! Reference ClickHouse integer codecs (clickhouse.com/docs/reference/statements/create/table/codec), for evaluating
//! KV byte streams in kv-codec-bench; nothing in the engine uses them. Elements are little-endian, `w` 1, 2 or 4
//! bytes; the len % w tail bytes are kept raw at the end.
//!   delta        Delta: each value less its neighbour, the first unchanged (a preparation stage: needs LZ4/ZSTD)
//!   doubleDelta  DoubleDelta: Gorilla TSDB's delta-of-delta buckets (0 | 10+7 | 110+9 | 1110+12 | 11110+w bits)
//!   t64          T64: per block the bits min and max share are cropped; each 64 values become a transposed bit
//!                matrix of the remaining bits (a preparation stage, ClickHouse's example pairs it with LZ4)
//!   gcd          GCD: every value divided by the block's greatest common divisor (a preparation stage)
const std = @import("std");
const bits = @import("kv_bits.zig");
const Error = bits.Error;
const load = bits.load;
const store = bits.store;

fn tail(src: []const u8, w: u8) []const u8 {
    return src[src.len / w * w ..];
}

/// Delta: same length as `src`.
pub fn delta(w: u8, src: []const u8, dst: []u8) void {
    const n = src.len / w;
    var prev: u32 = 0;
    for (0..n) |i| {
        const v = load(src, w, i);
        store(dst, w, i, (v -% prev) & bits.mask(w));
        prev = v;
    }
    @memcpy(dst[n * w ..], tail(src, w));
}

pub fn undelta(w: u8, buf: []u8) void {
    var prev: u32 = 0;
    for (0..buf.len / w) |i| {
        prev = (prev +% load(buf, w, i)) & bits.mask(w);
        store(buf, w, i, prev);
    }
}

fn zigzag(w: u8, d: u32) u32 {
    const shift: u5 = @intCast(8 * @as(u32, w) - 1);
    const sign = (d >> shift) & 1;
    return ((d << 1) ^ (0 -% sign)) & bits.mask(w);
}

fn unzigzag(w: u8, z: u32) u32 {
    return ((z >> 1) ^ (0 -% (z & 1))) & bits.mask(w);
}

pub fn doubleDelta(w: u8, src: []const u8, out: []u8) Error!usize {
    const n = src.len / w;
    const wb: u7 = @intCast(8 * @as(u32, w));
    var bw: bits.Writer = .{ .out = out };
    var prev: u32 = 0;
    var pdelta: u32 = 0;
    for (0..n) |i| {
        const v = load(src, w, i);
        const d = (v -% prev) & bits.mask(w);
        if (i == 0) try bw.put(v, wb) else if (i == 1) try bw.put(d, wb) else {
            const z = zigzag(w, (d -% pdelta) & bits.mask(w));
            if (z == 0) try bw.put(0, 1) else if (z < 1 << 7) {
                try bw.put(0b01, 2);
                try bw.put(z, 7);
            } else if (z < 1 << 9) {
                try bw.put(0b011, 3);
                try bw.put(z, 9);
            } else if (z < 1 << 12) {
                try bw.put(0b0111, 4);
                try bw.put(z, 12);
            } else {
                try bw.put(0b01111, 5);
                try bw.put(z, wb);
            }
        }
        prev = v;
        pdelta = d;
    }
    try bw.bytes(tail(src, w));
    return bw.finish();
}

pub fn unDoubleDelta(w: u8, in: []const u8, out: []u8) Error!void {
    const n = out.len / w;
    const wb: u7 = @intCast(8 * @as(u32, w));
    var br: bits.Reader = .{ .in = in };
    var prev: u32 = 0;
    var pdelta: u32 = 0;
    for (0..n) |i| {
        var v: u32 = undefined;
        if (i == 0) v = @intCast(try br.get(wb)) else if (i == 1) v = (prev +% @as(u32, @intCast(try br.get(wb)))) & bits.mask(w) else {
            var ones: u7 = 0;
            while (ones < 4 and try br.get(1) == 1) ones += 1;
            const z: u32 = switch (ones) {
                0 => 0,
                1 => @intCast(try br.get(7)),
                2 => @intCast(try br.get(9)),
                3 => @intCast(try br.get(12)),
                else => blk: {
                    if (try br.get(1) != 0) return error.Corrupt; // the 11110 prefix's terminating zero
                    break :blk @intCast(try br.get(wb));
                },
            };
            v = (prev +% pdelta +% unzigzag(w, z)) & bits.mask(w);
        }
        pdelta = (v -% prev) & bits.mask(w);
        prev = v;
        store(out, w, i, v);
    }
    try br.bytes(out[n * w ..]);
    try br.end();
}

/// T64 header: the cropped bit count, then min's shared high bits as one w-byte value.
pub fn t64(w: u8, src: []const u8, out: []u8) Error!usize {
    const n = src.len / w;
    var lo: u32 = bits.mask(w);
    var hi: u32 = 0;
    for (0..n) |i| {
        const v = load(src, w, i);
        lo = @min(lo, v);
        hi = @max(hi, v);
    }
    const keep: u8 = if (n == 0 or lo == hi) 0 else @intCast(32 - @clz(lo ^ hi)); // bits that differ somewhere
    const low_mask: u32 = if (keep == 32) 0xFFFF_FFFF else (@as(u32, 1) << @intCast(keep)) - 1;
    const groups = (n + 63) / 64;
    const need = 1 + w + groups * @as(usize, keep) * 8 + src.len % w;
    if (out.len < need) return error.NoRoom;
    out[0] = keep;
    store(out[1..], w, 0, lo & ~low_mask);
    var at: usize = 1 + w;
    for (0..groups) |g| {
        for (0..keep) |b| {
            var word: u64 = 0;
            for (0..64) |k| {
                const i = g * 64 + k;
                if (i < n) word |= @as(u64, (load(src, w, i) >> @intCast(b)) & 1) << @intCast(k);
            }
            std.mem.writeInt(u64, out[at..][0..8], word, .little);
            at += 8;
        }
    }
    @memcpy(out[at..][0 .. src.len % w], tail(src, w));
    return need;
}

pub fn unT64(w: u8, in: []const u8, out: []u8) Error!void {
    const n = out.len / w;
    if (in.len < 1 + w) return error.Truncated;
    const keep = in[0];
    if (keep > 8 * w) return error.Corrupt;
    const prefix = load(in[1..], w, 0);
    const groups = (n + 63) / 64;
    if (in.len != 1 + w + groups * @as(usize, keep) * 8 + out.len % w) return error.Corrupt;
    for (0..n) |i| store(out, w, i, prefix);
    var at: usize = 1 + w;
    for (0..groups) |g| for (0..keep) |b| {
        const word = std.mem.readInt(u64, in[at..][0..8], .little);
        at += 8;
        for (0..64) |k| {
            const i = g * 64 + k;
            if (i < n and (word >> @intCast(k)) & 1 != 0) store(out, w, i, load(out, w, i) | (@as(u32, 1) << @intCast(b)));
        }
    };
    @memcpy(out[n * w ..], in[at..]);
}

/// GCD: a w-byte divisor header, then the quotients (same length as the input).
pub fn gcd(w: u8, src: []const u8, out: []u8) Error!usize {
    const n = src.len / w;
    if (out.len < w + src.len) return error.NoRoom;
    var g: u32 = 0;
    for (0..n) |i| {
        const v = load(src, w, i);
        if (v != 0) g = if (g == 0) v else std.math.gcd(g, v);
    }
    if (g == 0) g = 1;
    store(out, w, 0, g & bits.mask(w));
    for (0..n) |i| store(out[w..], w, i, load(src, w, i) / g);
    @memcpy(out[w + n * w ..][0 .. src.len % w], tail(src, w));
    return w + src.len;
}

pub fn unGcd(w: u8, in: []const u8, out: []u8) Error!void {
    if (in.len != w + out.len) return error.Corrupt;
    const g = load(in, w, 0);
    if (g == 0) return error.Corrupt;
    const n = out.len / w;
    for (0..n) |i| store(out, w, i, (load(in[w..], w, i) *% g) & bits.mask(w));
    @memcpy(out[n * w ..], in[w + n * w ..]);
}

test "kv ch int: Delta, DoubleDelta, T64 and GCD invert at every width and odd length" {
    var prng = std.Random.DefaultPrng.init(31);
    var src: [1031]u8 = undefined;
    var out: [4096]u8 = undefined;
    var back: [1031]u8 = undefined;
    for ([_]u8{ 1, 2, 4 }) |w| for ([_]usize{ 0, 1, 3, 4, 7, 130, 1031 }) |len| for (0..3) |shape| {
        switch (shape) {
            0 => prng.random().bytes(src[0..len]),
            1 => for (0..len / w) |i| store(&src, w, i, @intCast((1000 + 7 * i) & bits.mask(w))), // constant stride
            else => for (0..len / w) |i| store(&src, w, i, @intCast((12 * (i % 5)) & bits.mask(w))), // GCD 12 or 4
        }
        delta(w, src[0..len], back[0..len]);
        undelta(w, back[0..len]);
        try std.testing.expectEqualSlices(u8, src[0..len], back[0..len]);
        inline for (.{ .{ doubleDelta, unDoubleDelta }, .{ t64, unT64 }, .{ gcd, unGcd } }) |c| {
            const k = try c[0](w, src[0..len], &out);
            @memset(back[0..len], 0xEE);
            try c[1](w, out[0..k], back[0..len]);
            try std.testing.expectEqualSlices(u8, src[0..len], back[0..len]);
        }
    };
}

test "kv ch int: a constant stride costs one bit a value in DoubleDelta; T64 keeps only differing bits" {
    var src: [4 * 256]u8 = undefined;
    for (0..256) |i| store(&src, 4, i, @intCast(0x4000_0000 + 3 * i));
    var out: [2048]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 8 + 254 / 8 + 1), try doubleDelta(4, &src, &out));
    const k = try t64(4, &src, &out);
    try std.testing.expectEqual(@as(u8, 10), out[0]); // 3 x 255 = 765 < 1024: the low 10 bits differ
    try std.testing.expectEqual(@as(usize, 1 + 4 + 4 * 10 * 8), k);
}
