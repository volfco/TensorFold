//! Reference ClickHouse float codecs for kv-codec-bench (nothing in the engine uses them); values are the raw bit
//! patterns, little-endian, `w` bytes (FPC: 4); the len % w tail bytes are kept raw at the end.
//!   gorilla  Gorilla TSDB: XOR with the previous value; 0 when equal, else 10 + the meaningful bits inside the last
//!            window, or 11 + leading-zero count + length + the meaningful bits
//!   fpc      FPC (Burtscher & Ratanaworabhan, DCC 2007): the better of an FCM and a DFCM hash predictor, XOR, and a
//!            nibble of predictor bit + leading zero bytes; tables of 2^level entries
//!   alpRd    ALP's RD mode: each value split into left (sign, exponent, top mantissa) and right bits; the left part
//!            from a dictionary of at most 8 entries (else an exception), the right part bit-packed
const std = @import("std");
const bits = @import("kv_bits.zig");
const Error = bits.Error;
const load = bits.load;
const store = bits.store;

fn log2w(w: u8) u7 {
    return switch (w) {
        1 => 3,
        2 => 4,
        4 => 5,
        else => unreachable,
    };
}

pub fn gorilla(w: u8, src: []const u8, out: []u8) Error!usize {
    const n = src.len / w;
    const wb: u7 = @intCast(8 * @as(u32, w));
    const lb = log2w(w);
    var bw: bits.Writer = .{ .out = out };
    var prev: u32 = 0;
    var lead: u7 = wb; // no window yet
    var trail: u7 = 0;
    for (0..n) |i| {
        const v = load(src, w, i);
        defer prev = v;
        if (i == 0) {
            try bw.put(v, wb);
            continue;
        }
        const x = v ^ prev;
        if (x == 0) {
            try bw.put(0, 1);
            continue;
        }
        const l: u7 = @as(u7, @clz(x)) - (32 - wb);
        const t: u7 = @intCast(@ctz(x));
        if (lead < wb and l >= lead and t >= trail) {
            try bw.put(0b01, 2); // 1 then 0: inside the last window
            try bw.put(x >> @intCast(trail), wb - lead - trail);
        } else {
            try bw.put(0b11, 2);
            const len = wb - l - t;
            try bw.put(l, lb);
            try bw.put(len - 1, lb);
            try bw.put(x >> @intCast(t), len);
            lead = l;
            trail = t;
        }
    }
    try bw.bytes(src[n * w ..]);
    return bw.finish();
}

pub fn unGorilla(w: u8, in: []const u8, out: []u8) Error!void {
    const n = out.len / w;
    const wb: u7 = @intCast(8 * @as(u32, w));
    const lb = log2w(w);
    var br: bits.Reader = .{ .in = in };
    var prev: u32 = 0;
    var lead: u7 = wb;
    var trail: u7 = 0;
    for (0..n) |i| {
        var v: u32 = undefined;
        if (i == 0) v = @intCast(try br.get(wb)) else if (try br.get(1) == 0) v = prev else if (try br.get(1) == 0) {
            if (lead >= wb) return error.Corrupt; // a window before any was set
            v = prev ^ (@as(u32, @intCast(try br.get(wb - lead - trail))) << @intCast(trail));
        } else {
            const l: u7 = @intCast(try br.get(lb));
            const len: u7 = @as(u7, @intCast(try br.get(lb))) + 1;
            if (@as(u32, l) + len > wb) return error.Corrupt;
            const t = wb - l - len;
            v = prev ^ (@as(u32, @intCast(try br.get(len))) << @intCast(t));
            lead = l;
            trail = t;
        }
        store(out, w, i, v);
        prev = v;
    }
    try br.bytes(out[n * w ..]);
    try br.end();
}

/// FPC's two hash predictors over 32-bit values; `tables` holds 2 << level entries.
const Fpc = struct {
    fcm: []u32,
    dfcm: []u32,
    mask: u32,
    h1: u32 = 0,
    h2: u32 = 0,
    last: u32 = 0,

    fn init(level: u5, tables: []u32) Fpc {
        const size = @as(usize, 1) << level;
        std.debug.assert(tables.len >= 2 * size);
        @memset(tables[0 .. 2 * size], 0);
        return .{ .fcm = tables[0..size], .dfcm = tables[size .. 2 * size], .mask = @intCast(size - 1) };
    }

    fn predictions(f: *const Fpc) [2]u32 {
        return .{ f.fcm[f.h1], f.dfcm[f.h2] +% f.last };
    }

    fn update(f: *Fpc, v: u32) void {
        f.fcm[f.h1] = v;
        f.h1 = ((f.h1 << 6) ^ (v >> 20)) & f.mask;
        const d = v -% f.last;
        f.dfcm[f.h2] = d;
        f.h2 = ((f.h2 << 2) ^ (d >> 20)) & f.mask;
        f.last = v;
    }

    /// The nibble (predictor bit, then leading zero bytes 0-4) and the residual.
    fn code(f: *const Fpc, v: u32) struct { nib: u8, x: u32 } {
        const p = f.predictions();
        const x0 = v ^ p[0];
        const x1 = v ^ p[1];
        const z0: u8 = @intCast(@clz(x0) / 8);
        const z1: u8 = @intCast(@clz(x1) / 8);
        return if (z1 > z0) .{ .nib = 8 | z1, .x = x1 } else .{ .nib = z0, .x = x0 };
    }
};

/// FPC on 4-byte values: a header byte per pair (two nibbles), then each value's 4 - zero-bytes residual bytes.
pub fn fpc(level: u5, tables: []u32, src: []const u8, out: []u8) Error!usize {
    const n = src.len / 4;
    var f = Fpc.init(level, tables);
    var at: usize = 0;
    var i: usize = 0;
    while (i < n) : (i += 2) {
        const pair = @min(2, n - i);
        var head: u8 = 0;
        var res: [2]u32 = undefined;
        var keep: [2]u8 = .{ 0, 0 };
        for (0..pair) |k| {
            const v = load(src, 4, i + k);
            const c = f.code(v);
            head |= c.nib << @intCast(4 * k);
            res[k] = c.x;
            keep[k] = 4 - (c.nib & 7);
            f.update(v);
        }
        if (out.len - at < 1 + keep[0] + keep[1]) return error.NoRoom;
        out[at] = head;
        at += 1;
        for (0..pair) |k| for (0..keep[k]) |b| {
            out[at] = @truncate(res[k] >> @intCast(8 * b));
            at += 1;
        };
    }
    const t = src[n * 4 ..];
    if (out.len - at < t.len) return error.NoRoom;
    @memcpy(out[at..][0..t.len], t);
    return at + t.len;
}

pub fn unFpc(level: u5, tables: []u32, in: []const u8, out: []u8) Error!void {
    const n = out.len / 4;
    var f = Fpc.init(level, tables);
    var at: usize = 0;
    var i: usize = 0;
    while (i < n) : (i += 2) {
        if (at >= in.len) return error.Truncated;
        const head = in[at];
        at += 1;
        for (0..@min(2, n - i)) |k| {
            const nib: u8 = (head >> @intCast(4 * k)) & 0xF;
            const zeros = nib & 7;
            if (zeros > 4) return error.Corrupt;
            const keep = 4 - zeros;
            if (in.len - at < keep) return error.Truncated;
            var x: u32 = 0;
            for (0..keep) |b| x |= @as(u32, in[at + b]) << @intCast(8 * b);
            at += keep;
            const v = x ^ f.predictions()[nib >> 3];
            store(out, 4, i + k, v);
            f.update(v);
        }
    }
    if (in.len - at != out.len - n * 4) return error.Corrupt;
    @memcpy(out[n * 4 ..], in[at..]);
}

const most_entries = 8;

/// The left width (searched on a sample) and its dictionary of the most frequent left values.
const Split = struct { left: u7, dict: [most_entries]u32 = undefined, size: u8 = 0 };

fn codeBits(size: u8) u7 {
    return if (size <= 1) 0 else @intCast(32 - @clz(@as(u32, size - 1)));
}

fn dictionary(w: u8, src: []const u8, n: usize, left: u7, counts: []u32) Split {
    const wb: u7 = @intCast(8 * @as(u32, w));
    @memset(counts[0 .. @as(usize, 1) << @intCast(left)], 0);
    for (0..n) |i| counts[load(src, w, i) >> @intCast(wb - left)] += 1;
    var s: Split = .{ .left = left };
    while (s.size < most_entries) {
        var best: ?usize = null;
        for (counts[0 .. @as(usize, 1) << @intCast(left)], 0..) |c, v| if (c > 0 and (best == null or c > counts[best.?])) {
            best = v;
        };
        const b = best orelse break;
        s.dict[s.size] = @intCast(b);
        counts[b] = 0; // taken
        s.size += 1;
    }
    return s;
}

/// ALP RD: header (left width, dictionary size and entries as u16, exception count u32), then a code and the right
/// bits a value, then (position u32, left u16) per exception. `counts` holds 1 << 16 entries.
pub fn alpRd(w: u8, src: []const u8, out: []u8, counts: []u32) Error!usize {
    std.debug.assert(w == 2 or w == 4);
    const n = src.len / w;
    const wb: u7 = @intCast(8 * @as(u32, w));
    const sample = @min(n, 4096);
    var best: Split = .{ .left = 1 };
    var least: u64 = std.math.maxInt(u64);
    var left: u7 = 1;
    while (left <= @min(16, wb - 1)) : (left += 1) {
        const s = dictionary(w, src, sample, left, counts);
        var exceptions: u64 = sample;
        for (0..sample) |i| {
            const l = load(src, w, i) >> @intCast(wb - left);
            if (std.mem.indexOfScalar(u32, s.dict[0..s.size], l) != null) exceptions -= 1;
        }
        const cost = @as(u64, sample) * (codeBits(s.size) + wb - left) + exceptions * 48;
        if (cost < least) {
            least = cost;
            best = s;
        }
    }
    const s = if (n > 0) dictionary(w, src, n, best.left, counts) else best;
    const right_bits = wb - s.left;
    var bw: bits.Writer = .{ .out = out };
    try bw.put(s.left, 8);
    try bw.put(s.size, 8);
    for (s.dict[0..s.size]) |d| try bw.put(d, 16);
    var exceptions: u32 = 0;
    for (0..n) |i| {
        if (std.mem.indexOfScalar(u32, s.dict[0..s.size], load(src, w, i) >> @intCast(right_bits)) == null) exceptions += 1;
    }
    try bw.put(exceptions, 32);
    for (0..n) |i| {
        const v = load(src, w, i);
        const code = std.mem.indexOfScalar(u32, s.dict[0..s.size], v >> @intCast(right_bits)) orelse 0;
        try bw.put(code, codeBits(s.size));
        try bw.put(v, right_bits);
    }
    try bw.flush();
    for (0..n) |i| {
        const l = load(src, w, i) >> @intCast(right_bits);
        if (std.mem.indexOfScalar(u32, s.dict[0..s.size], l) == null) {
            try bw.put(@intCast(i), 32);
            try bw.put(l, 16);
        }
    }
    try bw.bytes(src[n * w ..]);
    return bw.finish();
}

pub fn unAlpRd(w: u8, in: []const u8, out: []u8) Error!void {
    const n = out.len / w;
    const wb: u7 = @intCast(8 * @as(u32, w));
    var br: bits.Reader = .{ .in = in };
    const left: u7 = @intCast(try br.get(8));
    const size: u8 = @intCast(try br.get(8));
    if (left == 0 or left >= wb or left > 16 or size > most_entries or (size == 0 and n > 0)) return error.Corrupt;
    var dict: [most_entries]u32 = undefined;
    for (dict[0..size]) |*d| d.* = @intCast(try br.get(16));
    const exceptions: u32 = @intCast(try br.get(32));
    if (exceptions > n) return error.Corrupt;
    const right_bits = wb - left;
    for (0..n) |i| {
        const code = try br.get(codeBits(size));
        if (code >= size) return error.Corrupt;
        const right: u32 = @intCast(try br.get(right_bits));
        store(out, w, i, (dict[code] << @intCast(right_bits)) | right);
    }
    try br.alignByte();
    for (0..exceptions) |_| {
        const at: usize = @intCast(try br.get(32));
        const l: u32 = @intCast(try br.get(16));
        if (at >= n or l >> @intCast(left) != 0) return error.Corrupt;
        const right = load(out, w, at) & ((@as(u32, 1) << @intCast(right_bits)) - 1);
        store(out, w, at, (l << @intCast(right_bits)) | right);
    }
    try br.bytes(out[n * w ..]);
    try br.end();
}

test "kv ch float: Gorilla, FPC and ALP RD invert on random, slow and repeated values with tails" {
    var prng = std.Random.DefaultPrng.init(37);
    var src: [4099]u8 = undefined;
    var out: [16384]u8 = undefined;
    var back: [4099]u8 = undefined;
    const tables = try std.testing.allocator.alloc(u32, 2 << 12);
    defer std.testing.allocator.free(tables);
    const counts = try std.testing.allocator.alloc(u32, 1 << 16);
    defer std.testing.allocator.free(counts);
    for ([_]usize{ 0, 1, 3, 4, 9, 4099 }) |len| for (0..3) |shape| {
        switch (shape) {
            0 => prng.random().bytes(src[0..len]),
            1 => for (0..len / 4) |i| store(&src, 4, i, @bitCast(@as(f32, 1.0) + @as(f32, @floatFromInt(i)) * 1e-4)),
            else => for (src[0..len], 0..) |*b, i| {
                b.* = @truncate(i / 64);
            },
        }
        for ([_]u8{ 1, 2, 4 }) |w| {
            const k = try gorilla(w, src[0..len], &out);
            try unGorilla(w, out[0..k], back[0..len]);
            try std.testing.expectEqualSlices(u8, src[0..len], back[0..len]);
        }
        const k = try fpc(12, tables, src[0..len], &out);
        try unFpc(12, tables, out[0..k], back[0..len]);
        try std.testing.expectEqualSlices(u8, src[0..len], back[0..len]);
        for ([_]u8{ 2, 4 }) |w| {
            const a = try alpRd(w, src[0..len], &out, counts);
            try unAlpRd(w, out[0..a], back[0..len]);
            try std.testing.expectEqualSlices(u8, src[0..len], back[0..len]);
        }
    };
}

test "kv ch float: equal values cost Gorilla one bit and FPC half a byte" {
    var src: [4 * 128]u8 = undefined;
    for (0..128) |i| store(&src, 4, i, 0x3F80_0000);
    var out: [1024]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 4 + (127 + 7) / 8), try gorilla(4, &src, &out));
    var tables: [2 << 4]u32 = undefined;
    // the first pair predicts 0 (4 residual bytes), the second value onwards hits the FCM or DFCM exactly
    try std.testing.expect(try fpc(4, &tables, &src, &out) < 128); // raw: 512
}
