//! Order-0 rANS over bytes: four interleaved 32-bit states, 12-bit frequencies, byte renormalization (ryg_rans layout).
//! The encoder runs backwards and leaves its output at the end of the caller's buffer; the decoder refuses any stream
//! that does not end exactly at its input's end with every state back at `low`.
const std = @import("std");

pub const prob_bits = 12;
pub const scale: u32 = 1 << prob_bits;
pub const lanes = 4;
const low: u32 = 1 << 23; // a state stays in [low, low << 8)

pub const Error = error{ NoRoom, BadTable, Truncated, Corrupt };

/// Normalized frequencies: every counted symbol at least 1, the sum exactly `scale`.
pub const Table = struct {
    freq: [256]u16 = @splat(0),
    cum: [257]u16 = @splat(0),

    /// `counts` must hold at least one nonzero entry.
    pub fn fromCounts(counts: *const [256]u64) Table {
        var t: Table = .{};
        var total: u64 = 0;
        for (counts) |c| total += c;
        std.debug.assert(total > 0);
        var sum: u32 = 0;
        for (counts, 0..) |c, s| if (c > 0) {
            const f: u32 = @intCast(@max(1, (@as(u128, c) * scale) / total));
            t.freq[s] = @intCast(f);
            sum += f;
        };
        // floor rounding leaves room; clamps to 1 can overshoot by at most one a symbol
        while (sum != scale) {
            var big: usize = 0;
            for (t.freq, 0..) |f, s| if (f > t.freq[big]) {
                big = s;
            };
            if (sum < scale) {
                t.freq[big] += @intCast(scale - sum);
                sum = scale;
            } else {
                const cut: u32 = @min(sum - scale, @as(u32, t.freq[big]) - 1);
                std.debug.assert(cut > 0);
                t.freq[big] -= @intCast(cut);
                sum -= cut;
            }
        }
        t.finish();
        return t;
    }

    fn finish(t: *Table) void {
        var at: u16 = 0;
        for (t.freq, 0..) |f, s| {
            t.cum[s] = at;
            at += f;
        }
        t.cum[256] = at;
    }

    pub fn present(t: *const Table) usize {
        var n: usize = 0;
        for (t.freq) |f| n += @intFromBool(f > 0);
        return n;
    }

    /// A 32-byte presence bitmap, then each present symbol's frequency as a little-endian u16.
    pub fn serializedLen(t: *const Table) usize {
        return 32 + 2 * t.present();
    }

    pub fn write(t: *const Table, out: []u8) Error!usize {
        const n = t.serializedLen();
        if (out.len < n) return error.NoRoom;
        @memset(out[0..32], 0);
        var at: usize = 32;
        for (t.freq, 0..) |f, s| if (f > 0) {
            out[s / 8] |= @as(u8, 1) << @intCast(s % 8);
            std.mem.writeInt(u16, out[at..][0..2], f, .little);
            at += 2;
        };
        return n;
    }

    /// The table at the start of `in` and the bytes it took; a forged table is refused before use.
    pub fn read(in: []const u8) Error!struct { table: Table, used: usize } {
        if (in.len < 32) return error.Truncated;
        var t: Table = .{};
        var at: usize = 32;
        var sum: u32 = 0;
        for (0..256) |s| if (in[s / 8] & (@as(u8, 1) << @intCast(s % 8)) != 0) {
            if (in.len < at + 2) return error.Truncated;
            const f = std.mem.readInt(u16, in[at..][0..2], .little);
            if (f == 0 or f > scale) return error.BadTable;
            t.freq[s] = f;
            sum += f;
            at += 2;
        };
        if (sum != scale) return error.BadTable;
        t.finish();
        return .{ .table = t, .used = at };
    }
};

/// The worst output for `n` symbols is under 1.5 bytes each (a 1/4096 symbol) plus the states; callers give `n` and
/// fall back to raw on NoRoom, so this is only the size a caller wanting a guaranteed fit would provide.
pub fn bound(n: usize) usize {
    return n + n / 2 + 4 * lanes + 8;
}

/// One symbol pushed onto state `x` (encoding runs backwards), its renormalization bytes written below `ptr`.
inline fn put(t: *const Table, x: *u32, s: u8, out: []u8, ptr: *usize) Error!void {
    const f: u32 = t.freq[s];
    if (f == 0) return error.Corrupt;
    const x_max = ((low >> prob_bits) << 8) * f;
    while (x.* >= x_max) {
        if (ptr.* == 0) return error.NoRoom;
        ptr.* -= 1;
        out[ptr.*] = @truncate(x.*);
        x.* >>= 8;
    }
    x.* = ((x.* / f) << prob_bits) + (x.* % f) + t.cum[s];
}

/// Encode `src` with `t` (which must count every symbol in it) into the tail of `out`; returns where the stream starts.
/// Symbol i rides state i % lanes; the four states step together (comptime indices keep them in registers).
pub fn encode(t: *const Table, src: []const u8, out: []u8) Error!usize {
    var st: [lanes]u32 = @splat(low);
    var ptr = out.len;
    const whole = src.len / lanes * lanes;
    var i = src.len;
    while (i > whole) { // the last, partial group first: encoding runs backwards
        i -= 1;
        try put(t, &st[i % lanes], src[i], out, &ptr);
    }
    while (i > 0) {
        i -= lanes;
        inline for (0..lanes) |j| {
            const k = lanes - 1 - j;
            try put(t, &st[k], src[i + k], out, &ptr);
        }
    }
    var k: usize = lanes;
    while (k > 0) {
        k -= 1;
        if (ptr < 4) return error.NoRoom;
        ptr -= 4;
        std.mem.writeInt(u32, out[ptr..][0..4], st[k], .little);
    }
    return ptr;
}

/// A slot's symbol with its frequency and start, one load a decoded symbol.
const Entry = extern struct { freq: u16, cum: u16, sym: u8, pad: [3]u8 = @splat(0) };

/// A table's slot lookup (32 KiB), built once a plane.
pub const Decoder = struct {
    t: Table,
    slots: [scale]Entry,

    pub fn init(d: *Decoder, t: Table) void {
        d.t = t;
        for (0..256) |s| @memset(d.slots[t.cum[s]..t.cum[s + 1]], .{ .freq = t.freq[s], .cum = t.cum[s], .sym = @intCast(s) });
    }

    inline fn step(d: *const Decoder, x: *u32) u8 {
        const slot = x.* & (scale - 1);
        const e = d.slots[slot];
        x.* = @as(u32, e.freq) * (x.* >> prob_bits) + slot - e.cum;
        return e.sym;
    }

    /// Decode exactly `out.len` symbols from all of `in`.
    pub fn decode(d: *const Decoder, in: []const u8, out: []u8) Error!void {
        if (in.len < 4 * lanes) return error.Truncated;
        var st: [lanes]u32 = undefined;
        for (&st, 0..) |*x, k| {
            x.* = std.mem.readInt(u32, in[4 * k ..][0..4], .little);
            if (x.* < low or x.* >= low << 8) return error.Corrupt;
        }
        var ptr: usize = 4 * lanes;
        var i: usize = 0;
        // a decoded state is at least 2^11, so it takes at most two bytes back: a group needs at most 2 x lanes
        while (i + lanes <= out.len and ptr + 2 * lanes <= in.len) : (i += lanes) {
            inline for (0..lanes) |k| out[i + k] = d.step(&st[k]);
            inline for (0..lanes) |k| {
                while (st[k] < low) {
                    st[k] = (st[k] << 8) | in[ptr];
                    ptr += 1;
                }
            }
        }
        while (i < out.len) : (i += 1) { // the tail, and any group near the input's end, checked byte by byte
            const x = &st[i % lanes];
            out[i] = d.step(x);
            while (x.* < low) {
                if (ptr >= in.len) return error.Truncated;
                x.* = (x.* << 8) | in[ptr];
                ptr += 1;
            }
        }
        if (ptr != in.len) return error.Corrupt;
        for (st) |x| if (x != low) return error.Corrupt;
    }
};

/// Order-0 entropy of `counts` in bytes (rounded up), the bound an ideal coder approaches; 0 for one symbol.
pub fn entropyBytes(counts: *const [256]u64) u64 {
    var total: u64 = 0;
    for (counts) |c| total += c;
    if (total == 0) return 0;
    const n: f64 = @floatFromInt(total);
    var bits: f64 = 0;
    for (counts) |c| if (c > 0) {
        const x: f64 = @floatFromInt(c);
        bits += x * @log2(n / x);
    };
    return @intFromFloat(@ceil(bits / 8));
}
