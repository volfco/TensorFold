//! LSB-first bit streams for the ClickHouse reference codecs (kv_ch_int.zig, kv_ch_float.zig).
const std = @import("std");

pub const Error = error{ NoRoom, Truncated, Corrupt };

pub const Writer = struct {
    out: []u8,
    at: usize = 0,
    acc: u64 = 0,
    held: u7 = 0,

    /// The low `k` bits of `v`, k at most 32.
    pub fn put(w: *Writer, v: u64, k: u7) Error!void {
        std.debug.assert(k <= 32);
        if (k == 0) return;
        w.acc |= (v & ((@as(u64, 1) << @intCast(k)) - 1)) << @intCast(w.held);
        w.held += k;
        while (w.held >= 8) {
            if (w.at >= w.out.len) return error.NoRoom;
            w.out[w.at] = @truncate(w.acc);
            w.at += 1;
            w.acc >>= 8;
            w.held -= 8;
        }
    }

    /// Whole bytes, after any held bits (padded with zeros to a byte).
    pub fn bytes(w: *Writer, b: []const u8) Error!void {
        try w.flush();
        if (w.out.len - w.at < b.len) return error.NoRoom;
        @memcpy(w.out[w.at..][0..b.len], b);
        w.at += b.len;
    }

    pub fn flush(w: *Writer) Error!void {
        if (w.held > 0) try w.put(0, 8 - (w.held % 8));
    }

    /// The bytes written, held bits padded out.
    pub fn finish(w: *Writer) Error!usize {
        try w.flush();
        return w.at;
    }
};

pub const Reader = struct {
    in: []const u8,
    at: usize = 0,
    acc: u64 = 0,
    held: u7 = 0,

    pub fn get(r: *Reader, k: u7) Error!u64 {
        std.debug.assert(k <= 32);
        if (k == 0) return 0;
        while (r.held < k) {
            if (r.at >= r.in.len) return error.Truncated;
            r.acc |= @as(u64, r.in[r.at]) << @intCast(r.held);
            r.at += 1;
            r.held += 8;
        }
        const v = r.acc & ((@as(u64, 1) << @intCast(k)) - 1);
        r.acc >>= @intCast(k);
        r.held -= k;
        return v;
    }

    /// Drop the partial byte's remaining bits (the writer padded them with zeros, which is checked).
    pub fn alignByte(r: *Reader) Error!void {
        const drop = r.held % 8;
        if (try r.get(drop) != 0) return error.Corrupt;
        if (r.held != 0) return error.Corrupt; // whole bytes still held: the stream ran past its end
    }

    pub fn bytes(r: *Reader, out: []u8) Error!void {
        try r.alignByte();
        if (r.in.len - r.at < out.len) return error.Truncated;
        @memcpy(out, r.in[r.at..][0..out.len]);
        r.at += out.len;
    }

    /// The whole input consumed, no stray bits.
    pub fn end(r: *Reader) Error!void {
        try r.alignByte();
        if (r.at != r.in.len) return error.Corrupt;
    }
};

/// Little-endian element `i` of `w` bytes (1, 2, 4).
pub fn load(b: []const u8, w: u8, i: usize) u32 {
    return switch (w) {
        1 => b[i],
        2 => std.mem.readInt(u16, b[2 * i ..][0..2], .little),
        4 => std.mem.readInt(u32, b[4 * i ..][0..4], .little),
        else => unreachable,
    };
}

pub fn store(b: []u8, w: u8, i: usize, v: u32) void {
    switch (w) {
        1 => b[i] = @truncate(v),
        2 => std.mem.writeInt(u16, b[2 * i ..][0..2], @truncate(v), .little),
        4 => std.mem.writeInt(u32, b[4 * i ..][0..4], v, .little),
        else => unreachable,
    }
}

pub fn mask(w: u8) u32 {
    return if (w == 4) 0xFFFF_FFFF else (@as(u32, 1) << @intCast(8 * @as(u32, w))) - 1;
}
