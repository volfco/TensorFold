//! kv-codec-bench's ClickHouse legs: the common column codecs of clickhouse.com/docs/reference/statements/create/table/codec
//! as ClickHouse composes them (a preparation codec, then a general-purpose one), a block at a time, one thread.
//! LZ4, LZ4HC and ZSTD need -Dkv-bench-libs (the system's libzstd and liblz4); the rest are in-tree references.
const std = @import("std");
const kv = @import("kv");
const options = @import("kv_bench_options");
const ci = kv.ch_int;
const cf = kv.ch_float;

pub const Prep = enum { none, delta, gcd, t64 };
pub const Back = enum { store, lz4, lz4hc9, zstd1, zstd3, double_delta, gorilla, fpc12, alp_rd };
pub const Pipe = struct { name: []const u8, prep: Prep = .none, back: Back };

const lib_pipes = [_]Pipe{
    .{ .name = "ch-lz4", .back = .lz4 },
    .{ .name = "ch-lz4hc9", .back = .lz4hc9 },
    .{ .name = "ch-zstd1", .back = .zstd1 },
    .{ .name = "ch-zstd3", .back = .zstd3 },
    .{ .name = "ch-delta-lz4", .prep = .delta, .back = .lz4 },
    .{ .name = "ch-delta-zstd1", .prep = .delta, .back = .zstd1 },
    .{ .name = "ch-gcd-zstd1", .prep = .gcd, .back = .zstd1 },
    .{ .name = "ch-t64-lz4", .prep = .t64, .back = .lz4 },
    .{ .name = "ch-t64-zstd1", .prep = .t64, .back = .zstd1 },
};
const native_pipes = [_]Pipe{
    .{ .name = "ch-doubledelta", .back = .double_delta },
    .{ .name = "ch-gorilla", .back = .gorilla },
    .{ .name = "ch-fpc12", .back = .fpc12 },
    .{ .name = "ch-alp-rd", .back = .alp_rd },
};
pub const pipes = (if (options.system_codecs) lib_pipes else [_]Pipe{}) ++ native_pipes;

extern "c" fn ZSTD_compress(dst: [*]u8, cap: usize, src: [*]const u8, n: usize, level: c_int) usize;
extern "c" fn ZSTD_decompress(dst: [*]u8, cap: usize, src: [*]const u8, n: usize) usize;
extern "c" fn ZSTD_isError(code: usize) c_uint;
extern "c" fn LZ4_compress_default(src: [*]const u8, dst: [*]u8, n: c_int, cap: c_int) c_int;
extern "c" fn LZ4_compress_HC(src: [*]const u8, dst: [*]u8, n: c_int, cap: c_int, level: c_int) c_int;
extern "c" fn LZ4_decompress_safe(src: [*]const u8, dst: [*]u8, n: c_int, cap: c_int) c_int;

const slack = 64 << 10;

/// One stream through one pipeline; each block's payload has a slot of twice its bytes plus slack.
pub const Run = struct {
    p: Pipe,
    w: u8, // the column's value width: 1, 2 or 4 bytes
    raw: []const u8,
    cuts: []const kv.codec.Cut,
    mid: []u8, // a block's preparation output
    store: []u8,
    lens: []usize, // a block's stored bytes (0: stored raw)
    mids: []usize, // a block's preparation bytes
    back: []u8,
    tables: []u32, // FPC's predictor tables (2 << 12)
    counts: []u32, // ALP RD's left-value counts (1 << 16)

    fn slot(r: *const Run, i: usize) []u8 {
        const x = r.cuts[i];
        return r.store[2 * x.off + i * slack ..][0 .. 2 * x.len + slack];
    }

    fn prep(r: *Run, raw: []const u8) ![]const u8 {
        const n = switch (r.p.prep) {
            .none => return raw,
            .delta => blk: {
                ci.delta(r.w, raw, r.mid[0..raw.len]);
                break :blk raw.len;
            },
            .gcd => try ci.gcd(r.w, raw, r.mid),
            .t64 => try ci.t64(r.w, raw, r.mid),
        };
        return r.mid[0..n];
    }

    fn unprep(r: *Run, mid: []u8, out: []u8) !void {
        switch (r.p.prep) {
            .none => unreachable,
            .delta => {
                @memcpy(out, mid[0..out.len]);
                ci.undelta(r.w, out);
            },
            .gcd => try ci.unGcd(r.w, mid, out),
            .t64 => try ci.unT64(r.w, mid, out),
        }
    }

    fn alpWidth(r: *const Run) u8 {
        return if (r.w == 1) 2 else r.w; // ALP is Float32/Float64 only; bytes are read as 16-bit words
    }

    fn compress(r: *Run, src: []const u8, dst: []u8) !?usize {
        switch (r.p.back) {
            .store => {
                @memcpy(dst[0..src.len], src);
                return src.len;
            },
            .double_delta => return ci.doubleDelta(r.w, src, dst) catch null,
            .gorilla => return cf.gorilla(r.w, src, dst) catch null,
            .fpc12 => return cf.fpc(12, r.tables, src, dst) catch null, // float_size 4: BF16 and byte columns read as Float32 words
            .alp_rd => return cf.alpRd(r.alpWidth(), src, dst, r.counts) catch null,
            else => if (comptime options.system_codecs) switch (r.p.back) {
                .lz4 => {
                    const n = LZ4_compress_default(src.ptr, dst.ptr, @intCast(src.len), @intCast(dst.len));
                    return if (n <= 0) null else @intCast(n);
                },
                .lz4hc9 => {
                    const n = LZ4_compress_HC(src.ptr, dst.ptr, @intCast(src.len), @intCast(dst.len), 9);
                    return if (n <= 0) null else @intCast(n);
                },
                .zstd1, .zstd3 => {
                    const n = ZSTD_compress(dst.ptr, dst.len, src.ptr, src.len, if (r.p.back == .zstd1) 1 else 3);
                    return if (ZSTD_isError(n) != 0) null else n;
                },
                else => unreachable,
            } else unreachable,
        }
    }

    fn decompress(r: *Run, src: []const u8, dst: []u8) !usize {
        switch (r.p.back) {
            .store => unreachable,
            .double_delta => try ci.unDoubleDelta(r.w, src, dst),
            .gorilla => try cf.unGorilla(r.w, src, dst),
            .fpc12 => try cf.unFpc(12, r.tables, src, dst),
            .alp_rd => try cf.unAlpRd(r.alpWidth(), src, dst),
            else => if (comptime options.system_codecs) switch (r.p.back) {
                .lz4, .lz4hc9 => {
                    const n = LZ4_decompress_safe(src.ptr, dst.ptr, @intCast(src.len), @intCast(dst.len));
                    if (n < 0) return error.Corrupt;
                    return @intCast(n);
                },
                .zstd1, .zstd3 => {
                    const n = ZSTD_decompress(dst.ptr, dst.len, src.ptr, src.len);
                    if (ZSTD_isError(n) != 0) return error.Corrupt;
                    return n;
                },
                else => unreachable,
            } else unreachable,
        }
        return dst.len;
    }

    pub fn encode(r: *Run) anyerror!void {
        for (r.cuts, 0..) |x, i| {
            const raw = r.raw[x.off..][0..x.len];
            const s = r.slot(i);
            const mid = try r.prep(raw);
            r.mids[i] = mid.len;
            const n = try r.compress(mid, s);
            if (n != null and n.? < raw.len) r.lens[i] = n.? else {
                r.lens[i] = 0; // no gain: the block is stored raw, as ClickHouse's adaptive mode would
                @memcpy(s[0..raw.len], raw);
            }
        }
    }

    pub fn decode(r: *Run) anyerror!void {
        for (r.cuts, 0..) |x, i| {
            const out = r.back[x.off..][0..x.len];
            const s = r.slot(i);
            if (r.lens[i] == 0) {
                @memcpy(out, s[0..x.len]);
                continue;
            }
            if (r.p.prep == .none) {
                if (try r.decompress(s[0..r.lens[i]], out) != out.len) return error.Corrupt;
            } else {
                const mid = r.mid[0..r.mids[i]];
                if (try r.decompress(s[0..r.lens[i]], mid) != mid.len) return error.Corrupt;
                try r.unprep(mid, out);
            }
        }
    }

    pub fn stored(r: *const Run) usize {
        var n: usize = 0;
        for (r.cuts, r.lens) |x, len| n += if (len == 0) x.len else len;
        return n;
    }

    pub fn rawBlocks(r: *const Run) usize {
        var n: usize = 0;
        for (r.lens) |len| n += @intFromBool(len == 0);
        return n;
    }
};
