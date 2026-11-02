//! Phase 0 of the KV-cache compression plan: a kept state's bytes by class, the producer bit patterns that would allow
//! exact packing, high-byte distributions, and the modeled order-0 ceiling the lossless codec is judged against.
const std = @import("std");
const codec = @import("kv_codec.zig");

/// What a stretch of a kept state holds. GLM: the arena head (KDA recurrent state, conv history and other fields up
/// to the first cache), MLA latent rows, their FP8 scales, the decode rings, DSA pooled-key rows, and the arena tail.
pub const Class = enum(u8) { prefix, latent, scale, ring, pooled, tail };
pub const classes = @typeInfo(Class).@"enum".field_names.len;

pub const Entry = struct {
    bytes: u64 = 0,
    counts: codec.Counts = .{}, // positions mod 4 from each span's start
    words32: u64 = 0, // whole little-endian u32 words seen
    low16_zero: u64 = 0, // ... whose low 16 bits are zero (an FP32 widened from BF16)
    low24_zero: u64 = 0, // ... whose low 24 bits are zero (an FP32 holding an FP8-sized mantissa)
    words16: u64 = 0,
    low8_zero: u64 = 0, // u16 words whose low byte is zero (a BF16 holding at most an FP8 mantissa)
    nonfinite32: u64 = 0, // u32 words with an all-ones FP32 exponent
    nonfinite16: u64 = 0, // u16 words with an all-ones BF16 exponent
    spans: u64 = 0,
};

pub const Ledger = struct {
    e: [classes]Entry = @splat(.{}),

    pub fn observe(l: *Ledger, class: Class, bytes: []const u8) void {
        const x = &l.e[@intFromEnum(class)];
        x.bytes += bytes.len;
        x.spans += 1;
        x.counts.add(bytes);
        var i: usize = 0;
        while (i + 4 <= bytes.len) : (i += 4) {
            const w = std.mem.readInt(u32, bytes[i..][0..4], .little);
            x.words32 += 1;
            x.low16_zero += @intFromBool(w & 0xFFFF == 0);
            x.low24_zero += @intFromBool(w & 0xFF_FFFF == 0);
            x.nonfinite32 += @intFromBool(w & 0x7F80_0000 == 0x7F80_0000);
        }
        i = 0;
        while (i + 2 <= bytes.len) : (i += 2) {
            const w = std.mem.readInt(u16, bytes[i..][0..2], .little);
            x.words16 += 1;
            x.low8_zero += @intFromBool(w & 0xFF == 0);
            x.nonfinite16 += @intFromBool(w & 0x7F80 == 0x7F80);
        }
    }

    pub fn total(l: *const Ledger) u64 {
        var n: u64 = 0;
        for (l.e) |x| n += x.bytes;
        return n;
    }

    /// A class's modeled bytes at its best width; whole-class histograms, so a ceiling estimate, not a block's cost.
    pub fn modeled(l: *const Ledger, class: Class) codec.Choice {
        const x = &l.e[@intFromEnum(class)];
        return if (x.bytes == 0) .{ .width = 1, .bytes = 0 } else codec.choose(&x.counts, @intCast(x.bytes));
    }

    pub fn modeledTotal(l: *const Ledger) u64 {
        var n: u64 = 0;
        for (0..classes) |c| n += l.modeled(@enumFromInt(c)).bytes;
        return n;
    }

    /// Gate M0's question: the modeled saving as a fraction of the raw bytes.
    pub fn ceiling(l: *const Ledger) f64 {
        const raw = l.total();
        if (raw == 0) return 0;
        return 1 - @as(f64, @floatFromInt(l.modeledTotal())) / @as(f64, @floatFromInt(raw));
    }

    pub fn add(l: *Ledger, o: *const Ledger) void {
        for (&l.e, o.e) |*x, y| {
            x.bytes += y.bytes;
            x.spans += y.spans;
            for (&x.counts.h, y.counts.h) |*a, b| {
                for (a, b) |*p, q| p.* += q;
            }
            x.words32 += y.words32;
            x.low16_zero += y.low16_zero;
            x.low24_zero += y.low24_zero;
            x.words16 += y.words16;
            x.low8_zero += y.low8_zero;
            x.nonfinite32 += y.nonfinite32;
            x.nonfinite16 += y.nonfinite16;
        }
    }

    /// One line a class, then the totals: KVLEDGER lines a receipt can grep.
    pub fn print(l: *const Ledger, w: *std.Io.Writer, tag: []const u8) !void {
        for (l.e, 0..) |x, c| {
            if (x.bytes == 0) continue;
            const m = l.modeled(@enumFromInt(c));
            try w.print("KVLEDGER {s} class={s} bytes={d} spans={d} modeled={d} width={d} low16_zero={d}/{d} low24_zero={d}/{d} low8_zero={d}/{d} nonfinite32={d} nonfinite16={d}\n", .{
                tag,         @tagName(@as(Class, @enumFromInt(c))), x.bytes,       x.spans, m.bytes,      m.width,
                x.low16_zero, x.words32,                            x.low24_zero,  x.words32, x.low8_zero, x.words16,
                x.nonfinite32, x.nonfinite16,
            });
        }
        try w.print("KVLEDGER {s} total bytes={d} modeled={d} ceiling={d:.4}\n", .{ tag, l.total(), l.modeledTotal(), l.ceiling() });
    }

    /// The high byte of each u16 (BF16 sign and exponent) as a 256-bin histogram, for the distribution check.
    pub fn highBytes(l: *const Ledger, class: Class) [256]u64 {
        return l.e[@intFromEnum(class)].counts.plane(2, 1);
    }
};

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: [*]const u8, size: usize, n: usize, f: *anyopaque) usize;
extern "c" fn fclose(f: *anyopaque) c_int;

/// A raw dump for offline Phase 1 work: `path` gets the bytes, `path`.parts one "class length row" line a span (row:
/// the bytes of one token's row in it, 0 when it holds no rows), the form tf-kv-codec-bench --file reads.
pub const Dump = struct {
    data: *anyopaque,
    parts: *anyopaque,

    pub fn open(path: [:0]const u8, parts_path: [:0]const u8) !Dump {
        const data = fopen(path.ptr, "wb") orelse return error.DumpOpen;
        errdefer _ = fclose(data);
        const parts = fopen(parts_path.ptr, "w") orelse return error.DumpOpen;
        return .{ .data = data, .parts = parts };
    }

    pub fn span(d: *Dump, class: Class, row: u32, bytes: []const u8) !void {
        if (bytes.len > 0 and fwrite(bytes.ptr, 1, bytes.len, d.data) != bytes.len) return error.DumpWrite;
        var line: [64]u8 = undefined;
        const text = try std.fmt.bufPrint(&line, "{s} {d} {d}\n", .{ @tagName(class), bytes.len, row });
        if (fwrite(text.ptr, 1, text.len, d.parts) != text.len) return error.DumpWrite;
    }

    pub fn close(d: *Dump) !void {
        const a = fclose(d.data);
        const b = fclose(d.parts);
        if (a != 0 or b != 0) return error.DumpWrite;
    }
};
