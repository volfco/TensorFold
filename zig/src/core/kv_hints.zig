//! Cache-entry hints: a sidecar file an external post-processor writes next to a kept state (tools/kvprobe/hints.py;
//! format in tools/kvprobe/HINTS.md). The reader trusts nothing: every identity field must equal the entry's, every
//! section must lie inside the file, and the trailing checksum must hold, or it refuses by name before any use.
const std = @import("std");

pub const magic = "TFKVHNT1".*;
pub const version: u32 = 1;
pub const header_bytes: usize = 192;
pub const entry_bytes: usize = 32;
pub const all_layers: u32 = std.math.maxInt(u32);

pub const Kind = enum(u32) {
    bounds = 1, // per DSA layer: [pages][min | max][i_dim] bf16 over each whole page's pooled index keys
    selected = 2, // per DSA layer: [pools] u32, how often the indexer selected each pool
    hot = 3, // all layers: [n] u32 page indices, a restore order (the last round's touched pages first)
    experts = 4, // all layers: [n] { token u32, moe layer u16, count u16, ids [8] u16 }, a token-keyed expert prior
};

pub const expert_entry: usize = 24;

/// What the consumer knows of the entry the sidecar claims to describe.
pub const Expect = struct {
    config: [32]u8, // sha256 of config.json (the rank fingerprint's `config`)
    checkpoint: [32]u8, // the checkpoint digest (the fingerprint's `checkpoint`, as bytes)
    layout: u32, // the kept-state layout version (cuda_snapshot_layout manifest: tfkvplan 1)
    page_rows: u32,
    pool: u32,
    i_dim: u32,
    dsa_layers: u32,
    experts: u32,
    moe_layers: u32,
    token_count: u64,
    tokens_hash: u64, // tokensHash of the entry's prompt ids
};

pub const Error = error{
    HintTruncated,
    HintMagic,
    HintVersion,
    HintHeader,
    HintReserved,
    HintChecksum,
    HintModel,
    HintCheckpoint,
    HintLayout,
    HintPageSize,
    HintShape,
    HintTokenCount,
    HintTokens,
    HintSection,
    HintKind,
    HintLayer,
    HintBounds,
    HintHotPage,
    HintExpert,
};

pub const Section = struct { kind: Kind, layer: u32, count: u32, width: u32, bytes: []const u8 };

pub fn tokensHash(ids: []const u32) u64 {
    var h = std.hash.Wyhash.init(0);
    for (ids) |t| h.update(&std.mem.toBytes(std.mem.nativeToLittle(u32, t)));
    return h.final();
}

fn rd(comptime T: type, b: []const u8, at: usize) T {
    return std.mem.readInt(T, b[at..][0..@sizeOf(T)], .little);
}

fn bf16(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}

/// A validated sidecar: its sections borrow `file`.
pub const Hints = struct {
    file: []const u8,
    sections: []Section,

    pub fn deinit(h: *Hints, gpa: std.mem.Allocator) void {
        gpa.free(h.sections);
    }

    pub fn find(h: *const Hints, kind: Kind, layer: u32) ?Section {
        for (h.sections) |s| if (s.kind == kind and s.layer == layer) return s;
        return null;
    }

    /// A DSA layer's page bounds as f32 [pages][2][i_dim] into `out`, or null when the sidecar has none for it.
    pub fn bounds(h: *const Hints, layer: u32, i_dim: usize, out: []f32) ?usize {
        const s = h.find(.bounds, layer) orelse return null;
        const n = @as(usize, s.count) * 2 * i_dim;
        std.debug.assert(out.len >= n);
        for (0..n) |i| out[i] = bf16(rd(u16, s.bytes, 2 * i));
        return s.count;
    }
};

/// Validate `file` against `want` and index its sections. Refuses by name; allocates only the section index.
pub fn parse(gpa: std.mem.Allocator, file: []const u8, want: Expect) !Hints {
    if (file.len < header_bytes + 8) return error.HintTruncated;
    if (!std.mem.eql(u8, file[0..8], &magic)) return error.HintMagic;
    if (rd(u32, file, 8) != version) return error.HintVersion;
    if (rd(u32, file, 12) != header_bytes) return error.HintHeader;
    const total = rd(u64, file, 120);
    if (total != file.len) return error.HintTruncated;
    for (file[136..header_bytes]) |b| if (b != 0) return error.HintReserved;
    if (rd(u64, file, file.len - 8) != std.hash.Wyhash.hash(0, file[0 .. file.len - 8])) return error.HintChecksum;
    if (!std.mem.eql(u8, file[56..88], &want.config)) return error.HintModel;
    if (!std.mem.eql(u8, file[88..120], &want.checkpoint)) return error.HintCheckpoint;
    if (rd(u32, file, 16) != want.layout) return error.HintLayout;
    if (rd(u32, file, 20) != want.page_rows or rd(u32, file, 24) != want.pool) return error.HintPageSize;
    if (rd(u32, file, 28) != want.i_dim or rd(u32, file, 32) != want.dsa_layers or rd(u32, file, 128) != want.experts or
        rd(u32, file, 132) != want.moe_layers) return error.HintShape;
    if (rd(u64, file, 40) != want.token_count) return error.HintTokenCount;
    if (rd(u64, file, 48) != want.tokens_hash) return error.HintTokens;
    const n = rd(u32, file, 36);
    const table_end = header_bytes + @as(u64, n) * entry_bytes;
    if (table_end > file.len - 8) return error.HintSection;
    const out = try gpa.alloc(Section, n);
    errdefer gpa.free(out);
    var last: u64 = table_end;
    const pages = want.token_count / want.page_rows;
    for (out, 0..) |*s, i| {
        const e = header_bytes + i * entry_bytes;
        const kind = std.enums.fromInt(Kind, rd(u32, file, e)) orelse return error.HintKind;
        const layer = rd(u32, file, e + 4);
        const count = rd(u32, file, e + 8);
        const width = rd(u32, file, e + 12);
        const off = rd(u64, file, e + 16);
        const len = rd(u64, file, e + 24);
        // in order, 8-aligned, inside the payload area, and exactly count x width bytes
        if (off < last or off % 8 != 0 or off > file.len - 8 or len > file.len - 8 - off) return error.HintSection;
        if (@as(u64, count) * width != len) return error.HintSection;
        last = off + len;
        const bytes = file[@intCast(off)..][0..@intCast(len)];
        switch (kind) {
            .bounds, .selected => if (layer >= want.dsa_layers) return error.HintLayer,
            .hot, .experts => if (layer != all_layers) return error.HintLayer,
        }
        switch (kind) {
            .bounds => {
                if (width != 4 * want.i_dim or count > pages) return error.HintSection;
                try checkBounds(bytes, count, want.i_dim);
            },
            .selected => if (width != 4 or count != want.token_count / want.pool) return error.HintSection,
            .hot => {
                if (width != 4) return error.HintSection;
                const held = std.math.divCeil(u64, want.token_count, want.page_rows) catch unreachable;
                for (0..count) |k| if (rd(u32, bytes, 4 * k) >= held) return error.HintHotPage;
            },
            .experts => {
                if (width != expert_entry) return error.HintSection;
                for (0..count) |k| {
                    const x = bytes[k * expert_entry ..][0..expert_entry];
                    if (rd(u16, x, 4) >= want.moe_layers or rd(u16, x, 6) > 8) return error.HintExpert;
                    for (0..rd(u16, x, 6)) |m| if (rd(u16, x, 8 + 2 * m) >= want.experts) return error.HintExpert;
                }
            },
        }
        s.* = .{ .kind = kind, .layer = layer, .count = count, .width = width, .bytes = bytes };
    }
    return .{ .file = file, .sections = out };
}

/// Each dimension's min at most its max, both finite: a bound a prune can rely on.
fn checkBounds(bytes: []const u8, pages: u32, i_dim: u32) !void {
    for (0..pages) |p| for (0..i_dim) |d| {
        const lo = bf16(rd(u16, bytes, 2 * ((p * 2) * i_dim + d)));
        const hi = bf16(rd(u16, bytes, 2 * ((p * 2 + 1) * i_dim + d)));
        if (!std.math.isFinite(lo) or !std.math.isFinite(hi) or lo > hi) return error.HintBounds;
    };
}

/// The writer the post-processor's format follows (tools/kvprobe/hints.py writes the same bytes); tests use it.
pub const Writer = struct {
    head: [header_bytes]u8 = @splat(0),
    table: std.ArrayList(u8) = .empty,
    body: std.ArrayList(u8) = .empty,
    count: u32 = 0,

    pub fn init(id: Expect) Writer {
        var w: Writer = .{};
        @memcpy(w.head[0..8], &magic);
        inline for (.{ .{ 8, version }, .{ 12, @as(u32, header_bytes) }, .{ 16, id.layout }, .{ 20, id.page_rows }, .{ 24, id.pool }, .{ 28, id.i_dim }, .{ 32, id.dsa_layers }, .{ 128, id.experts }, .{ 132, id.moe_layers } }) |f|
            std.mem.writeInt(u32, w.head[f[0]..][0..4], f[1], .little);
        std.mem.writeInt(u64, w.head[40..48], id.token_count, .little);
        std.mem.writeInt(u64, w.head[48..56], id.tokens_hash, .little);
        @memcpy(w.head[56..88], &id.config);
        @memcpy(w.head[88..120], &id.checkpoint);
        return w;
    }

    pub fn deinit(w: *Writer, gpa: std.mem.Allocator) void {
        w.table.deinit(gpa);
        w.body.deinit(gpa);
    }

    pub fn section(w: *Writer, gpa: std.mem.Allocator, kind: Kind, layer: u32, count: u32, width: u32, bytes: []const u8) !void {
        std.debug.assert(bytes.len == @as(usize, count) * width);
        var e: [entry_bytes]u8 = undefined;
        std.mem.writeInt(u32, e[0..4], @intFromEnum(kind), .little);
        std.mem.writeInt(u32, e[4..8], layer, .little);
        std.mem.writeInt(u32, e[8..12], count, .little);
        std.mem.writeInt(u32, e[12..16], width, .little);
        std.mem.writeInt(u64, e[16..24], w.body.items.len, .little); // relative until finish
        std.mem.writeInt(u64, e[24..32], bytes.len, .little);
        try w.table.appendSlice(gpa, &e);
        try w.body.appendSlice(gpa, bytes);
        try w.body.appendNTimes(gpa, 0, (8 - w.body.items.len % 8) % 8);
        w.count += 1;
    }

    /// The whole file: header, table (offsets made absolute), payloads, checksum.
    pub fn finish(w: *Writer, gpa: std.mem.Allocator) ![]u8 {
        const base = header_bytes + w.table.items.len;
        const total = base + w.body.items.len + 8;
        std.mem.writeInt(u32, w.head[36..40], w.count, .little);
        std.mem.writeInt(u64, w.head[120..128], total, .little);
        const out = try gpa.alloc(u8, total);
        @memcpy(out[0..header_bytes], &w.head);
        @memcpy(out[header_bytes..base], w.table.items);
        for (0..w.count) |i| {
            const at = header_bytes + i * entry_bytes + 16;
            std.mem.writeInt(u64, out[at..][0..8], rd(u64, out, at) + base, .little);
        }
        @memcpy(out[base..][0..w.body.items.len], w.body.items);
        std.mem.writeInt(u64, out[total - 8 ..][0..8], std.hash.Wyhash.hash(0, out[0 .. total - 8]), .little);
        return out;
    }
};
