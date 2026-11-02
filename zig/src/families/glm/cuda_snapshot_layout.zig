//! A compact prefix of a rank's committed GLM state; live sequence addresses and kernel layouts stay fixed.
const std = @import("std");
const State = @import("cuda_state.zig").State;
const page_rows = @import("cuda_state.zig").page_rows;
const Config = @import("config.zig").Config;
const cache = @import("cuda_dsa_block.zig");
const sparse = @import("cuda_sparse.zig");
pub const Class = @import("core").kv_ledger.Class;

/// `len` bytes from `live`, or on a paged sequence's pages `span` bytes a page, the pages `page` bytes apart.
/// `row`: the bytes of one token's row in it (0: not rows), the column codecs' token-axis lag.
/// `cache`: the DSA cache the part's rows belong to (the MTP layer's is the last), `no_cache` for the arena's head and tail.
pub const Part = struct { live: u64, len: usize, span: usize = 0, page: u64 = 0, class: Class = .prefix, row: u32 = 0, cache: u16 = no_cache };
pub const no_cache = std.math.maxInt(u16);
pub const Plan = struct {
    parts: [5 * (@import("config.zig").max_layers + 1) + 2]Part = undefined,
    count: usize = 0,
    len: usize = 0,
    ids: []const u32 = &.{}, // a paged sequence's pages in row order
    sizing: bool = false, // `size`: the rows counted, not read, so no page need be mapped
    replay: u32 = 0, // TF_GLM_KDA_REPLAY: kept without its arena head, rebuilt from this many tokens (0: head kept)

    pub fn init(st: *const State, c: *const Config, at: usize, head: ?usize) !Plan {
        return make(st, c, at, head, false);
    }

    /// A prefix's bytes at `at`, rows to `capacity` counted held or not (a paged sequence holds its request's alone).
    pub fn size(st: *const State, c: *const Config, capacity: usize, at: usize, head: ?usize) !usize {
        var whole = st.*;
        whole.capacity = capacity;
        return (try make(&whole, c, at, head, true)).len;
    }

    fn make(st: *const State, c: *const Config, at: usize, head: ?usize, sizing: bool) !Plan {
        if (at > st.capacity or st.caches.len > @import("config.zig").max_layers) return error.SnapshotShape;
        if (st.rings.len != 2 * (st.caches.len + 1)) return error.SnapshotShape;
        var p: Plan = .{ .ids = st.pages, .sizing = sizing };
        // a paged sequence's caches are its pages: its fields up to them end at its first ring
        const first = if (st.table != 0) st.rings[0] else if (st.caches.len > 0) st.caches[0].lc else st.mtp_caches.lc;
        if (first < st.arena.buf.ptr) return error.SnapshotShape;
        try p.add(st, st.arena.buf.ptr, @intCast(first - st.arena.buf.ptr), .prefix);
        for (st.caches, 0..) |x, j| try p.rows(st, c, x, j, at);
        if (head) |n| {
            if (n > st.capacity) return error.SnapshotShape;
            try p.rows(st, c, st.mtp_caches, st.caches.len, n);
        }
        if (st.mtp_pos_dev < st.arena.buf.ptr or st.mtp_pos_dev - st.arena.buf.ptr > st.arena.buf.len)
            return error.SnapshotShape;
        const end = if (st.table != 0) st.table else st.arena.buf.ptr + st.arena.buf.len; // its table stays its own
        if (end < st.mtp_pos_dev) return error.SnapshotShape;
        try p.add(st, st.mtp_pos_dev, @intCast(end - st.mtp_pos_dev), .tail);
        return p;
    }

    fn rows(p: *Plan, st: *const State, c: *const Config, x: cache.Caches, j: usize, at: usize) !void {
        const latent = try std.math.mul(usize, at, @as(usize, c.kv_lora) * @as(usize, if (x.ls == 0) 2 else 1));
        const ring = try std.math.mul(usize, @min(at, st.ring_rows), st.ring_row);
        const pooled = try std.math.mul(usize, @min(at / sparse.pool + 2, x.pk_rows), @as(usize, c.i_dim) * 2);
        try p.addCache(st, x, x.lc, latent, page_rows * @as(usize, c.kv_lora) * 2, .latent);
        p.rowed(@intCast(@as(usize, c.kv_lora) * @as(usize, if (x.ls == 0) 2 else 1)), j);
        if (x.ls != 0) try p.add(st, x.ls, try std.math.mul(usize, at, @sizeOf(f32)), .scale);
        p.rowed(@sizeOf(f32), j);
        try p.add(st, st.rings[2 * j], ring, .ring);
        p.rowed(@intCast(st.ring_row), j);
        try p.add(st, st.rings[2 * j + 1], ring, .ring);
        p.rowed(@intCast(st.ring_row), j);
        try p.addCache(st, x, x.pk, pooled, page_rows / sparse.pool * @as(usize, c.i_dim) * 2, .pooled);
        p.rowed(@intCast(@as(usize, c.i_dim) * 2), j);
    }

    /// The part just added holds rows of `bytes` of cache `j` (a part of no bytes was not added, so nothing is marked).
    fn rowed(p: *Plan, bytes: u32, j: usize) void {
        if (p.count == 0) return;
        const last = &p.parts[p.count - 1];
        if (last.row != 0 or last.class == .prefix) return;
        last.row = bytes;
        last.cache = @intCast(j);
    }

    /// A cache's rows: in the arena, or on the sequence's pages, `span` bytes a page.
    fn addCache(p: *Plan, st: *const State, x: cache.Caches, live: u64, len: usize, span: usize, class: Class) !void {
        if (x.pt == 0) return p.add(st, live, len, class);
        if (len == 0) return;
        if ((!p.sizing and (len - 1) / span >= p.ids.len) or p.count == p.parts.len) return error.SnapshotShape;
        p.len = try std.math.add(usize, p.len, len);
        p.parts[p.count] = .{ .live = live, .len = len, .span = span, .page = x.ps, .class = class };
        p.count += 1;
    }

    /// Where a part's bytes from `done` on sit, as many as lie in a row there.
    pub fn piece(p: *const Plan, part: Part, done: usize) Part {
        if (part.span == 0) return .{ .live = part.live + done, .len = part.len - done };
        const off = done % part.span;
        return .{ .live = part.live + @as(u64, p.ids[done / part.span]) * part.page + off, .len = @min(part.span - off, part.len - done) };
    }

    fn add(p: *Plan, st: *const State, live: u64, len: usize, class: Class) !void {
        if (len == 0) return;
        if (live < st.arena.buf.ptr) return error.SnapshotShape;
        const off = live - st.arena.buf.ptr;
        if (off > st.arena.buf.len or len > st.arena.buf.len - off or p.count == p.parts.len)
            return error.SnapshotShape;
        p.len = try std.math.add(usize, p.len, len);
        p.parts[p.count] = .{ .live = live, .len = len, .class = class };
        p.count += 1;
    }
};

/// TF_GLM_KV_DUMP's third file beside a keep's bytes and cuts (tools/kvprobe/kvdump.py reads it): the plan's parts
/// before cutting, the arena head's fields, the KDA parity and ring bases, so a dump reads back as tensors.
pub fn manifest(p: *const Plan, st: *const State, c: *const Config, rank: usize, world: usize, w: *std.Io.Writer) !void {
    const base = st.arena.buf.ptr;
    try w.print("tfkvplan 1\nrank {d} world {d} pos {d} mtp_len {d} capacity {d} base {d} mtp_base {d} paged {d} len {d}\n", .{
        rank, world, st.pos, st.mtp_len, st.capacity, st.base, st.mtp_base, @intFromBool(st.table != 0), p.len,
    });
    try w.print("shape kv_lora {d} i_dim {d} hidden {d} kda_layers {d} heads {d} dk 128 dv 128 conv_c {d} rec_layer {d} proj_row {d} ring_rows {d} ring_row {d} page_rows {d} pool {d} caches {d}\n", .{
        c.kv_lora, c.i_dim, c.hidden, st.kda_layers, st.heads, st.conv_c, st.rec_layer, st.proj_row, st.ring_rows, st.ring_row, page_rows, sparse.pool, st.caches.len,
    });
    try w.writeAll("cur ");
    for (st.cur) |b| try w.writeByte('0' + @as(u8, b));
    try w.writeByte('\n');
    if (p.replay != 0) try w.print("replay {d}\n", .{p.replay}); // no arena head: its KDA states are rebuilt
    const window = @import("cuda_state.zig").window_rows;
    const fields = .{
        .{ "pos_dev", st.pos_dev, 4 },
        .{ "conv", st.conv, st.kda_layers * 3 * st.conv_c * 2 },
        .{ "rec", st.rec, 2 * st.kda_layers * st.rec_layer },
        .{ "proj", st.proj, st.kda_layers * window * st.proj_row * 2 },
        .{ "save_k", st.save_k, st.kda_layers * window * st.heads * 128 * 4 },
        .{ "save_v", st.save_v, st.kda_layers * window * st.heads * 128 * 2 },
        .{ "save_g", st.save_g, st.kda_layers * window * st.heads * 128 * 4 },
        .{ "save_b", st.save_b, st.kda_layers * window * st.heads * 4 },
    };
    inline for (fields) |f| {
        if (f[1] < base) return error.SnapshotShape;
        try w.print("field {s} {d} {d}\n", .{ f[0], f[1] - base, f[2] });
    }
    for (st.caches, 0..) |x, j| try w.print("cache {d} kv8 {d} pk_rows {d}\n", .{ j, @intFromBool(x.ls != 0), x.pk_rows });
    try w.print("cache {d} kv8 {d} pk_rows {d} mtp\n", .{ st.caches.len, @intFromBool(st.mtp_caches.ls != 0), st.mtp_caches.pk_rows });
    for (p.parts[0..p.count], 0..) |part, i| {
        if (part.cache == no_cache) {
            try w.print("part {d} {s} {d} {d} -\n", .{ i, @tagName(part.class), part.len, part.row });
        } else try w.print("part {d} {s} {d} {d} {d}\n", .{ i, @tagName(part.class), part.len, part.row, part.cache });
    }
    try w.writeAll("end\n");
}
