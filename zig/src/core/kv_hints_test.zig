//! kv_hints: a sidecar the writer made reads back, and every identity, bound and section fault is refused by name.
const std = @import("std");
const hints = @import("kv_hints.zig");
const t = std.testing;

// 1026 tokens: two whole 512-row pages and a partial third
const ids: [1026]u32 = blk: {
    @setEvalBranchQuota(4000);
    var a: [1026]u32 = undefined;
    for (&a, 0..) |*x, i| x.* = @intCast(7 + i % 3);
    break :blk a;
};
fn expect() hints.Expect {
    return .{ .config = @splat(0xC0), .checkpoint = @splat(0xC7), .layout = 1, .page_rows = 512, .pool = 4, .i_dim = 2, .dsa_layers = 11, .experts = 288, .moe_layers = 42, .token_count = ids.len, .tokens_hash = hints.tokensHash(&ids) };
}

fn bf(x: f32) [2]u8 {
    return std.mem.toBytes(std.mem.nativeToLittle(u16, @truncate(@as(u32, @bitCast(x)) >> 16)));
}

/// The golden sidecar tools/kvprobe/test_hints.py rebuilds byte for byte.
fn golden(gpa: std.mem.Allocator) ![]u8 {
    var w = hints.Writer.init(expect());
    defer w.deinit(gpa);
    var b: [2 * 2 * 2 * 2]u8 = undefined; // two pages: min then max, i_dim 2
    for ([_]f32{ -1, -2, 1, 2, 0, -0.5, 0.25, 0.5 }, 0..) |x, i| @memcpy(b[2 * i ..][0..2], &bf(x));
    try w.section(gpa, .bounds, 3, 2, 8, &b);
    var sel: [256 * 4]u8 = @splat(0); // 1026 / 4 pools
    sel[0] = 5;
    try w.section(gpa, .selected, 3, 256, 4, &sel);
    const hot = [2]u32{ std.mem.nativeToLittle(u32, 1), 0 };
    try w.section(gpa, .hot, hints.all_layers, 2, 4, std.mem.sliceAsBytes(&hot));
    var e: [hints.expert_entry]u8 = @splat(0);
    std.mem.writeInt(u32, e[0..4], 9, .little);
    std.mem.writeInt(u16, e[4..6], 41, .little);
    std.mem.writeInt(u16, e[6..8], 2, .little);
    std.mem.writeInt(u16, e[8..10], 287, .little);
    std.mem.writeInt(u16, e[10..12], 3, .little);
    try w.section(gpa, .experts, hints.all_layers, 1, hints.expert_entry, &e);
    return w.finish(gpa);
}

test "KV hints: the golden sidecar reads back, bounds decode and the format hash is pinned" {
    const f = try golden(t.allocator);
    defer t.allocator.free(f);
    var h = try hints.parse(t.allocator, f, expect());
    defer h.deinit(t.allocator);
    try t.expectEqual(@as(usize, 4), h.sections.len);
    var out: [8]f32 = undefined;
    try t.expectEqual(@as(?usize, 2), h.bounds(3, 2, &out));
    try t.expectEqualSlices(f32, &.{ -1, -2, 1, 2, 0, -0.5, 0.25, 0.5 }, &out);
    try t.expectEqual(@as(?usize, null), h.bounds(4, 2, &out));
    try t.expect(h.find(.hot, hints.all_layers) != null);
    // the Python writer (tools/kvprobe/hints.py) must produce these exact bytes: test_hints.py pins the same hash
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(f, &d, .{});
    try t.expectEqualStrings("0b40a058ad3fbd9b9736ab2cf398efc6e4de94228b3d3dda23db28214ddb1df1", &std.fmt.bytesToHex(d, .lower));
}

fn refused(want: anyerror, f: []const u8, e: hints.Expect) !void {
    try t.expectError(want, hints.parse(t.allocator, f, e));
}

/// A copy of the golden file with `at` set to `v`, its checksum refreshed so only the named check can fire.
fn patched(f: []const u8, comptime T: type, at: usize, v: T) ![]u8 {
    const g = try t.allocator.dupe(u8, f);
    std.mem.writeInt(T, g[at..][0..@sizeOf(T)], v, .little);
    std.mem.writeInt(u64, g[g.len - 8 ..][0..8], std.hash.Wyhash.hash(0, g[0 .. g.len - 8]), .little);
    return g;
}

test "KV hints: every identity mismatch is refused by its name" {
    const f = try golden(t.allocator);
    defer t.allocator.free(f);
    inline for (.{
        .{ "config", @as([32]u8, @splat(0xC1)), error.HintModel },
        .{ "checkpoint", @as([32]u8, @splat(0)), error.HintCheckpoint },
        .{ "layout", 2, error.HintLayout },
        .{ "page_rows", 256, error.HintPageSize },
        .{ "pool", 8, error.HintPageSize },
        .{ "i_dim", 128, error.HintShape },
        .{ "experts", 256, error.HintShape },
        .{ "token_count", 1025, error.HintTokenCount },
        .{ "tokens_hash", 1, error.HintTokens },
    }) |c| {
        var e = expect();
        @field(e, c[0]) = c[1];
        try refused(c[2], f, e);
    }
}

test "KV hints: a damaged file is refused by name before any section is used" {
    const f = try golden(t.allocator);
    defer t.allocator.free(f);
    try refused(error.HintTruncated, f[0..100], expect());
    try refused(error.HintTruncated, f[0 .. f.len - 8], expect()); // its recorded length no longer matches
    var g = try t.allocator.dupe(u8, f);
    defer t.allocator.free(g);
    g[0] = 'X';
    try refused(error.HintMagic, g, expect());
    g[0] = f[0];
    g[g.len - 20] ^= 1; // a payload byte: only the checksum sees it
    try refused(error.HintChecksum, g, expect());
    const cases = .{
        .{ u32, 8, 2, error.HintVersion },
        .{ u32, 12, 200, error.HintHeader },
        .{ u32, 140, 1, error.HintReserved },
        .{ u32, 36, 5000, error.HintSection }, // more sections than the file holds
        .{ u32, hints.header_bytes, 9, error.HintKind },
        .{ u32, hints.header_bytes + 4, 11, error.HintLayer }, // bounds for a DSA layer past the last
        .{ u32, hints.header_bytes + 2 * 32 + 4, 0, error.HintLayer }, // a hot list must name all layers
        .{ u32, hints.header_bytes + 8, 3, error.HintSection }, // count x width no longer the bytes
        .{ u64, hints.header_bytes + 32 + 16, 1 << 40, error.HintSection }, // payload past the end
    };
    inline for (cases) |c| {
        const h = try patched(f, c[0], c[1], c[2]);
        defer t.allocator.free(h);
        try refused(c[3], h, expect());
    }
}

test "KV hints: bounds, hot pages and expert ids are range-checked" {
    const f = try golden(t.allocator);
    defer t.allocator.free(f);
    const base = std.mem.readInt(u64, f[hints.header_bytes + 16 ..][0..8], .little); // the bounds payload
    // page 0's min of dimension 0 above its max (1.0): a bound a prune would wrongly trust
    const lo = try patched(f, u16, @intCast(base), 0x4000); // 2.0
    defer t.allocator.free(lo);
    try refused(error.HintBounds, lo, expect());
    const nan = try patched(f, u16, @intCast(base + 2), 0x7FC0);
    defer t.allocator.free(nan);
    try refused(error.HintBounds, nan, expect());
    const hot = std.mem.readInt(u64, f[hints.header_bytes + 2 * 32 + 16 ..][0..8], .little);
    const page = try patched(f, u32, @intCast(hot), 3); // 1026 tokens hold pages 0 to 2
    defer t.allocator.free(page);
    try refused(error.HintHotPage, page, expect());
    const ex = std.mem.readInt(u64, f[hints.header_bytes + 3 * 32 + 16 ..][0..8], .little);
    const id = try patched(f, u16, @intCast(ex + 8), 288);
    defer t.allocator.free(id);
    try refused(error.HintExpert, id, expect());
    const layer = try patched(f, u16, @intCast(ex + 4), 42);
    defer t.allocator.free(layer);
    try refused(error.HintExpert, layer, expect());
    const many = try patched(f, u16, @intCast(ex + 6), 9);
    defer t.allocator.free(many);
    try refused(error.HintExpert, many, expect());
}
