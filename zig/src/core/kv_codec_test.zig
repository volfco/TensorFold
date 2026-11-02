//! KV codec oracles: rANS edge lengths and tables, block round trips for each kept-state byte class, the container's
//! exact length and its refusals before allocation, and the Phase 0 ledger's bit-pattern checks.
const std = @import("std");
const t = std.testing;
const rans = @import("kv_rans.zig");
const codec = @import("kv_codec.zig");
const ledger = @import("kv_ledger.zig");
pub const synth = @import("kv_synth.zig");

test {
    _ = @import("kv_ch_int.zig");
    _ = @import("kv_ch_float.zig");
}

fn ransTrip(src: []const u8) !usize {
    var counts: [256]u64 = @splat(0);
    for (src) |b| counts[b] += 1;
    if (src.len == 0) counts[0] = 1;
    const tab = rans.Table.fromCounts(&counts);
    const out = try t.allocator.alloc(u8, rans.bound(src.len));
    defer t.allocator.free(out);
    const start = try rans.encode(&tab, src, out);
    const back = try t.allocator.alloc(u8, src.len);
    defer t.allocator.free(back);
    const d = try t.allocator.create(rans.Decoder);
    defer t.allocator.destroy(d);
    d.init(tab);
    try d.decode(out[start..], back);
    try t.expectEqualSlices(u8, src, back);
    return out.len - start;
}

test "kv rans: every length below the interleave and past it round trips" {
    var prng = std.Random.DefaultPrng.init(7);
    var buf: [67]u8 = undefined;
    for (0..buf.len) |n| {
        prng.random().bytes(buf[0..n]);
        _ = try ransTrip(buf[0..n]);
    }
}

test "kv rans: one symbol, two symbols, uniform 256 and a 1 MiB skewed stream" {
    const one: [1000]u8 = @splat(9);
    try t.expectEqual(@as(usize, 4 * rans.lanes), try ransTrip(&one)); // a certain symbol costs only the states
    var two: [4096]u8 = undefined;
    for (&two, 0..) |*b, i| b.* = if (i % 7 == 0) 1 else 200;
    _ = try ransTrip(&two);
    var all: [256 * 16]u8 = undefined;
    for (&all, 0..) |*b, i| b.* = @truncate(i);
    const uniform = try ransTrip(&all);
    try t.expect(uniform >= all.len); // no gain on uniform bytes: callers must fall back to raw
    const big = try t.allocator.alloc(u8, 1 << 20);
    defer t.allocator.free(big);
    var prng = std.Random.DefaultPrng.init(11);
    for (big) |*b| b.* = @intCast(@min(255, @as(u32, @intFromFloat(@abs(prng.random().floatNorm(f64)) * 12))));
    const coded = try ransTrip(big);
    var counts: [256]u64 = @splat(0);
    for (big) |b| counts[b] += 1;
    try t.expect(coded < big.len * 5 / 8); // a half-normal of sigma 12 holds about 4.6 bits a byte
    try t.expect(coded <= rans.entropyBytes(&counts) * 101 / 100 + 64); // within 1 % of order-0 entropy
}

test "kv rans: normalized tables sum to the scale and keep every counted symbol" {
    var prng = std.Random.DefaultPrng.init(3);
    for (0..200) |round| {
        var counts: [256]u64 = @splat(0);
        const present = 1 + round % 256;
        for (0..present) |k| counts[(k * 37 + round) % 256] = 1 + prng.random().uintLessThan(u64, if (round % 3 == 0) 1 << 40 else 100);
        const tab = rans.Table.fromCounts(&counts);
        var sum: u32 = 0;
        for (counts, tab.freq) |c, f| {
            try t.expectEqual(c > 0, f > 0);
            sum += f;
        }
        try t.expectEqual(rans.scale, sum);
        var bytes: [32 + 512]u8 = undefined;
        const n = try tab.write(&bytes);
        try t.expectEqual(tab.serializedLen(), n);
        const back = try rans.Table.read(bytes[0..n]);
        try t.expectEqual(n, back.used);
        try t.expectEqualSlices(u16, &tab.freq, &back.table.freq);
    }
}

test "kv rans: forged tables and streams are refused" {
    var counts: [256]u64 = @splat(0);
    counts[1] = 3;
    counts[2] = 5;
    const tab = rans.Table.fromCounts(&counts);
    var bytes: [36]u8 = undefined;
    _ = try tab.write(&bytes);
    var forged = bytes;
    forged[33] +%= 1; // a frequency off by one: the sum is wrong
    try t.expectError(error.BadTable, rans.Table.read(&forged));
    forged = bytes;
    std.mem.writeInt(u16, forged[32..34], 0, .little);
    try t.expectError(error.BadTable, rans.Table.read(&forged));
    try t.expectError(error.Truncated, rans.Table.read(bytes[0..35]));
    const unit = [_]u8{ 1, 2, 2, 1, 2, 2, 2, 1, 2 };
    var src: [unit.len * 50]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = unit[i % unit.len];
    var out: [rans.bound(src.len)]u8 = undefined;
    const start = try rans.encode(&tab, &src, &out);
    var d: rans.Decoder = undefined;
    d.init(tab);
    var back: [src.len]u8 = undefined;
    try t.expectError(error.Truncated, d.decode(out[start .. out.len - 1], &back));
    try t.expectError(error.Corrupt, d.decode(out[start..], back[0 .. back.len - 1])); // input left over
    var bad = out;
    bad[start + 3] ^= 0xFF; // the first state's top byte: outside its range
    try t.expectError(error.Corrupt, d.decode(bad[start..], &back));
}

fn blockTrip(s: *codec.Scratch, raw: []const u8) !codec.Block {
    return blockTripWith(s, raw, 0, .plain);
}

fn blockTripWith(s: *codec.Scratch, raw: []const u8, row: u32, o: codec.Options) !codec.Block {
    const r = codec.encodeBlock(s, raw, 0, row, o);
    try t.expect(r.payload.len <= raw.len);
    try t.expectEqual(@as(usize, r.block.len), r.payload.len);
    const copy = try t.allocator.dupe(u8, r.payload); // the payload may live in the scratch the decode reuses
    defer t.allocator.free(copy);
    const back = try t.allocator.alloc(u8, raw.len);
    defer t.allocator.free(back);
    try codec.decodeBlock(s, r.block, copy, back);
    try t.expectEqualSlices(u8, raw, back);
    return r.block;
}

test "kv codec: each synthetic class round trips and the widened FP32 state codes as constant low planes" {
    var s = try codec.Scratch.init(t.allocator);
    defer s.deinit(t.allocator);
    const raw = try t.allocator.alloc(u8, codec.block_size);
    defer t.allocator.free(raw);
    for (std.enums.values(synth.Kind)) |kind| {
        synth.fill(kind, raw, 42);
        const b = try blockTrip(&s, raw);
        switch (kind) {
            .random => try t.expectEqual(@intFromEnum(codec.Kind.raw), b.kind),
            .zeros => try t.expect(b.len < 64),
            .f32_from_bf16 => {
                try t.expectEqual(@as(u8, 4), b.width);
                try t.expect(b.len < raw.len / 2); // the low 16 bits cost two constant planes
            },
            .bf16_gauss, .bf16_rows => try t.expect(b.len < raw.len),
            else => {},
        }
    }
    for ([_]usize{ 0, 1, 2, 3, 5, 63, 64, 65, 4097 }) |n| {
        synth.fill(.bf16_gauss, raw[0..n], @intCast(n));
        if (n > 0) _ = try blockTrip(&s, raw[0..n]);
    }
}

test "kv codec: a tampered payload or record is refused, never decoded wrong" {
    var s = try codec.Scratch.init(t.allocator);
    defer s.deinit(t.allocator);
    const raw = try t.allocator.alloc(u8, 1 << 16);
    defer t.allocator.free(raw);
    synth.fill(.bf16_gauss, raw, 5);
    const r = codec.encodeBlock(&s, raw, 0, 0, .plain);
    try t.expectEqual(@intFromEnum(codec.Kind.planes), r.block.kind);
    const payload = try t.allocator.dupe(u8, r.payload);
    defer t.allocator.free(payload);
    const back = try t.allocator.alloc(u8, raw.len);
    defer t.allocator.free(back);
    var prng = std.Random.DefaultPrng.init(9);
    for (0..64) |_| {
        const i = prng.random().uintLessThan(usize, payload.len);
        payload[i] ^= @as(u8, 1) << prng.random().int(u3);
        if (codec.decodeBlock(&s, r.block, payload, back)) |_| return error.TamperDecoded else |_| {}
        payload[i] = r.payload[i];
    }
    var b = r.block;
    b.raw -= 1;
    try t.expectError(error.BadBlock, codec.decodeBlock(&s, b, payload, back[0 .. back.len - 1]));
    b = r.block;
    b.kind = 7;
    try t.expectError(error.BadBlock, codec.decodeBlock(&s, b, payload, back));
}

test "kv codec: blocks cut at every span end and every block size from each span's start" {
    const spans = [_]codec.Span{ .{ .len = 10, .class = 0 }, .{ .len = codec.block_size + 5, .class = 1 }, .{ .len = 0, .class = 2 }, .{ .len = 2 * codec.block_size, .class = 3 } };
    var cuts: [8]codec.Cut = undefined;
    const n = codec.cutCount(&spans);
    try t.expectEqual(@as(usize, 5), n);
    codec.cut(&spans, cuts[0..n]);
    var at: usize = 0;
    for (cuts[0..n]) |c| {
        try t.expectEqual(at, c.off);
        try t.expect(c.len > 0 and c.len <= codec.block_size);
        at += c.len;
    }
    try t.expectEqual(@as(usize, 10 + codec.block_size + 5 + 2 * codec.block_size), at);
    try t.expectEqual(@as(u8, 3), cuts[n - 1].class);
    try t.expectEqual(@as(usize, 10 + codec.block_size + 5), cuts[3].off); // the last span cut from its own start
    try t.expectEqual(codec.block_size, cuts[3].len);
}

test "kv codec: the container's length is serializedLen and every truncation or forgery is refused" {
    var s = try codec.Scratch.init(t.allocator);
    defer s.deinit(t.allocator);
    const raw = try t.allocator.alloc(u8, 3 * 4096 + 7);
    defer t.allocator.free(raw);
    synth.fill(.state_mix, raw, 1);
    const spans = [_]codec.Span{ .{ .len = 4096, .class = 0 }, .{ .len = 8192, .class = 1 }, .{ .len = 7, .class = 5 } };
    const bytes = try codec.encode(t.allocator, &s, raw, &spans, .atlas);
    defer t.allocator.free(bytes);
    const c = try codec.inspect(bytes, raw.len);
    var blocks: [3]codec.Block = undefined;
    for (&blocks, c.blocks) |*b, x| b.* = x;
    try t.expectEqual(bytes.len, codec.serializedLen(&blocks));
    const back = try codec.decode(t.allocator, &s, bytes, raw.len);
    defer t.allocator.free(back);
    try t.expectEqualSlices(u8, raw, back);
    for (0..bytes.len) |n| if (codec.decode(t.allocator, &s, bytes[0..n], raw.len)) |x| {
        t.allocator.free(x);
        return error.TruncationDecoded;
    } else |_| {};
    try t.expectError(error.TooLarge, codec.inspect(bytes, raw.len - 1));
    // refused before the output allocation: a failing allocator is never asked
    var none = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    try t.expectError(error.TooLarge, codec.decode(none.allocator(), &s, bytes, raw.len - 1));
    const forged = try t.allocator.dupe(u8, bytes);
    defer t.allocator.free(forged);
    forged[@sizeOf(codec.Header)] ^= 1; // a record's raw length: the table hash catches it
    try t.expectError(error.Mismatch, codec.inspect(forged, raw.len));
    @memcpy(forged, bytes);
    forged[0] = 'X';
    try t.expectError(error.BadHeader, codec.inspect(forged, raw.len));
    @memcpy(forged, bytes);
    forged[forged.len - 1] ^= 0x40; // the last block's payload: decode's hash catches it
    try t.expect(if (codec.decode(t.allocator, &s, forged, raw.len)) |x| blk: {
        t.allocator.free(x);
        break :blk false;
    } else |_| true);
}

test "kv codec: parallel work covers every item once" {
    var hits: [1000]std.atomic.Value(u32) = undefined;
    for (&hits) |*h| h.* = .init(0);
    const Ctx = struct {
        h: []std.atomic.Value(u32),
        fn run(c: *const @This(), i: usize, _: usize) void {
            _ = c.h[i].fetchAdd(1, .monotonic);
        }
    };
    const ctx: Ctx = .{ .h = &hits };
    codec.each(8, hits.len, &ctx, Ctx.run);
    for (&hits) |*h| try t.expectEqual(@as(u32, 1), h.load(.monotonic));
}

test "kv ledger: classes sum to the stream, widened FP32 and FP8-in-BF16 patterns are counted" {
    var l: ledger.Ledger = .{};
    var f32s: [1024]u8 = undefined;
    synth.fill(.f32_from_bf16, &f32s, 2);
    l.observe(.prefix, &f32s);
    var bf: [512]u8 = undefined;
    synth.fill(.bf16_gauss, &bf, 3);
    l.observe(.latent, &bf);
    l.observe(.latent, bf[0..3]);
    try t.expectEqual(@as(u64, 1024 + 515), l.total());
    const x = l.e[@intFromEnum(ledger.Class.prefix)];
    try t.expectEqual(x.words32, x.low16_zero); // every word widened: exact packing is available
    try t.expectEqual(@as(u64, 2), l.e[@intFromEnum(ledger.Class.latent)].spans);
    try t.expect(l.modeled(.prefix).bytes < 1024);
    try t.expect(l.ceiling() > 0);
    var buf: [2048]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try l.print(&w, "test");
    try t.expect(std.mem.indexOf(u8, w.buffered(), "KVLEDGER test total bytes=1539") != null);
}

test "kv codec: every Atlas column candidate round trips every class, and smallest-of-N never loses to the model" {
    var s = try codec.Scratch.init(t.allocator);
    defer s.deinit(t.allocator);
    const raw = try t.allocator.alloc(u8, 256 * 1024 + 6);
    defer t.allocator.free(raw);
    var used: [3][4]usize = @splat(@splat(0)); // kind x transform: the engagement histogram
    for (std.enums.values(synth.Kind)) |kind| for ([_]u32{ 0, 1024, 2048 }) |row| {
        synth.fill(kind, raw, row + 1);
        const plain = try blockTripWith(&s, raw, row, .plain);
        const picked = try blockTripWith(&s, raw, row, .atlas);
        const best = try blockTripWith(&s, raw, row, .atlas_exhaustive);
        try t.expect(best.len <= picked.len);
        try t.expect(best.len <= plain.len); // plain planes are one of the candidates
        used[best.kind][best.xform] += 1;
    };
    var coded: usize = 0;
    for (used[1..]) |k| for (k) |n| {
        coded += n;
    };
    try t.expect(coded > 0);
}

test "kv codec: a token-row column with a constant step codes far below its plain planes" {
    var s = try codec.Scratch.init(t.allocator);
    defer s.deinit(t.allocator);
    const row = 512; // 256 BF16 channels a row
    const raw = try t.allocator.alloc(u8, 64 * row);
    defer t.allocator.free(raw);
    var prng = std.Random.DefaultPrng.init(4);
    var base: [256]u16 = undefined;
    for (&base) |*b| b.* = prng.random().int(u16);
    for (0..64) |r| for (0..256) |ch| std.mem.writeInt(u16, raw[r * row + 2 * ch ..][0..2], base[ch] +% @as(u16, @intCast(r * 3)), .little);
    const plain = try blockTripWith(&s, raw, row, .plain);
    const atlas = try blockTripWith(&s, raw, row, .atlas_exhaustive);
    try t.expect(atlas.xform != @intFromEnum(codec.Xform.none));
    try t.expectEqual(@as(u32, row), atlas.lag * atlas.width); // along the token axis, one row back
    try t.expect(atlas.len * 4 < plain.len);
}

test "kv codec: forged transform records are refused" {
    var s = try codec.Scratch.init(t.allocator);
    defer s.deinit(t.allocator);
    const raw = try t.allocator.alloc(u8, 4096);
    defer t.allocator.free(raw);
    synth.fill(.bf16_rows, raw, 8);
    const r = codec.encodeBlock(&s, raw, 0, 1024, .atlas_exhaustive);
    const payload = try t.allocator.dupe(u8, r.payload);
    defer t.allocator.free(payload);
    const back = try t.allocator.alloc(u8, raw.len);
    defer t.allocator.free(back);
    var b = r.block;
    b.xform = 9;
    try t.expectError(error.BadBlock, codec.decodeBlock(&s, b, payload, back));
    b = r.block;
    b.xform = @intFromEnum(codec.Xform.xor);
    b.lag = 0;
    try t.expectError(error.BadBlock, codec.decodeBlock(&s, b, payload, back));
}
