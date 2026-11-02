//! Lossless kept-state blocks: each block's bytes split into planes at width 1, 2 or 4 (sign/exponent bytes apart from
//! mantissa bytes), each plane constant, raw or order-0 rANS, and the whole block raw whenever coding does not pay.
//! A block's payload is never longer than its raw bytes, so a stream compacts in place block by block.
//!
//! Options.atlas adds the Atlas column-codec candidates (kv_xform.zig: Gorilla XOR, Delta, DoubleDelta along elements
//! or token rows, and T64 bit-packing), chosen a block at a time by the model or by smallest-of-N; every block records
//! its choice, so a run's histogram of choices proves which candidates were engaged.
//!
//! Container (`encode`/`decode`, for files and tests): Header, then `blocks` Block records, then the payloads in order.
//! Its length is `serializedLen` by construction, and `decode` refuses a forged length, table or payload before it
//! allocates the output.
const std = @import("std");
const rans = @import("kv_rans.zig");
const xf = @import("kv_xform.zig");
pub const Xform = xf.Xform;

/// A block holds at most this, cut from its span's start: equal spans (one prefix kept twice) cut into equal blocks.
pub const worker_pool = @import("kv_codec_pool.zig"); // persistent pinned threads (linked with libc)
pub const block_size: usize = 1 << 20;
pub const version: u32 = 1;
pub const magic = [8]u8{ 'T', 'F', 'K', 'V', 'C', 0, 0, 1 };

pub const Error = rans.Error || error{ BadHeader, BadBlock, Mismatch, TooLarge };

pub const Kind = enum(u8) { raw = 0, planes = 1, t64 = 2 };
const Mode = enum(u8) { constant = 0, raw = 1, rans = 2 };

/// One block's record: its raw and stored lengths, the raw bytes' hash, how it is stored (kind, element width,
/// prediction transform and its lag in elements) and the ledger class it holds.
pub const Block = extern struct {
    raw: u32,
    len: u32,
    hash: u64,
    kind: u8,
    width: u8,
    class: u8,
    xform: u8 = 0,
    lag: u32 = 0,
};

/// Which candidates a block may take.
pub const Options = struct {
    xforms: []const Xform = &.{.none},
    t64: bool = false,
    exhaustive: bool = false, // encode every candidate and keep the smallest, rather than the model's pick

    pub const plain: Options = .{};
    pub const atlas: Options = .{ .xforms = &.{ .none, .xor, .delta, .delta2 }, .t64 = true };
    pub const atlas_exhaustive: Options = .{ .xforms = atlas.xforms, .t64 = true, .exhaustive = true };

    /// "plain", "atlas" or "atlas-exhaustive".
    pub fn parse(text: []const u8) ?Options {
        if (std.mem.eql(u8, text, "plain")) return plain;
        if (std.mem.eql(u8, text, "atlas")) return atlas;
        if (std.mem.eql(u8, text, "atlas-exhaustive")) return atlas_exhaustive;
        return null;
    }
};

pub const Header = extern struct {
    magic: [8]u8 = magic,
    version: u32 = version,
    blocks: u32,
    raw: u64,
    table_hash: u64,
};

/// A stretch of the raw stream that one ledger class fills; `row` is its row's bytes when it holds rows (0: unknown).
pub const Span = struct { len: usize, class: u8 = 0, row: u32 = 0 };
/// A block's place in the raw stream.
pub const Cut = struct { off: usize, len: usize, class: u8, row: u32 = 0 };

/// How many blocks `spans` cut into: one at every span end and every block_size from each span's start.
pub fn cutCount(spans: []const Span) usize {
    var n: usize = 0;
    for (spans) |s| n += (s.len + block_size - 1) / block_size;
    return n;
}

pub fn cut(spans: []const Span, out: []Cut) void {
    var n: usize = 0;
    var off: usize = 0;
    for (spans) |s| {
        var at = off;
        const end = off + s.len;
        while (at < end) {
            const next = @min(end, at + block_size);
            out[n] = .{ .off = at, .len = next - at, .class = s.class, .row = s.row };
            n += 1;
            at = next;
        }
        off = end;
    }
    std.debug.assert(n == out.len);
}

/// Byte histograms at the four positions mod 4 from a block's start; width 2 and 1 histograms are their sums.
pub const Counts = struct {
    h: [4][256]u64 = @splat(@splat(0)),

    pub fn add(c: *Counts, bytes: []const u8) void {
        const whole = bytes.len & ~@as(usize, 3);
        var i: usize = 0;
        while (i < whole) : (i += 4) {
            c.h[0][bytes[i]] += 1;
            c.h[1][bytes[i + 1]] += 1;
            c.h[2][bytes[i + 2]] += 1;
            c.h[3][bytes[i + 3]] += 1;
        }
        for (bytes[whole..], 0..) |b, k| c.h[k][b] += 1;
    }

    /// Plane `p`'s histogram at width `w`.
    pub fn plane(c: *const Counts, w: u8, p: usize) [256]u64 {
        var out: [256]u64 = @splat(0);
        var q = p;
        while (q < 4) : (q += w) {
            for (&out, c.h[q]) |*o, x| o.* += x;
        }
        return out;
    }
};

/// A width's modeled bytes: each plane's order-0 entropy plus its table, or raw if that is smaller.
pub fn modeled(c: *const Counts, w: u8, len: usize) u64 {
    var total: u64 = 1 + len % w;
    for (0..w) |p| {
        const h = c.plane(w, p);
        var present: usize = 0;
        for (h) |x| present += @intFromBool(x > 0);
        const n = len / w;
        total += if (present <= 1) 2 else @min(1 + @as(u64, n), 5 + 32 + 2 * present + rans.entropyBytes(&h) + 4 * rans.lanes);
    }
    return total;
}

pub const Choice = struct { width: u8, bytes: u64 };

/// The width the model prefers and its modeled bytes.
pub fn choose(c: *const Counts, len: usize) Choice {
    var best: Choice = .{ .width = 1, .bytes = modeled(c, 1, len) };
    for ([_]u8{ 2, 4 }) |w| {
        const m = modeled(c, w, len);
        if (m < best.bytes) best = .{ .width = w, .bytes = m };
    }
    return best;
}

/// One worker's buffers for one block at a time; nothing allocates per block.
pub const Scratch = struct {
    planes: []u8, // block_size: a block's planes, each len / width bytes
    coded: []u8, // rANS output for one plane, written from its end
    out: []u8, // block_size: the stored payload
    input: []u8, // block_size: a payload gathered from pages
    prep: []u8, // block_size: a candidate's transformed elements
    best: []u8, // block_size: the smallest payload so far (exhaustive)
    dec: *rans.Decoder,

    pub fn init(gpa: std.mem.Allocator) !Scratch {
        const planes = try gpa.alloc(u8, block_size);
        errdefer gpa.free(planes);
        const coded = try gpa.alloc(u8, block_size + 1024); // a whole-block plane and its largest table
        errdefer gpa.free(coded);
        const out = try gpa.alloc(u8, block_size);
        errdefer gpa.free(out);
        const input = try gpa.alloc(u8, block_size);
        errdefer gpa.free(input);
        const prep = try gpa.alloc(u8, block_size);
        errdefer gpa.free(prep);
        const best = try gpa.alloc(u8, block_size);
        errdefer gpa.free(best);
        return .{ .planes = planes, .coded = coded, .out = out, .input = input, .prep = prep, .best = best, .dec = try gpa.create(rans.Decoder) };
    }

    pub fn deinit(s: *Scratch, gpa: std.mem.Allocator) void {
        gpa.free(s.planes);
        gpa.free(s.coded);
        gpa.free(s.out);
        gpa.free(s.input);
        gpa.free(s.prep);
        gpa.free(s.best);
        gpa.destroy(s.dec);
    }
};

/// The integrity hash of a block's raw bytes and of the container's table. Wyhash: std's XxHash3 measured 0.17 GB/s
/// against Wyhash's 15 GB/s on x86 (Zig 0.17), and it runs on every block both ways.
pub fn hash(raw: []const u8) u64 {
    return std.hash.Wyhash.hash(0, raw);
}

pub const Encoded = struct { block: Block, payload: []const u8 };

const Candidate = struct { xform: Xform = .none, kind: Kind = .planes, width: u8 = 0, lag: u32 = 0 };

/// The candidates `o` allows for a block whose rows are `row` bytes: plain planes, then each transform at element
/// widths 2 and 4 along neighbouring elements and along rows, each also into T64.
fn candidates(o: Options, row: u32, out: *[64]Candidate) []const Candidate {
    var n: usize = 0;
    out[n] = .{};
    n += 1;
    for ([_]u8{ 2, 4 }) |w| {
        var lags: [2]u32 = .{ 1, 0 };
        if (row % w == 0 and row / w > 1) lags[1] = row / w;
        for (o.xforms) |x| for (lags) |lag| {
            if (lag == 0 or (x == .none and lag > 1)) continue;
            if (x != .none) {
                out[n] = .{ .xform = x, .width = w, .lag = lag };
                n += 1;
            }
            if (o.t64) {
                out[n] = .{ .xform = x, .kind = .t64, .width = w, .lag = if (x == .none) 0 else lag };
                n += 1;
            }
        };
    }
    return out[0..n];
}

/// A candidate's input: the raw bytes, or its transform of them in `s.prep`.
fn prepared(s: *Scratch, raw: []const u8, c: Candidate) []const u8 {
    if (c.xform == .none) return raw;
    xf.forward(c.xform, c.width, c.lag, raw, s.prep[0..raw.len]);
    return s.prep[0..raw.len];
}

/// Encode candidate `c` into `s.out` (at most raw.len - 1 bytes): its length and width, or null when it cannot pay.
fn attempt(s: *Scratch, raw: []const u8, c: Candidate) ?struct { len: usize, width: u8 } {
    const src = prepared(s, raw, c);
    switch (c.kind) {
        .t64 => return .{ .len = xf.t64Encode(c.width, src, s.out[0 .. raw.len - 1]) catch return null, .width = c.width },
        .planes => {
            var counts: Counts = .{};
            counts.add(src);
            const w = if (c.width == 0) choose(&counts, src.len).width else c.width;
            return .{ .len = encodePlanes(s, src, &counts, w) catch return null, .width = w };
        },
        .raw => unreachable,
    }
}

/// A candidate's modeled bytes (exact for T64).
fn model(s: *Scratch, raw: []const u8, c: Candidate) u64 {
    const src = prepared(s, raw, c);
    if (c.kind == .t64) return xf.t64Size(c.width, src);
    var counts: Counts = .{};
    counts.add(src);
    return if (c.width == 0) choose(&counts, src.len).bytes else modeled(&counts, c.width, src.len);
}

/// The block's record, its payload in scratch or `raw` itself (raw): never longer than `raw`.
pub fn encodeBlock(s: *Scratch, raw: []const u8, class: u8, row: u32, o: Options) Encoded {
    std.debug.assert(raw.len <= block_size);
    const as_raw: Block = .{ .raw = @intCast(raw.len), .len = @intCast(raw.len), .hash = hash(raw), .kind = @intFromEnum(Kind.raw), .width = 1, .class = class };
    if (raw.len < 64) return .{ .block = as_raw, .payload = raw };
    var buf: [64]Candidate = undefined;
    const all = candidates(o, row, &buf);
    var best: ?struct { c: Candidate, len: usize, width: u8 } = null;
    if (o.exhaustive) {
        for (all) |c| if (attempt(s, raw, c)) |r| if (best == null or r.len < best.?.len) {
            std.mem.swap([]u8, &s.out, &s.best); // the smallest so far moves aside, out to `best`
            best = .{ .c = c, .len = r.len, .width = r.width };
        };
        if (best != null) std.mem.swap([]u8, &s.out, &s.best);
    } else {
        var pick = all[0];
        var least: u64 = std.math.maxInt(u64);
        for (all) |c| {
            const m = model(s, raw, c);
            if (m < least) {
                least = m;
                pick = c;
            }
        }
        // the model is within a table's bytes of the coder: skip coding that cannot pay
        if (least + raw.len / 64 >= raw.len) return .{ .block = as_raw, .payload = raw };
        if (attempt(s, raw, pick)) |r| best = .{ .c = pick, .len = r.len, .width = r.width };
    }
    const b = best orelse return .{ .block = as_raw, .payload = raw };
    if (b.len >= raw.len) return .{ .block = as_raw, .payload = raw };
    var rec = as_raw;
    rec.len = @intCast(b.len);
    rec.kind = @intFromEnum(b.c.kind);
    rec.width = b.width;
    rec.xform = @intFromEnum(b.c.xform);
    rec.lag = b.c.lag;
    return .{ .block = rec, .payload = s.out[0..b.len] };
}

fn split(raw: []const u8, w: u8, planes: []u8) void {
    const n = raw.len / w;
    switch (w) {
        1 => @memcpy(planes[0..n], raw[0..n]),
        2 => for (0..n) |i| {
            planes[i] = raw[2 * i];
            planes[n + i] = raw[2 * i + 1];
        },
        4 => for (0..n) |i| {
            planes[i] = raw[4 * i];
            planes[n + i] = raw[4 * i + 1];
            planes[2 * n + i] = raw[4 * i + 2];
            planes[3 * n + i] = raw[4 * i + 3];
        },
        else => unreachable,
    }
}

fn join(planes: []const u8, w: u8, raw: []u8) void {
    const n = raw.len / w;
    switch (w) {
        1 => @memcpy(raw[0..n], planes[0..n]),
        2 => for (0..n) |i| {
            raw[2 * i] = planes[i];
            raw[2 * i + 1] = planes[n + i];
        },
        4 => for (0..n) |i| {
            raw[4 * i] = planes[i];
            raw[4 * i + 1] = planes[n + i];
            raw[4 * i + 2] = planes[2 * n + i];
            raw[4 * i + 3] = planes[3 * n + i];
        },
        else => unreachable,
    }
}

/// Bounded writes into `s.out` up to `limit` bytes; NoRoom sends the block raw.
const Out = struct {
    buf: []u8,
    at: usize = 0,

    fn put(o: *Out, bytes: []const u8) rans.Error!void {
        if (o.buf.len - o.at < bytes.len) return error.NoRoom;
        @memcpy(o.buf[o.at..][0..bytes.len], bytes);
        o.at += bytes.len;
    }

    fn byte(o: *Out, b: u8) rans.Error!void {
        return o.put(&.{b});
    }
};

/// Payload: width, then each plane (mode byte; constant: its byte; raw: its bytes; rans: u32 length, table, stream),
/// then the len % width tail bytes raw.
fn encodePlanes(s: *Scratch, raw: []const u8, c: *const Counts, w: u8) rans.Error!usize {
    const n = raw.len / w;
    split(raw, w, s.planes);
    var o: Out = .{ .buf = s.out[0 .. raw.len - 1] };
    try o.byte(w);
    for (0..w) |p| {
        const plane = s.planes[p * n ..][0..n];
        if (n > 0 and std.mem.allEqual(u8, plane, plane[0])) {
            try o.byte(@intFromEnum(Mode.constant));
            try o.byte(plane[0]);
            continue;
        }
        const coded = codePlane(s, plane, c.plane(w, p));
        if (coded) |stream| {
            try o.byte(@intFromEnum(Mode.rans));
            var len: [4]u8 = undefined;
            std.mem.writeInt(u32, &len, @intCast(stream.len), .little);
            try o.put(&len);
            try o.put(stream);
        } else {
            try o.byte(@intFromEnum(Mode.raw));
            try o.put(plane);
        }
    }
    try o.put(raw[n * w ..]);
    return o.at;
}

/// The plane's table and stream, contiguous at the end of `s.coded`, or null when raw is no larger.
fn codePlane(s: *Scratch, plane: []const u8, counts: [256]u64) ?[]const u8 {
    if (plane.len == 0) return null;
    const t = rans.Table.fromCounts(&counts);
    const table_len = t.serializedLen();
    if (table_len + 4 * rans.lanes >= plane.len) return null;
    const room = s.coded[0..@min(s.coded.len, plane.len + table_len)];
    const start = rans.encode(&t, plane, room[table_len..]) catch return null;
    const at = table_len + start;
    _ = t.write(room[at - table_len .. at]) catch return null;
    const stream = room[at - table_len ..];
    return if (stream.len + 5 >= plane.len) null else stream;
}

/// Decode `payload` (the block's stored bytes, exactly `b.len`) into `out` (exactly `b.raw`), its hash checked.
pub fn decodeBlock(s: *Scratch, b: Block, payload: []const u8, out: []u8) Error!void {
    if (payload.len != b.len or out.len != b.raw or b.raw > block_size or b.len > b.raw) return error.BadBlock;
    const x = std.enums.fromInt(Xform, b.xform) orelse return error.BadBlock;
    if (x != .none and ((b.width != 2 and b.width != 4) or b.lag == 0)) return error.BadBlock;
    switch (std.enums.fromInt(Kind, b.kind) orelse return error.BadBlock) {
        .raw => {
            if (b.len != b.raw or x != .none) return error.BadBlock;
            @memcpy(out, payload);
        },
        .planes => {
            try decodePlanes(s, payload, out);
            if (payload[0] != b.width) return error.BadBlock;
        },
        .t64 => try xf.t64Decode(b.width, payload, out),
    }
    if (x != .none) xf.inverse(x, b.width, b.lag, out);
    if (hash(out) != b.hash) return error.Mismatch;
}

fn decodePlanes(s: *Scratch, payload: []const u8, out: []u8) Error!void {
    if (payload.len < 1) return error.Truncated;
    const w = payload[0];
    if (w != 1 and w != 2 and w != 4) return error.BadBlock;
    const n = out.len / w;
    var at: usize = 1;
    for (0..w) |p| {
        const plane = s.planes[p * n ..][0..n];
        if (at >= payload.len) return error.Truncated;
        const mode = std.enums.fromInt(Mode, payload[at]) orelse return error.BadBlock;
        at += 1;
        switch (mode) {
            .constant => {
                if (at >= payload.len) return error.Truncated;
                @memset(plane, payload[at]);
                at += 1;
            },
            .raw => {
                if (payload.len - at < n) return error.Truncated;
                @memcpy(plane, payload[at..][0..n]);
                at += n;
            },
            .rans => {
                if (payload.len - at < 4) return error.Truncated;
                const len = std.mem.readInt(u32, payload[at..][0..4], .little);
                at += 4;
                if (payload.len - at < len) return error.Truncated;
                const stream = payload[at..][0..len];
                const t = try rans.Table.read(stream);
                s.dec.init(t.table);
                try s.dec.decode(stream[t.used..], plane);
                at += len;
            },
        }
    }
    const tail = out.len - n * w;
    if (payload.len - at != tail) return error.Corrupt;
    join(s.planes[0 .. n * w], w, out[0 .. n * w]);
    @memcpy(out[n * w ..], payload[at..]);
}

/// The container's exact length for these records.
pub fn serializedLen(blocks: []const Block) usize {
    var n: usize = @sizeOf(Header) + blocks.len * @sizeOf(Block);
    for (blocks) |b| n += b.len;
    return n;
}

/// A whole stream as one container, one thread; `spans` (summing to raw.len) set the classes, or one span if empty.
pub fn encode(gpa: std.mem.Allocator, s: *Scratch, raw: []const u8, spans: []const Span, o: Options) ![]u8 {
    const whole = [_]Span{.{ .len = raw.len }};
    const sp = if (spans.len == 0) whole[0..] else spans;
    const cuts = try gpa.alloc(Cut, cutCount(sp));
    defer gpa.free(cuts);
    cut(sp, cuts);
    const blocks = try gpa.alloc(Block, cuts.len);
    defer gpa.free(blocks);
    var payloads: std.ArrayList(u8) = .empty;
    defer payloads.deinit(gpa);
    for (cuts, blocks) |c, *b| {
        const r = encodeBlock(s, raw[c.off..][0..c.len], c.class, c.row, o);
        b.* = r.block;
        try payloads.appendSlice(gpa, r.payload);
    }
    const table = std.mem.sliceAsBytes(blocks);
    const h: Header = .{ .blocks = @intCast(blocks.len), .raw = raw.len, .table_hash = hash(table) };
    const out = try gpa.alloc(u8, serializedLen(blocks));
    @memcpy(out[0..@sizeOf(Header)], std.mem.asBytes(&h));
    @memcpy(out[@sizeOf(Header)..][0..table.len], table);
    @memcpy(out[@sizeOf(Header) + table.len ..], payloads.items);
    return out;
}

/// The records of a container, every length and the table hash checked; nothing is allocated.
pub fn inspect(bytes: []const u8, max_raw: u64) Error!struct { header: Header, blocks: []align(1) const Block, payloads: []const u8 } {
    if (bytes.len < @sizeOf(Header)) return error.Truncated;
    const h = std.mem.bytesToValue(Header, bytes[0..@sizeOf(Header)]);
    if (!std.mem.eql(u8, &h.magic, &magic) or h.version != version) return error.BadHeader;
    if (h.raw > max_raw) return error.TooLarge;
    const table_len = std.math.mul(usize, h.blocks, @sizeOf(Block)) catch return error.BadHeader;
    if (bytes.len - @sizeOf(Header) < table_len) return error.Truncated;
    const table = bytes[@sizeOf(Header)..][0..table_len];
    if (hash(table) != h.table_hash) return error.Mismatch;
    const blocks = std.mem.bytesAsSlice(Block, table);
    var raw: u64 = 0;
    var stored: u64 = 0;
    for (blocks) |b| {
        if (b.raw == 0 or b.raw > block_size or b.len > b.raw) return error.BadBlock;
        if (std.enums.fromInt(Kind, b.kind) == null or std.enums.fromInt(Xform, b.xform) == null) return error.BadBlock;
        raw += b.raw;
        stored += b.len;
    }
    const payloads = bytes[@sizeOf(Header) + table_len ..];
    if (raw != h.raw) return error.BadHeader;
    if (stored != payloads.len) return if (stored > payloads.len) error.Truncated else error.Corrupt;
    return .{ .header = h, .blocks = blocks, .payloads = payloads };
}

/// The raw stream of a container, refused (before allocating) when it would exceed `max_raw` or any record lies.
pub fn decode(gpa: std.mem.Allocator, s: *Scratch, bytes: []const u8, max_raw: u64) ![]u8 {
    const c = try inspect(bytes, max_raw);
    const out = try gpa.alloc(u8, @intCast(c.header.raw));
    errdefer gpa.free(out);
    var raw: usize = 0;
    var at: usize = 0;
    for (c.blocks) |b| {
        try decodeBlock(s, b, c.payloads[at..][0..b.len], out[raw..][0..b.raw]);
        raw += b.raw;
        at += b.len;
    }
    return out;
}

/// Run `job(ctx, item, worker)` for every item below `count` on up to `workers` threads (this one included);
/// a thread that cannot start leaves its share to the others.
pub fn each(workers: usize, count: usize, ctx: anytype, comptime job: fn (@TypeOf(ctx), usize, usize) void) void {
    const C = @TypeOf(ctx);
    const Shared = struct {
        next: std.atomic.Value(usize) = .init(0),
        fn run(sh: *@This(), c: C, n: usize, worker: usize) void {
            while (true) {
                const i = sh.next.fetchAdd(1, .monotonic);
                if (i >= n) return;
                job(c, i, worker);
            }
        }
    };
    var sh: Shared = .{};
    const most = 64;
    var threads: [most]?std.Thread = @splat(null);
    const extra = @min(most, @min(workers, count) -| 1);
    for (threads[0..extra], 1..) |*t, w| t.* = std.Thread.spawn(.{}, Shared.run, .{ &sh, ctx, count, w }) catch null;
    sh.run(ctx, count, 0);
    for (threads[0..extra]) |t| if (t) |th| th.join();
}

test {
    _ = @import("kv_codec_test.zig");
    _ = xf;
}
