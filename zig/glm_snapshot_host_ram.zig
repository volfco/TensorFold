//! GLM kept states in pinned host memory (TF_GLM_HOST_CACHE) through the actual rank owner with a recording driver.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const t = std.testing;
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "GLM snapshots: TF_GLM_HOST_CACHE keeps states in pinned host memory and restores them byte for byte" {
    var d = rec.init();
    _ = setenv("TF_GLM_HOST_CACHE", "1", 1);
    defer _ = unsetenv("TF_GLM_HOST_CACHE");
    const before = cuda.usage(false).device;
    for (0..2) |rank| {
        var f: Fixture = undefined;
        try f.init(&d, rank, 2, true);
        defer f.deinit();
        const device_allocs = rec.allocs;
        const host_allocs = rec.host_allocs;
        try f.e.prepareSnapshots(2 * f.e.stateBytes(8192));
        try t.expectEqual(device_allocs, rec.allocs); // the arena is host memory: no device allocation
        try t.expectEqual(host_allocs + 1, rec.host_allocs);
        try t.expect(f.e.snap_pages.?.arena.? == .host);
        for ([_]usize{ 2048, 4096, 8192 }) |at| {
            try f.mark(at, at, @intCast(at % 251));
            const saved = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
            defer t.allocator.free(saved);
            const id = try f.e.keepState();
            @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0x5A);
            try f.seq.st.restoreHost(f.e.stream, 17, 18, &.{0}, .{ 0, 0 });
            try f.e.restoreState(id);
            try f.expectPrefix(saved, at, at);
            try t.expectEqual(at, f.seq.st.pos);
            f.e.dropState(id);
            f.e.dropped.clearRetainingCapacity();
        }
    }
    try t.expectEqual(before, cuda.usage(false).device);
}

/// `f.mark` with a compressible GLM-shaped state in place of the pattern bytes (which leave every block raw).
fn markCompressible(f: *Fixture, at: usize, head: usize) !void {
    try f.mark(at, head, 0);
    const s = &f.seq.st;
    @import("core").kv_synth.fill(.state_mix, rec.bytes(s.arena.buf.ptr, s.arena.buf.len), at);
    try s.restoreHost(f.e.stream, at, head, &.{@as(u1, @truncate(at))}, .{ if (at > 4096) at / 4 * 4 else 0, if (head > 4096) (head -| 8) / 4 * 4 else 0 });
}

test "GLM snapshots: TF_GLM_KV_CODEC compacts kept states in their host pages and restores them byte for byte" {
    var d = rec.init();
    _ = setenv("TF_GLM_HOST_CACHE", "1", 1);
    defer _ = unsetenv("TF_GLM_HOST_CACHE");
    _ = setenv("TF_GLM_KV_CODEC", "1", 1);
    defer _ = unsetenv("TF_GLM_KV_CODEC");
    _ = setenv("TF_GLM_KV_LEDGER", "1", 1); // the Phase 0 ledger runs too: its classes must cover every keep
    defer _ = unsetenv("TF_GLM_KV_LEDGER");
    const before = cuda.usage(false).device;
    try codecRun(&d);
    try t.expectEqual(before, cuda.usage(false).device); // after the fixture's own teardown
}

fn codecRun(d: *const cuda.Driver) !void {
    var f: Fixture = undefined;
    try f.init(d, 0, 1, true);
    defer f.deinit();
    var ledgered: u64 = 0;
    const device_allocs = rec.allocs;
    try f.e.prepareSnapshots(3 * f.e.stateBytes(8192));
    try t.expectEqual(device_allocs, rec.allocs); // pages and staging are host memory
    try t.expect(f.e.snap_codec != null);
    for ([_]usize{ 2048, 4096, 8192 }) |at| for ([_]bool{ false, true }) |compressible| {
        if (compressible) try markCompressible(&f, at, at) else try f.mark(at, at, @intCast(at % 251));
        const saved = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
        defer t.allocator.free(saved);
        const host_allocs = rec.host_allocs;
        const id = try f.e.keepState();
        try t.expectEqual(host_allocs, rec.host_allocs); // a keep allocates no pinned memory
        const kept = f.e.snaps.get(id).?;
        ledgered += kept.len;
        try t.expectEqual(ledgered, f.e.snap_codec.?.seen.total());
        try t.expect(f.e.snap_codec.?.seen.e[@intFromEnum(glm.snapshots.layout.Class.latent)].bytes > 0);
        const k = kept.compact orelse return error.ExpectedCompact;
        try t.expect(k.full());
        try t.expect(k.out <= kept.len);
        if (compressible) try t.expect(k.out < kept.len * 9 / 10);
        const charge = glm.snapshots.charged(&f.e, id);
        try t.expectEqual(k.pages() * glm.snapshots.pages.page, charge);
        try t.expectEqual(kept.lease.ids.items.len * glm.snapshots.pages.page, charge); // alone: trimmed at once
        try t.expect(charge <= f.e.stateBytes(at));
        try t.expectEqual(f.e.snap_pages.?.arena.?.len(), charge + f.e.snap_pages.?.spare());
        @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0x5A);
        try f.seq.st.restoreHost(f.e.stream, 17, 18, &.{0}, .{ 0, 0 });
        try f.e.restoreState(id);
        try f.expectPrefix(saved, at, at);
        try t.expectEqual(at, f.seq.st.pos);
        // a second restore of the same compacted pages gives the same bytes: the decode does not consume them
        @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0xA5);
        try f.e.restoreState(id);
        try f.expectPrefix(saved, at, at);
        f.e.dropState(id);
        f.e.dropped.clearRetainingCapacity();
    };
    try t.expectEqual(f.e.snap_pages.?.arena.?.len(), f.e.snap_pages.?.spare());
}

test "GLM snapshots: TF_GLM_KV_CODEC without TF_GLM_HOST_CACHE is refused before any page is made" {
    var d = rec.init();
    _ = setenv("TF_GLM_KV_CODEC", "1", 1);
    defer _ = unsetenv("TF_GLM_KV_CODEC");
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    try t.expectError(error.KvCodecNeedsHostCache, f.e.prepareSnapshots(2 * f.e.stateBytes(2048)));
    try t.expect(f.e.snap_pages == null);
}

test "GLM snapshots: TF_GLM_KV_DUMP writes a keep's bytes, rowed cuts and a manifest that covers its plan" {
    var d = rec.init();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const io = t.io;
    // TF_KVPROBE_FIXTURE_DUMP: an existing directory the dump stays in, for tools/kvprobe's format check
    const keep_dir = std.c.getenv("TF_KVPROBE_FIXTURE_DUMP");
    const real = try tmp.dir.realPathFileAlloc(io, ".", t.allocator);
    defer t.allocator.free(real);
    const dir: []const u8 = if (keep_dir) |v| std.mem.span(v) else real;
    const dir_z = try t.allocator.dupeSentinel(u8, dir, 0);
    defer t.allocator.free(dir_z);
    inline for (.{ .{ "TF_GLM_HOST_CACHE", "1" }, .{ "TF_GLM_KV_CODEC", "1" } }) |x| _ = setenv(x[0], x[1], 1);
    _ = setenv("TF_GLM_KV_DUMP", dir_z.ptr, 1);
    defer inline for (.{ "TF_GLM_HOST_CACHE", "TF_GLM_KV_CODEC", "TF_GLM_KV_DUMP" }) |x| {
        _ = unsetenv(x);
    };
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    try f.e.prepareSnapshots(2 * f.e.stateBytes(4096));
    try f.mark(4096, 4100, 7);
    const id = try f.e.keepState();
    const kept = f.e.snaps.get(id).?;
    const a = t.allocator;
    const stem = try std.fmt.allocPrint(a, "{s}/glm-rank0-keep{d}", .{ dir, id });
    defer a.free(stem);
    const names = [_][]const u8{ ".bin", ".parts", ".plan" };
    var files: [3][]u8 = undefined;
    inline for (names, &files) |ext, *out| {
        const path = try std.fmt.allocPrint(a, "{s}{s}", .{ stem, ext });
        defer a.free(path);
        out.* = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .unlimited);
    }
    defer for (files) |x| a.free(x);
    try t.expectEqual(kept.len, files[0].len);
    // the cuts: "class length row", rows marked on every cache part
    var cut_total: usize = 0;
    var lines = std.mem.tokenizeScalar(u8, files[1], '\n');
    while (lines.next()) |line| {
        var w = std.mem.tokenizeScalar(u8, line, ' ');
        const class = std.meta.stringToEnum(glm.snapshots.layout.Class, w.next().?).?;
        cut_total += try std.fmt.parseInt(usize, w.next().?, 10);
        const row = try std.fmt.parseInt(u32, w.next().?, 10);
        try t.expect((row == 0) == (class == .prefix or class == .tail));
    }
    try t.expectEqual(kept.len, cut_total);
    // the manifest: its parts sum to the plan (delta 0), its fields are the State's own offsets
    const st = &f.seq.st;
    var part_total: usize = 0;
    var caches_seen: [2]bool = .{ false, false };
    var fields: usize = 0;
    var ended = false;
    lines = std.mem.tokenizeScalar(u8, files[2], '\n');
    try t.expectEqualStrings("tfkvplan 1", lines.next().?);
    while (lines.next()) |line| {
        var w = std.mem.tokenizeScalar(u8, line, ' ');
        const key = w.next().?;
        if (std.mem.eql(u8, key, "part")) {
            _ = w.next();
            const class = std.meta.stringToEnum(glm.snapshots.layout.Class, w.next().?).?;
            part_total += try std.fmt.parseInt(usize, w.next().?, 10);
            const row = try std.fmt.parseInt(u32, w.next().?, 10);
            const cache = w.next().?;
            if (class == .prefix or class == .tail) {
                try t.expectEqualStrings("-", cache);
                try t.expectEqual(@as(u32, 0), row);
            } else {
                caches_seen[try std.fmt.parseInt(usize, cache, 10)] = true;
                try t.expect(row > 0);
            }
        } else if (std.mem.eql(u8, key, "field")) {
            const name = w.next().?;
            const off = try std.fmt.parseInt(u64, w.next().?, 10);
            if (std.mem.eql(u8, name, "rec")) try t.expectEqual(st.rec - st.arena.buf.ptr, off);
            if (std.mem.eql(u8, name, "conv")) try t.expectEqual(st.conv - st.arena.buf.ptr, off);
            fields += 1;
        } else if (std.mem.eql(u8, key, "cur")) {
            try t.expectEqual(st.kda_layers, w.next().?.len);
        } else if (std.mem.eql(u8, key, "end")) ended = true;
    }
    try t.expect(ended);
    try t.expectEqual(@as(usize, 8), fields);
    try t.expectEqual(kept.len, part_total);
    try t.expect(caches_seen[0] and caches_seen[1]); // the DSA layer's cache and the MTP layer's
}

// ---- TF_GLM_KDA_REPLAY: kept states without their KDA states, rebuilt by a (recording) replay pass on restore ----

const kda_replay = glm.engine.kda_replay;

/// The replay pass as the engine runs it, recorded: KDA halves flip, conv and states are written, the position moves.
const FakePass = struct {
    f: *Fixture,
    stored: ?[]const u8 = null, // check mode: the stored arena whose live states the pass reproduces, head 1 perturbed
    stored_cur: u1 = 0,
    calls: usize = 0,
    start: usize = 0,
    ids: [2048]u32 = undefined,
    n: usize = 0,

    fn runner(x: *FakePass) kda_replay.Runner {
        return .{ .ctx = x, .run = run };
    }

    fn run(ptr: *anyopaque, ids: []const u32) anyerror!void {
        const x: *FakePass = @ptrCast(@alignCast(ptr));
        const st = &x.f.seq.st;
        const base = st.arena.buf.ptr;
        x.calls += 1;
        x.start = st.pos;
        @memcpy(x.ids[0..ids.len], ids);
        x.n = ids.len;
        const conv = rec.bytes(st.conv, st.kda_layers * 3 * st.conv_c * 2);
        if (x.stored) |s| {
            @memcpy(conv, s[st.conv - base ..][0..conv.len]);
        } else for (conv, 0..) |*b, i| {
            b.* = @truncate(ids[i % ids.len] *% 31 +% i);
        }
        for (st.cur, 0..) |*c, li| {
            const out = std.mem.bytesAsSlice(f32, rec.bytes(st.recAt(1 - c.*, li), st.rec_layer));
            if (x.stored) |s| {
                @memcpy(std.mem.sliceAsBytes(out), s[st.recAt(x.stored_cur, li) - base ..][0..st.rec_layer]);
                out[128 * 128 + 5] += 0.5; // head 1 alone differs
            } else for (out, 0..) |*v, i| {
                v.* = @as(f32, @floatFromInt(ids[i % ids.len] % 97)) * 0.01;
            }
            c.* = 1 - c.*;
        }
        try st.setPos(x.f.e.stream, st.pos + ids.len);
    }
};

/// Prompt tokens 0..at committed on slot 0 (as prefillChunk notes them), in 2048-token chunks.
fn notePrompt(r: *kda_replay.Replay, at: usize) !void {
    var ids: [2048]u32 = undefined;
    var start: usize = 0;
    while (start < at) : (start += 2048) {
        const n = @min(2048, at - start);
        for (ids[0..n], 0..) |*x, i| x.* = @intCast(1000 + start + i);
        try r.note(0, start, ids[0..n]);
    }
}

/// The live KDA states and conv history as finite numbers (the marked pattern's bytes can be NaN as floats).
fn saneStates(f: *Fixture) void {
    const st = &f.seq.st;
    for (std.mem.bytesAsSlice(f32, rec.bytes(st.rec, 2 * st.kda_layers * st.rec_layer)), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.25 - 1;
    for (std.mem.bytesAsSlice(u16, rec.bytes(st.conv, st.kda_layers * 3 * st.conv_c * 2)), 0..) |*v, i| v.* = 0x3f80 + @as(u16, @intCast(i % 64));
}

fn fullPlanLen(f: *Fixture, at: usize) !usize {
    return (try glm.snapshots.layout.Plan.init(&f.seq.st, &f.e.c, at, at)).len;
}

fn headLen(f: *Fixture) usize {
    const st = &f.seq.st;
    return @intCast(st.caches[0].lc - st.arena.buf.ptr);
}

test "GLM snapshots: TF_GLM_KDA_REPLAY_KEEP keeps an entry without its KDA states and a restore replays its window" {
    var d = rec.init();
    _ = setenv("TF_GLM_HOST_CACHE", "1", 1);
    defer _ = unsetenv("TF_GLM_HOST_CACHE");
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    var pass: FakePass = .{ .f = &f };
    f.e.kda_replay = try kda_replay.Replay.create(t.allocator, 64, true, false, pass.runner());
    try f.e.prepareSnapshots(2 * f.e.stateBytes(8192));
    const at = 4096;
    try f.mark(at, at, 9);
    try notePrompt(f.e.kda_replay.?, at);
    const saved = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
    defer t.allocator.free(saved);
    const id = try f.e.keepState();
    const kept = f.e.snaps.get(id).?;
    const head = headLen(&f);
    try t.expectEqual(kda_replay.Form.replay, f.e.kda_replay.?.form(id).?);
    try t.expectEqual(try fullPlanLen(&f, at) - head, kept.len); // the arena head is not kept
    try t.expect(kept.lease.ids.items.len * glm.snapshots.pages.page < f.e.stateBytes(at));
    @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0x5A);
    try f.seq.st.restoreHost(f.e.stream, 17, 18, &.{0}, .{ 0, 0 });
    try f.e.restoreState(id);
    // the pass ran once, from at - 64 over the 64 prompt tokens before the kept position, and ended at it
    try t.expectEqual(@as(usize, 1), pass.calls);
    try t.expectEqual(@as(usize, at - 64), pass.start);
    for (pass.ids[0..pass.n], 0..) |x, i| try t.expectEqual(@as(u32, @intCast(1000 + at - 64 + i)), x);
    try t.expectEqual(@as(usize, at), f.seq.st.pos);
    // the head is the pass's alone (zeroed before it: nothing of the clobbered bytes), the rest the kept bytes
    const st = &f.seq.st;
    const live = rec.bytes(st.arena.buf.ptr, head);
    const expected = try t.allocator.dupe(u8, saved);
    defer t.allocator.free(expected);
    @memset(expected[0..head], 0);
    for ([_]u64{ st.pos_dev, st.conv }, [_]usize{ 4, st.kda_layers * 3 * st.conv_c * 2 }) |p, n| {
        const o: usize = @intCast(p - st.arena.buf.ptr);
        @memcpy(expected[o..][0..n], live[o..][0..n]);
    }
    const half: usize = @intCast(st.recAt(1, 0) - st.arena.buf.ptr);
    @memcpy(expected[half..][0..st.rec_layer], live[half..][0..st.rec_layer]);
    try t.expectEqual(@as(u1, 1), st.cur[0]);
    try t.expectEqualSlices(u8, expected[0..head], live);
    try t.expect(std.mem.indexOfScalar(u8, live[half..][0..st.rec_layer], 0x5A) == null or pass.n > 0);
    try f.expectPrefix(expected, at, at);
    // the same pages without the knob (a restart without TF_GLM_KDA_REPLAY) are refused by name
    const r = f.e.kda_replay.?;
    f.e.kda_replay = null;
    try t.expectError(error.KdaReplayNeedsWindow, f.e.restoreState(id));
    f.e.kda_replay = r;
    // a keep whose tokens are not all known (a decode commit moved the position) is kept whole, without a window
    try f.mark(8192, 8192, 3);
    try r.note(0, 8000, &.{ 1, 2, 3 });
    const whole = try f.e.keepState();
    try t.expect(r.form(whole) == null);
    try t.expectEqual(try fullPlanLen(&f, 8192), f.e.snaps.get(whole).?.len);
    f.e.dropState(id);
    f.e.dropState(whole);
    f.e.dropped.clearRetainingCapacity();
}

test "GLM snapshots: TF_GLM_KDA_REPLAY_CHECK reports a full entry's replay and serves the stored state" {
    var d = rec.init();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const io = t.io;
    const real = try tmp.dir.realPathFileAlloc(io, ".", t.allocator);
    defer t.allocator.free(real);
    // TF_KVPROBE_REPLAY_DUMP: an existing directory the dumps stay in, for tools/kvprobe/replay_check.py's format check
    const dir: []const u8 = if (std.c.getenv("TF_KVPROBE_REPLAY_DUMP")) |v| std.mem.span(v) else real;
    const dir_z = try t.allocator.dupeSentinel(u8, dir, 0);
    defer t.allocator.free(dir_z);
    _ = setenv("TF_GLM_HOST_CACHE", "1", 1);
    _ = setenv("TF_GLM_KV_DUMP", dir_z.ptr, 1);
    defer inline for (.{ "TF_GLM_HOST_CACHE", "TF_GLM_KV_DUMP" }) |x| {
        _ = unsetenv(x);
    };
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    var pass: FakePass = .{ .f = &f };
    f.e.kda_replay = try kda_replay.Replay.create(t.allocator, 256, false, true, pass.runner());
    try f.e.prepareSnapshots(2 * f.e.stateBytes(8192));
    const at = 4096;
    try f.mark(at, at, 5);
    saneStates(&f);
    try notePrompt(f.e.kda_replay.?, at);
    const saved = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
    defer t.allocator.free(saved);
    pass.stored = saved;
    pass.stored_cur = f.seq.st.cur[0];
    const id = try f.e.keepState();
    try t.expectEqual(kda_replay.Form.full, f.e.kda_replay.?.form(id).?);
    try t.expectEqual(try fullPlanLen(&f, at), f.e.snaps.get(id).?.len);
    @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0x5A);
    try f.seq.st.restoreHost(f.e.stream, 17, 18, &.{0}, .{ 0, 0 });
    const before = cuda.usage(false).device;
    try f.e.restoreState(id);
    try t.expectEqual(before, cuda.usage(false).device); // the scratch copy of the head is freed
    try t.expectEqual(@as(usize, 1), pass.calls);
    try t.expectEqual(@as(usize, at - 256), pass.start);
    // served: the stored state byte for byte, its parity and position
    try f.expectPrefix(saved, at, at);
    try t.expectEqual(pass.stored_cur, f.seq.st.cur[0]);
    try t.expectEqual(@as(usize, at), f.seq.st.pos);
    try t.expectEqual(@as(usize, 0), f.seq.st.pending);
    // reported: every head but head 1 bit-identical, the conv history identical everywhere
    const sum = f.e.kda_replay.?.last;
    const heads = f.seq.st.heads * f.seq.st.kda_layers;
    try t.expectEqual(heads, sum.heads);
    try t.expectEqual(heads - 1, sum.rec_same);
    try t.expectEqual(heads, sum.conv_same);
    try t.expectEqual(@as(f64, 0.5), sum.rec_abs_max);
    try t.expect(sum.rec_rel_max > 1e-3 and sum.rec_rel_max < 1e-2); // 0.5 against a head of norm about 140
    try t.expectEqual(@as(f64, 0), sum.conv_rel_max);
    // dumped: both heads as head-only keeps kvdump.py reads (one prefix part the length of the head)
    const head = headLen(&f);
    for ([_][]const u8{ "stored", "replayed" }) |tag| {
        const stem = try std.fmt.allocPrint(t.allocator, "{s}/kda-replay-rank0-keep{d}-w256-{s}", .{ dir, id, tag });
        defer t.allocator.free(stem);
        const bin = try std.fmt.allocPrint(t.allocator, "{s}.bin", .{stem});
        defer t.allocator.free(bin);
        const data = try std.Io.Dir.cwd().readFileAlloc(io, bin, t.allocator, .unlimited);
        defer t.allocator.free(data);
        try t.expectEqual(head, data.len);
        if (std.mem.eql(u8, tag, "stored")) try t.expectEqualSlices(u8, saved[0..head], data);
        const plan_path = try std.fmt.allocPrint(t.allocator, "{s}.plan", .{stem});
        defer t.allocator.free(plan_path);
        const plan = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, t.allocator, .unlimited);
        defer t.allocator.free(plan);
        const line = try std.fmt.allocPrint(t.allocator, "part 0 prefix {d} 0 -\nend\n", .{head});
        defer t.allocator.free(line);
        try t.expect(std.mem.endsWith(u8, plan, line));
        try t.expect(std.mem.indexOf(u8, plan, "replay ") == null);
    }
    f.e.dropState(id);
    f.e.dropped.clearRetainingCapacity();
}

test "GLM snapshots: a replay-form keep through TF_GLM_KV_CODEC dumps a manifest marked with its window" {
    var d = rec.init();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const io = t.io;
    const dir = try tmp.dir.realPathFileAlloc(io, ".", t.allocator);
    defer t.allocator.free(dir);
    const dir_z = try t.allocator.dupeSentinel(u8, dir, 0);
    defer t.allocator.free(dir_z);
    inline for (.{ .{ "TF_GLM_HOST_CACHE", "1" }, .{ "TF_GLM_KV_CODEC", "1" } }) |x| _ = setenv(x[0], x[1], 1);
    _ = setenv("TF_GLM_KV_DUMP", dir_z.ptr, 1);
    defer inline for (.{ "TF_GLM_HOST_CACHE", "TF_GLM_KV_CODEC", "TF_GLM_KV_DUMP" }) |x| {
        _ = unsetenv(x);
    };
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    var pass: FakePass = .{ .f = &f };
    f.e.kda_replay = try kda_replay.Replay.create(t.allocator, 2048, true, false, pass.runner());
    try f.e.prepareSnapshots(2 * f.e.stateBytes(4096));
    try markCompressible(&f, 4096, 4096);
    try notePrompt(f.e.kda_replay.?, 4096);
    const saved = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
    defer t.allocator.free(saved);
    const id = try f.e.keepState();
    const path = try std.fmt.allocPrint(t.allocator, "{s}/glm-rank0-keep{d}.plan", .{ dir, id });
    defer t.allocator.free(path);
    const plan = try std.Io.Dir.cwd().readFileAlloc(io, path, t.allocator, .unlimited);
    defer t.allocator.free(plan);
    try t.expect(std.mem.indexOf(u8, plan, "\nreplay 2048\n") != null);
    try t.expect(std.mem.indexOf(u8, plan, " prefix ") == null);
    @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0xA5);
    try f.e.restoreState(id);
    try t.expectEqual(@as(usize, 2048), pass.n);
    try t.expectEqual(@as(usize, 2048), pass.start);
    const head = headLen(&f);
    const expected = try t.allocator.dupe(u8, saved);
    defer t.allocator.free(expected);
    @memcpy(expected[0..head], rec.bytes(f.seq.st.arena.buf.ptr, head)); // the head is the pass's (checked above)
    try f.expectPrefix(expected, 4096, 4096);
    f.e.dropState(id);
    f.e.dropped.clearRetainingCapacity();
}
