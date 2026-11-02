//! TF_GLM_KDA_REPLAY_INIT=fp8 through the actual rank owner with a recording driver: the coarse copy in a kept entry's
//! pages, its restore alone (W = 0) and under a replay, the refusals, the manifest and the check's two reports.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const t = std.testing;
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;
const kda_replay = glm.engine.kda_replay;
const coarse = kda_replay.coarse;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// The replay pass, recorded: the conv history and half 0 it starts from, then (as the engine's) halves flipped,
/// states written (`stored`: the stored arena's live states, head 1 perturbed), the position moved.
const Pass = struct {
    f: *Fixture,
    stored: ?[]const u8 = null,
    stored_cur: u1 = 0,
    calls: usize = 0,
    start: usize = 0,
    n: usize = 0,
    first: usize = 0,
    seen: ?[]u8 = null, // conv history, then each layer's half 0, as the pass found them

    fn runner(x: *Pass) kda_replay.Runner {
        return .{ .ctx = x, .run = run };
    }

    fn run(ptr: *anyopaque, ids: []const u32) anyerror!void {
        const x: *Pass = @ptrCast(@alignCast(ptr));
        const st = &x.f.seq.st;
        const base = st.arena.buf.ptr;
        x.calls += 1;
        x.start = st.pos;
        x.n = ids.len;
        x.first = ids[0];
        const conv = rec.bytes(st.conv, st.kda_layers * 3 * st.conv_c * 2);
        if (x.seen == null) x.seen = try t.allocator.alloc(u8, conv.len + st.kda_layers * st.rec_layer);
        @memcpy(x.seen.?[0..conv.len], conv);
        for (0..st.kda_layers) |li| @memcpy(x.seen.?[conv.len + li * st.rec_layer ..][0..st.rec_layer], rec.bytes(st.recAt(0, li), st.rec_layer));
        if (x.stored) |s| @memcpy(conv, s[st.conv - base ..][0..conv.len]) else @memset(conv, 0x11);
        for (st.cur, 0..) |*c, li| {
            const out = std.mem.bytesAsSlice(f32, rec.bytes(st.recAt(1 - c.*, li), st.rec_layer));
            if (x.stored) |s| {
                @memcpy(std.mem.sliceAsBytes(out), s[st.recAt(x.stored_cur, li) - base ..][0..st.rec_layer]);
                out[128 * 128 + 5] += 0.5; // head 1 alone differs
            } else @memset(out, 0.25);
            c.* = 1 - c.*;
        }
        try st.setPos(x.f.e.stream, st.pos + ids.len);
    }

    fn deinit(x: *Pass) void {
        if (x.seen) |s| t.allocator.free(s);
    }
};

fn notePrompt(r: *kda_replay.Replay, at: usize) !void {
    var ids: [2048]u32 = undefined;
    var start: usize = 0;
    while (start < at) : (start += 2048) {
        const n = @min(2048, at - start);
        for (ids[0..n], 0..) |*x, i| x.* = @intCast(1000 + start + i);
        try r.note(0, start, ids[0..n]);
    }
}

/// Finite, varied KDA states and conv history (rows of different magnitudes, both signs), and one row of layer 0's
/// current half, head 0, exactly on the e4m3 grid times 2^-3 (its round trip must be bit-identical).
fn plantStates(f: *Fixture) void {
    const st = &f.seq.st;
    for (std.mem.bytesAsSlice(f32, rec.bytes(st.rec, 2 * st.kda_layers * st.rec_layer)), 0..) |*v, i| {
        const mag = std.math.ldexp(@as(f32, 1), @as(i32, @intCast(i / 128 % 23)) - 12);
        v.* = (@as(f32, @floatFromInt(i % 251)) - 125) / 125 * mag;
    }
    for (std.mem.bytesAsSlice(u16, rec.bytes(st.conv, st.kda_layers * 3 * st.conv_c * 2)), 0..) |*v, i| v.* = (0x3c00 + @as(u16, @intCast(i * 7 % 1021))) | (if (i % 3 == 0) @as(u16, 0x8000) else 0);
    const exact = std.mem.bytesAsSlice(f32, rec.bytes(st.recAt(st.cur[0], 0), 128 * 4));
    for (exact, 0..) |*v, i| v.* = e4m3Value(@intCast(i % 0x7F)) * 0.125;
}

fn e4m3Value(code: u8) f32 {
    return @import("core").exlr8.codec.e4m3Value(code);
}

/// The coarse copy of a saved arena's live states, decoded: what a restore writes, computed apart from it.
const Expected = struct {
    conv: []u8,
    rec: []u8, // each layer's half, in layer order

    fn of(f: *Fixture, saved: []const u8, cur: []const u1) !Expected {
        const st = &f.seq.st;
        const base = st.arena.buf.ptr;
        const s = try coarse.Shape.of(st);
        const per = st.rec_layer / 4;
        const recs = try t.allocator.alloc(f32, st.kda_layers * per);
        defer t.allocator.free(recs);
        for (0..st.kda_layers) |li| @memcpy(std.mem.sliceAsBytes(recs[li * per ..][0..per]), saved[st.recAt(cur[li], li) - base ..][0..st.rec_layer]);
        const convs = try t.allocator.alloc(u16, st.kda_layers * 3 * st.conv_c);
        defer t.allocator.free(convs);
        @memcpy(std.mem.sliceAsBytes(convs), saved[st.conv - base ..][0 .. convs.len * 2]);
        const copy = try t.allocator.alloc(u8, s.len());
        defer t.allocator.free(copy);
        coarse.encodeStates(s, copy, recs, convs);
        const out: Expected = .{ .conv = try t.allocator.alloc(u8, convs.len * 2), .rec = try t.allocator.alloc(u8, st.kda_layers * st.rec_layer) };
        errdefer out.deinit();
        try coarse.decodeConv(s, copy, @alignCast(std.mem.bytesAsSlice(u16, out.conv)));
        for (0..st.kda_layers) |li| try coarse.decodeRec(s, copy, li, @alignCast(std.mem.bytesAsSlice(f32, out.rec[li * st.rec_layer ..][0..st.rec_layer])));
        // the decoded states sit within e4m3's step of the saved ones, the planted grid row bit-identical
        for (std.mem.bytesAsSlice(f32, out.rec), recs, 0..) |y, x, i| {
            if (i < 128) try t.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(y)));
            try t.expect(@abs(y - x) <= @abs(x) / 16 + 1e-6);
        }
        return out;
    }

    fn deinit(x: Expected) void {
        t.allocator.free(x.conv);
        t.allocator.free(x.rec);
    }
};

fn headLen(f: *Fixture) usize {
    const st = &f.seq.st;
    return @intCast(st.caches[0].lc - st.arena.buf.ptr);
}

fn fullPlanLen(f: *Fixture, at: usize) !usize {
    return (try glm.snapshots.layout.Plan.init(&f.seq.st, &f.e.c, at, at)).len;
}

test "GLM snapshots: TF_GLM_KDA_REPLAY_INIT=fp8 keeps a coarse head in its pages and W = 0 restores it alone" {
    var d = rec.init();
    _ = setenv("TF_GLM_HOST_CACHE", "1", 1);
    defer _ = unsetenv("TF_GLM_HOST_CACHE");
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    var pass: Pass = .{ .f = &f };
    defer pass.deinit();
    f.e.kda_replay = try kda_replay.Replay.createInit(t.allocator, 0, .fp8, true, false, pass.runner());
    try f.e.prepareSnapshots(2 * f.e.stateBytes(8192));
    const at = 4096;
    try f.mark(at, at, 9);
    const st = &f.seq.st;
    try st.restoreHost(f.e.stream, at, at, &.{1}, .{ 0, 0 }); // the live half is half 1
    plantStates(&f);
    const saved = try t.allocator.dupe(u8, rec.bytes(st.arena.buf.ptr, st.arena.buf.len));
    defer t.allocator.free(saved);
    const want = try Expected.of(&f, saved, st.cur);
    defer want.deinit();
    // no tokens noted: a window of 0 needs none
    const id = try f.e.keepState();
    const kept = f.e.snaps.get(id).?;
    const head = headLen(&f);
    const c = (try coarse.Shape.of(st)).len();
    try t.expectEqual(kda_replay.Form.replay, f.e.kda_replay.?.form(id).?);
    try t.expectEqual(try fullPlanLen(&f, at) - head + c, kept.len); // the head's bytes, the coarse copy's instead
    try t.expect(c * 4 < head);
    @memset(rec.bytes(st.arena.buf.ptr, st.arena.buf.len), 0x5A);
    try st.restoreHost(f.e.stream, 17, 18, &.{0}, .{ 0, 0 });
    try f.e.restoreState(id);
    try t.expectEqual(@as(usize, 0), pass.calls); // the coarse copy alone, no pass
    try t.expectEqual(@as(usize, at), st.pos);
    try t.expectEqual(@as(u1, 0), st.cur[0]);
    // the head: zero but the position, the decoded conv history and half 0; every other byte the kept one
    const expected = try t.allocator.dupe(u8, saved);
    defer t.allocator.free(expected);
    @memset(expected[0..head], 0);
    const base = st.arena.buf.ptr;
    @memcpy(expected[st.pos_dev - base ..][0..4], std.mem.asBytes(&@as(i32, at)));
    @memcpy(expected[st.conv - base ..][0..want.conv.len], want.conv);
    for (0..st.kda_layers) |li| @memcpy(expected[st.recAt(0, li) - base ..][0..st.rec_layer], want.rec[li * st.rec_layer ..][0..st.rec_layer]);
    try f.expectPrefix(expected, at, at);
    // deterministic: a second restore writes the same bytes
    @memset(rec.bytes(st.arena.buf.ptr, head), 0xA5);
    try f.e.restoreState(id);
    try f.expectPrefix(expected, at, at);
    // refused by name: without the replay object, and under another init
    const r = f.e.kda_replay.?;
    f.e.kda_replay = null;
    try t.expectError(error.KdaReplayNeedsCoarseInit, f.e.restoreState(id));
    f.e.kda_replay = r;
    r.init = .zero;
    try t.expectError(error.KdaReplayInitDiffers, f.e.restoreState(id));
    r.init = .fp8;
    f.e.dropState(id);
    f.e.dropped.clearRetainingCapacity();
}

test "GLM snapshots: an fp8 entry restored with W = 64 replays its window on top of the decoded coarse head" {
    var d = rec.init();
    _ = setenv("TF_GLM_HOST_CACHE", "1", 1);
    defer _ = unsetenv("TF_GLM_HOST_CACHE");
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    var pass: Pass = .{ .f = &f };
    defer pass.deinit();
    f.e.kda_replay = try kda_replay.Replay.createInit(t.allocator, 64, .fp8, true, false, pass.runner());
    try f.e.prepareSnapshots(2 * f.e.stateBytes(8192));
    const at = 4096;
    try f.mark(at, at, 4);
    plantStates(&f);
    try notePrompt(f.e.kda_replay.?, at);
    const st = &f.seq.st;
    const saved = try t.allocator.dupe(u8, rec.bytes(st.arena.buf.ptr, st.arena.buf.len));
    defer t.allocator.free(saved);
    const want = try Expected.of(&f, saved, st.cur);
    defer want.deinit();
    const id = try f.e.keepState();
    @memset(rec.bytes(st.arena.buf.ptr, st.arena.buf.len), 0x5A);
    try f.e.restoreState(id);
    try t.expectEqual(@as(usize, 1), pass.calls);
    try t.expectEqual(@as(usize, at - 64), pass.start);
    try t.expectEqual(@as(usize, 64), pass.n);
    try t.expectEqual(@as(usize, 1000 + at - 64), pass.first);
    try t.expectEqual(@as(usize, at), st.pos);
    // the pass started from the decoded copy, bit for bit
    try t.expectEqualSlices(u8, want.conv, pass.seen.?[0..want.conv.len]);
    try t.expectEqualSlices(u8, want.rec, pass.seen.?[want.conv.len..]);
    // a keep after a gap stays whole without a window
    try f.mark(8192, 8192, 3);
    try f.e.kda_replay.?.note(0, 8000, &.{ 1, 2, 3 });
    const whole = try f.e.keepState();
    try t.expect(f.e.kda_replay.?.form(whole) == null);
    try t.expectEqual(try fullPlanLen(&f, 8192), f.e.snaps.get(whole).?.len);
    f.e.dropState(id);
    f.e.dropState(whole);
    f.e.dropped.clearRetainingCapacity();
}

test "GLM snapshots: an fp8 entry through TF_GLM_KV_CODEC is compacted, its manifest naming the form and the copy" {
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
    var pass: Pass = .{ .f = &f };
    defer pass.deinit();
    f.e.kda_replay = try kda_replay.Replay.createInit(t.allocator, 64, .fp8, true, false, pass.runner());
    try f.e.prepareSnapshots(2 * f.e.stateBytes(4096));
    try f.mark(4096, 4096, 0);
    @import("core").kv_synth.fill(.state_mix, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 4096);
    try f.seq.st.restoreHost(f.e.stream, 4096, 4096, &.{0}, .{ 0, 0 }); // the fill wrote the device positions too
    plantStates(&f);
    try notePrompt(f.e.kda_replay.?, 4096);
    const st = &f.seq.st;
    const saved = try t.allocator.dupe(u8, rec.bytes(st.arena.buf.ptr, st.arena.buf.len));
    defer t.allocator.free(saved);
    const want = try Expected.of(&f, saved, st.cur);
    defer want.deinit();
    const id = try f.e.keepState();
    try t.expect(f.e.snaps.get(id).?.compact != null);
    const path = try std.fmt.allocPrint(t.allocator, "{s}/glm-rank0-keep{d}.plan", .{ dir, id });
    defer t.allocator.free(path);
    const plan = try std.Io.Dir.cwd().readFileAlloc(io, path, t.allocator, .unlimited);
    defer t.allocator.free(plan);
    try t.expect(std.mem.indexOf(u8, plan, "\nreplay 64 init fp8 row 128\n") != null);
    const part = try std.fmt.allocPrint(t.allocator, "\npart 0 prefix {d} 0 -\n", .{(try coarse.Shape.of(st)).len()});
    defer t.allocator.free(part);
    try t.expect(std.mem.indexOf(u8, plan, part) != null);
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, plan, " prefix "));
    @memset(rec.bytes(st.arena.buf.ptr, st.arena.buf.len), 0xA5);
    try f.e.restoreState(id);
    try t.expectEqualSlices(u8, want.conv, pass.seen.?[0..want.conv.len]);
    try t.expectEqualSlices(u8, want.rec, pass.seen.?[want.conv.len..]);
    const expected = try t.allocator.dupe(u8, saved);
    defer t.allocator.free(expected);
    const head = headLen(&f);
    @memcpy(expected[0..head], rec.bytes(st.arena.buf.ptr, head)); // the head is the pass's (checked above)
    try f.expectPrefix(expected, 4096, 4096);
    f.e.dropState(id);
    f.e.dropped.clearRetainingCapacity();
}

test "GLM snapshots: TF_GLM_KDA_REPLAY_CHECK with fp8 reports the coarse copy alone and with its window, serving the stored state" {
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
    for ([_]usize{ 256, 0 }) |w| {
        var f: Fixture = undefined;
        try f.init(&d, 0, 1, true);
        defer f.deinit();
        var pass: Pass = .{ .f = &f };
        defer pass.deinit();
        f.e.kda_replay = try kda_replay.Replay.createInit(t.allocator, w, .fp8, false, true, pass.runner());
        try f.e.prepareSnapshots(2 * f.e.stateBytes(8192));
        const at = 4096;
        try f.mark(at, at, 5);
        plantStates(&f);
        try notePrompt(f.e.kda_replay.?, at);
        const st = &f.seq.st;
        const saved = try t.allocator.dupe(u8, rec.bytes(st.arena.buf.ptr, st.arena.buf.len));
        defer t.allocator.free(saved);
        pass.stored = saved;
        pass.stored_cur = st.cur[0];
        const want = try Expected.of(&f, saved, st.cur);
        defer want.deinit();
        const id = try f.e.keepState();
        try t.expectEqual(kda_replay.Form.full, f.e.kda_replay.?.form(id).?);
        try t.expectEqual(try fullPlanLen(&f, at), f.e.snaps.get(id).?.len);
        @memset(rec.bytes(st.arena.buf.ptr, st.arena.buf.len), 0x5A);
        try st.restoreHost(f.e.stream, 17, 18, &.{0}, .{ 0, 0 });
        const before = cuda.usage(false).device;
        try f.e.restoreState(id);
        try t.expectEqual(before, cuda.usage(false).device);
        // served: the stored state byte for byte
        try f.expectPrefix(saved, at, at);
        try t.expectEqual(pass.stored_cur, st.cur[0]);
        try t.expectEqual(@as(usize, at), st.pos);
        // (a) the coarse copy alone: every head within e4m3's step (relative L2 under 1/16), none of the conv history
        // exact but within it too
        const r = f.e.kda_replay.?;
        const heads = st.heads * st.kda_layers;
        try t.expectEqual(heads, r.last_coarse.heads);
        try t.expect(r.last_coarse.rec_rel_max > 0 and r.last_coarse.rec_rel_max < 0.0625);
        try t.expect(r.last_coarse.conv_rel_max < 0.0625);
        // (b) the copy plus the window: the pass ran from the decoded copy and (here) reproduced all but head 1
        if (w > 0) {
            try t.expectEqual(@as(usize, 1), pass.calls);
            try t.expectEqual(@as(usize, at - w), pass.start);
            try t.expectEqualSlices(u8, want.rec, pass.seen.?[want.conv.len..]);
            try t.expectEqual(heads - 1, r.last.rec_same);
            try t.expectEqual(heads, r.last.conv_same);
        } else {
            try t.expectEqual(@as(usize, 0), pass.calls);
            try t.expectEqual(r.last_coarse, r.last);
        }
        // dumped: stored, coarse and (with a window) replayed, as full-size head-only keeps under an fp8 stem
        const head = headLen(&f);
        for ([_][]const u8{ "stored", "coarse", "replayed" }) |tag| {
            const bin = try std.fmt.allocPrint(t.allocator, "{s}/kda-replay-rank0-keep{d}-w{d}-fp8-{s}.bin", .{ dir, id, w, tag });
            defer t.allocator.free(bin);
            const data = std.Io.Dir.cwd().readFileAlloc(io, bin, t.allocator, .unlimited) catch |err| {
                if (w == 0 and std.mem.eql(u8, tag, "replayed") and err == error.FileNotFound) continue;
                return err;
            };
            defer t.allocator.free(data);
            try t.expect(!(w == 0 and std.mem.eql(u8, tag, "replayed")));
            try t.expectEqual(head, data.len);
            if (std.mem.eql(u8, tag, "stored")) try t.expectEqualSlices(u8, saved[0..head], data);
            if (std.mem.eql(u8, tag, "coarse")) try t.expectEqualSlices(u8, want.conv, data[st.conv - st.arena.buf.ptr ..][0..want.conv.len]);
        }
        f.e.dropState(id);
        f.e.dropped.clearRetainingCapacity();
    }
}
