//! TF_GLM_KV_CODEC_LAZY: kept states stored raw and compacted later on the codec's workers, through the actual GLM
//! rank owner with the recording driver: every form restores byte for byte, a restore stops a compaction under way,
//! pages and the store's charge follow each transition, and two ranks agree on a compacted state's pages.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const t = std.testing;
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;
const pc = @import("engine_api").prompt_cache;
const adapter = @import("copies");

const page = glm.snapshots.pages.page;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

const knobs = [_][:0]const u8{ "TF_GLM_HOST_CACHE", "TF_GLM_KV_CODEC", "TF_GLM_KV_CODEC_LAZY", "TF_GLM_KV_CODEC_LAZY_MS", "TF_GLM_KV_CODEC_THREADS", "TF_GLM_KV_CODEC_CPUS" };

/// The host cache, the codec and its lazy mode, with `delay` milliseconds before a state is compacted.
fn lazyEnv(delay: [:0]const u8) void {
    for ([_][:0]const u8{ "TF_GLM_HOST_CACHE", "TF_GLM_KV_CODEC", "TF_GLM_KV_CODEC_LAZY" }) |k| _ = setenv(k, "1", 1);
    _ = setenv("TF_GLM_KV_CODEC_LAZY_MS", delay, 1);
    _ = setenv("TF_GLM_KV_CODEC_THREADS", "3", 1);
}

fn clearEnv() void {
    for (knobs) |k| _ = unsetenv(k);
}

/// `f.mark` with a compressible GLM-shaped state, so a compacted copy needs fewer pages than the raw one.
fn markCompressible(f: *Fixture, at: usize) !void {
    try f.mark(at, at, 0);
    const s = &f.seq.st;
    @import("core").kv_synth.fill(.state_mix, rec.bytes(s.arena.buf.ptr, s.arena.buf.len), at);
    try s.restoreHost(f.e.stream, at, at, &.{@as(u1, @truncate(at))}, .{ if (at > 4096) at / 4 * 4 else 0, if (at > 4096) (at -| 8) / 4 * 4 else 0 });
}

fn live(f: *Fixture) []u8 {
    return rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len);
}

/// Scribble over the live state, restore `id`, and compare with `saved` through the fixture's own read-set oracle.
fn restoreSame(f: *Fixture, id: u64, saved: []const u8, at: usize, fill: u8) !void {
    @memset(live(f), fill);
    try f.seq.st.restoreHost(f.e.stream, 17, 18, &.{0}, .{ 0, 0 });
    try f.e.restoreState(id);
    try f.expectPrefix(saved, at, at);
    try t.expectEqual(at, f.seq.st.pos);
}

fn arena(f: *Fixture) u64 {
    return f.e.snap_pages.?.arena.?.len();
}

fn leasePages(f: *Fixture, id: u64) usize {
    return f.e.snaps.get(id).?.lease.ids.items.len;
}

test "GLM snapshots: a lazy keep is served raw, compacted later, served compacted, and charged its pages at each step" {
    defer clearEnv();
    lazyEnv("3600000"); // never on its own: `flush` decides when
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    const at = 8192;
    try f.e.prepareSnapshots(3 * f.e.stateBytes(at));
    const c = &f.e.snap_codec.?;
    try t.expect(c.lazy);
    try markCompressible(&f, at);
    const saved = try t.allocator.dupe(u8, live(&f));
    defer t.allocator.free(saved);
    const raw_pages = f.e.stateBytes(at) / page;
    const id = try f.e.keepState();
    const x = f.e.snaps.get(id).?.compact.?;
    try t.expect(x.cuts.len >= 3);
    // raw: the keep returned before any compaction, and holds and is charged its raw pages
    try t.expectEqual(@as(usize, 0), x.done);
    try t.expectEqual(raw_pages, leasePages(&f, id));
    try t.expectEqual(raw_pages * page, glm.snapshots.charged(&f.e, id));
    try t.expectEqual(arena(&f), glm.snapshots.charged(&f.e, id) + glm.snapshots.spare(&f.e));
    try restoreSame(&f, id, saved, at, 0x5A);
    try t.expectEqual(@as(u64, 1), c.sh.totals.raw_restores);
    // compacted: the pages stay the lease's until the store asks again (alone, it is trimmed then)
    c.sh.flush();
    try t.expect(x.full());
    try t.expect(x.pages() < raw_pages);
    try t.expectEqual(raw_pages, leasePages(&f, id));
    try t.expectEqual(raw_pages * page, glm.snapshots.charged(&f.e, id));
    try t.expectEqual(x.pages() * page, glm.snapshots.recharged(&f.e, id));
    try t.expectEqual(x.pages(), leasePages(&f, id));
    try t.expectEqual(x.pages() * page, glm.snapshots.charged(&f.e, id));
    try t.expectEqual(arena(&f), glm.snapshots.charged(&f.e, id) + glm.snapshots.spare(&f.e));
    try restoreSame(&f, id, saved, at, 0xA5);
    try restoreSame(&f, id, saved, at, 0x3C); // a decode does not consume the pages
    try t.expectEqual(@as(u64, 1), c.sh.totals.compacted);
    try t.expectEqual(@as(u64, 3), c.sh.totals.restores);
    try t.expect(c.sh.totals.lazy_ns > 0);
    try t.expectEqual(@as(u64, 0), c.sh.totals.encode_ns); // nothing encoded inside the keep
    f.e.dropState(id);
    try t.expectEqual(arena(&f), glm.snapshots.spare(&f.e));
    try t.expectEqual(f.e.snap_pages.?.pages.total(), f.e.snap_pages.?.pages.available());
}

test "GLM snapshots: a restore racing a compaction stops it at a wave's end and serves the part-compacted form" {
    defer clearEnv();
    lazyEnv("0");
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    const at = 8192;
    try f.e.prepareSnapshots(3 * f.e.stateBytes(at));
    const sh = f.e.snap_codec.?.sh;
    sh.lock();
    sh.wave_blocks = 1;
    sh.hold_after = 1; // the compactor stands still after one block of an entry
    sh.unlock();
    try markCompressible(&f, at);
    const saved = try t.allocator.dupe(u8, live(&f));
    defer t.allocator.free(saved);
    const id = try f.e.keepState();
    const x = f.e.snaps.get(id).?.compact.?;
    sh.waitPaused();
    try restoreSame(&f, id, saved, at, 0x5A); // pins: the compactor stops after its block, and cannot take it back
    sh.lock();
    const counts = sh.totals; // read under the lock; checked outside it, so a failure cannot hold it
    sh.unlock();
    try t.expectEqual(@as(u64, 1), counts.partial_restores);
    try t.expectEqual(@as(u64, 1), counts.cancels);
    sh.waitPaused(); // resumed after the restore (no delay): one block more, then still again
    sh.lock();
    const done = x.done;
    sh.hold_after = null;
    sh.unlock();
    try t.expectEqual(@as(usize, 2), done); // resumed from block 1, not from the start
    sh.flush();
    try t.expect(x.full());
    try t.expectEqual(x.pages() * page, glm.snapshots.recharged(&f.e, id));
    try restoreSame(&f, id, saved, at, 0xA5);
    f.e.dropState(id);
    // unscripted: restores and drops land wherever the compactor is, a block at a time
    for (0..12) |i| {
        const where = ([_]usize{ 2048, 4096, 8192 })[i % 3];
        try markCompressible(&f, where);
        const again = try t.allocator.dupe(u8, live(&f));
        defer t.allocator.free(again);
        const k = try f.e.keepState();
        for (0..i % 4 + 1) |_| try restoreSame(&f, k, again, where, @intCast(0x40 + i));
        f.e.dropState(k);
    }
    // a drop during a compaction waits for its block, then every page goes back
    sh.lock();
    sh.hold_after = 2;
    sh.unlock();
    try markCompressible(&f, at);
    const last = try f.e.keepState();
    sh.waitPaused();
    f.e.dropState(last);
    try t.expectEqual(arena(&f), glm.snapshots.spare(&f.e));
    try t.expectEqual(f.e.snap_pages.?.pages.total(), f.e.snap_pages.?.pages.available());
}

test "GLM snapshots: the store charges a lazily compacted state its compacted pages from its next admission" {
    defer clearEnv();
    lazyEnv("3600000");
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    const at = 8192;
    const fit = 8; // raw states the budget holds
    const budget = fit * f.e.stateBytes(at);
    _ = try glm.native.snap_ready(&f.e, budget);
    const family: pc.Snapshots = .{ .ptr = &f.e, .vtable = &adapter.Snaps(glm.native, null).vtable };
    var store = pc.Store.init(t.allocator, family, adapter.cacheRules(glm.native), budget);
    defer store.deinit();
    var prompt: [at + 1]u32 = @splat(1);
    try markCompressible(&f, at);
    try t.expect(store.keep(&prompt, at, null, &.{}, &.{}));
    const raw = store.held;
    try t.expect(raw > 0 and raw <= f.e.stateBytes(at));
    try t.expectEqual(budget, store.held + glm.native.snap_spare(&f.e));
    f.e.snap_codec.?.sh.flush();
    try t.expectEqual(raw, store.held); // compacted, but not yet asked
    try t.expectEqual(budget, store.held + glm.native.snap_spare(&f.e));
    store.collectExpired(); // what every admission runs first
    const kept: *glm.engine.Kept = @ptrCast(@alignCast(store.entries.items[0].saved));
    const x = f.e.snaps.get(kept.id).?.compact.?;
    try t.expectEqual(@as(u64, x.pages()) * page, store.held);
    try t.expect(store.held < raw);
    try t.expectEqual(budget, store.held + glm.native.snap_spare(&f.e));
    // the room compacted states give back admits more states than raw ones fit, without an eviction
    var states: usize = 1;
    for (2..3 * fit) |i| {
        prompt[0] = @intCast(i);
        try markCompressible(&f, at);
        if (!glm.native.snap_reuses(&f.e, at)) break;
        try t.expect(store.keep(&prompt, at, null, &.{}, &.{}));
        f.e.snap_codec.?.sh.flush();
        store.collectExpired();
        states += 1;
        try t.expectEqual(budget, store.held + glm.native.snap_spare(&f.e));
    }
    try t.expect(states > fit);
    try t.expectEqual(@as(u64, 0), store.counts.evicted);
    try t.expectEqual(states, store.entries.items.len);
}

/// Two ranks' `shared` (an all-reduce max) over threads.
const Board = struct {
    m: std.c.pthread_mutex_t = .{},
    cv: std.c.pthread_cond_t = .{},
    arrived: usize = 0,
    gen: u64 = 0,
    most: f64 = 0,
    out: f64 = 0,

    fn shared(b: *Board, v: f64) f64 {
        _ = std.c.pthread_mutex_lock(&b.m);
        defer _ = std.c.pthread_mutex_unlock(&b.m);
        b.most = if (b.arrived == 0) v else @max(b.most, v);
        b.arrived += 1;
        if (b.arrived == 2) {
            b.out = b.most;
            b.arrived = 0;
            b.gen += 1;
            _ = std.c.pthread_cond_broadcast(&b.cv);
        } else {
            const g = b.gen;
            while (b.gen == g) _ = std.c.pthread_cond_wait(&b.cv, &b.m);
        }
        return b.out;
    }
};

const Side = struct {
    board: *Board,
    calls: usize = 0,
    pub fn shared(s: *Side, v: f64) !f64 {
        s.calls += 1;
        return s.board.shared(v);
    }
};

fn agreeBoth(f0: *Fixture, f1: *Fixture, s0: *Side, s1: *Side) !void {
    const Peer = struct {
        fn run(f: *Fixture, s: *Side, err: *?anyerror) void {
            glm.snapshots.codec.agreeLazy(false, &f.e.snaps, &f.e.snap_pages.?, &f.e.snap_codec.?, s) catch |e| {
                err.* = e;
            };
        }
    };
    var failed: ?anyerror = null;
    const th = try std.Thread.spawn(.{}, Peer.run, .{ f1, s1, &failed });
    const mine = glm.snapshots.codec.agreeLazy(true, &f0.e.snaps, &f0.e.snap_pages.?, &f0.e.snap_codec.?, s0);
    th.join();
    try mine;
    if (failed) |e| return e;
}

test "GLM snapshots: two ranks agree on a lazily compacted state's pages only once both copies are compacted" {
    defer clearEnv();
    lazyEnv("3600000");
    var d = rec.init();
    var r: [2]Fixture = undefined;
    const at = 8192;
    var saved: [2][]u8 = undefined;
    for (&r, &saved, 0..) |*f, *s, rank| {
        try f.init(&d, rank, 2, true);
        try f.e.prepareSnapshots(3 * f.e.stateBytes(at));
        try markCompressible(f, at);
        s.* = try t.allocator.dupe(u8, live(f));
        try f.e.keepAs(1); // rank 0 keeps and announces id 1; the gate has the other rank keep it under that id
    }
    defer for (&r, saved) |*f, s| {
        t.allocator.free(s);
        f.deinit();
    };
    const raw = [2]usize{ leasePages(&r[0], 1), leasePages(&r[1], 1) };
    var board: Board = .{};
    var s0: Side = .{ .board = &board };
    var s1: Side = .{ .board = &board };
    // nothing compacted: the leader proposes nothing, one word each
    try agreeBoth(&r[0], &r[1], &s0, &s1);
    try t.expectEqual(@as(usize, 1), s0.calls);
    try t.expectEqual(@as(usize, 1), s1.calls);
    // the leader's copy compacted, the other raw: proposed, refused, no rank gives a page back
    r[0].e.snap_codec.?.sh.flush();
    const x0 = r[0].e.snaps.get(1).?.compact.?;
    const x1 = r[1].e.snaps.get(1).?.compact.?;
    try t.expect(x0.full() and !x1.full());
    try agreeBoth(&r[0], &r[1], &s0, &s1);
    try t.expectEqual(@as(usize, 3), s0.calls);
    try t.expectEqual(@as(usize, 3), s1.calls);
    for (r, raw) |f, n| try t.expectEqual(n, leasePages(@constCast(&f), 1));
    try t.expect(!x0.agreed);
    try t.expectEqual(@as(u64, raw[0]) * page, glm.snapshots.recharged(&r[0].e, 1)); // still raw, still charged so
    try restoreSame(&r[0], 1, saved[0], at, 0x11); // compacted copy
    try restoreSame(&r[1], 1, saved[1], at, 0x22); // raw copy
    // both compacted: every rank keeps the most pages either copy needs
    r[1].e.snap_codec.?.sh.flush();
    try agreeBoth(&r[0], &r[1], &s0, &s1);
    const most = @max(x0.pages(), x1.pages());
    try t.expect(most < @min(raw[0], raw[1]));
    for (&r) |*f| try t.expectEqual(most, leasePages(f, 1));
    try t.expect(x0.agreed and x1.agreed);
    // the leader's store sees the room only when it asks again; until then the spare count holds it out
    try t.expectEqual(arena(&r[0]), @as(u64, raw[0]) * page + glm.snapshots.spare(&r[0].e));
    try t.expectEqual(@as(u64, most) * page, glm.snapshots.recharged(&r[0].e, 1));
    try t.expectEqual(arena(&r[0]), @as(u64, most) * page + glm.snapshots.spare(&r[0].e));
    for (&r, saved, 0..) |*f, s, i| try restoreSame(f, 1, s, at, @intCast(0x30 + i));
    // agreed: nothing left to propose
    const before = s0.calls;
    try agreeBoth(&r[0], &r[1], &s0, &s1);
    try t.expectEqual(before + 1, s0.calls);
}

test "GLM snapshots: lazy and worker settings are refused by name before any state is kept" {
    defer clearEnv();
    var d = rec.init();
    const Case = struct { key: [:0]const u8, value: [:0]const u8, err: anyerror };
    const cases = [_]Case{
        .{ .key = "TF_GLM_KV_CODEC_CPUS", .value = "0-3,2", .err = error.KvCodecCpus },
        .{ .key = "TF_GLM_KV_CODEC_CPUS", .value = "1023", .err = error.KvCodecCpuNotAllowed },
        .{ .key = "TF_GLM_KV_CODEC_LAZY_MS", .value = "soon", .err = error.KvCodecLazyMs },
        .{ .key = "TF_GLM_KV_CODEC_LAZY_MS", .value = "3600001", .err = error.KvCodecLazyMs },
    };
    for (cases) |x| {
        lazyEnv("0");
        _ = setenv(x.key, x.value, 1);
        var f: Fixture = undefined;
        try f.init(&d, 0, 1, true);
        defer f.deinit();
        if (x.err == error.KvCodecCpuNotAllowed and !@import("core").kv_codec.worker_pool.pinnable) continue;
        try t.expectError(x.err, f.e.prepareSnapshots(2 * f.e.stateBytes(2048)));
        try t.expect(f.e.snap_codec == null);
        clearEnv();
    }
    // lazy without the codec
    _ = setenv("TF_GLM_HOST_CACHE", "1", 1);
    _ = setenv("TF_GLM_KV_CODEC_LAZY", "1", 1);
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    try t.expectError(error.KvCodecLazyNeedsCodec, f.e.prepareSnapshots(2 * f.e.stateBytes(2048)));
    try t.expect(f.e.snap_pages == null);
}

test "GLM snapshots: pinned workers compact and restore a lazy state byte for byte" {
    if (!@import("core").kv_codec.worker_pool.pinnable) return error.SkipZigTest;
    defer clearEnv();
    lazyEnv("0");
    const mine = @import("core").kv_codec.worker_pool.current() orelse return error.SkipZigTest;
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try mine.format(&w);
    try w.writeByte(0);
    _ = setenv("TF_GLM_KV_CODEC_CPUS", @ptrCast(w.buffered().ptr), 1);
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    try f.e.prepareSnapshots(2 * f.e.stateBytes(4096));
    const sh = f.e.snap_codec.?.sh;
    try t.expectEqualDeep(mine, sh.cpus.?);
    try markCompressible(&f, 4096);
    const saved = try t.allocator.dupe(u8, live(&f));
    defer t.allocator.free(saved);
    const id = try f.e.keepState();
    sh.flush();
    try t.expectEqual(@as(usize, 0), sh.pool.refused());
    try restoreSame(&f, id, saved, 4096, 0x77);
    f.e.dropState(id);
}
