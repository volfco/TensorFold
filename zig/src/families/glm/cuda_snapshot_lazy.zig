//! A kept state's compaction, now (TF_GLM_KV_CODEC) or later off the prompt pass (TF_GLM_KV_CODEC_LAZY=1), on the
//! codec's persistent workers. An entry goes raw -> partly compacted -> compacted in place in its own pages; every
//! form restores byte for byte (cuda_snapshot_codec.restore), and the pages past a compacted state go back only on
//! the engine's thread (alone at once, with peers once every rank agrees: `agree`).
const std = @import("std");
const core = @import("core");
const codec = core.kv_codec;
const workers = codec.worker_pool;
const pages = @import("cuda_snapshot_pages.zig");
const c = std.c;

const page = pages.page;

const Timespec = extern struct { sec: isize, nsec: isize };
extern "c" fn clock_gettime(clock: c_int, ts: *Timespec) c_int;

pub fn now() u64 {
    return clock(1); // CLOCK_MONOTONIC
}

fn clock(id: c_int) u64 {
    var ts: Timespec = undefined;
    _ = clock_gettime(id, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// A lease's pages as one stream: byte `at` is page ids[at / page], offset at % page.
pub const View = struct {
    base: []u8,
    ids: []const u32,

    fn at(v: View, off: usize, len: usize) []u8 {
        std.debug.assert(off % page + len <= page);
        return v.base[@as(usize, v.ids[off / page]) * page + off % page ..][0..len];
    }

    /// `bytes` to stream offset `off` (ascending: a source further on in the same stream is never overwritten first).
    pub fn write(v: View, off: usize, bytes: []const u8) void {
        var done: usize = 0;
        while (done < bytes.len) {
            const o = off + done;
            const n = @min(page - o % page, bytes.len - done);
            std.mem.copyForwards(u8, v.at(o, n), bytes[done..][0..n]);
            done += n;
        }
    }

    /// `len` bytes at `off`: a slice when they lie in one page, else gathered into `into`.
    pub fn read(v: View, off: usize, len: usize, into: []u8) []const u8 {
        if (off % page + len <= page) return v.at(off, len);
        var done: usize = 0;
        while (done < len) {
            const o = off + done;
            const n = @min(page - o % page, len - done);
            @memcpy(into[done..][0..n], v.at(o, n));
            done += n;
        }
        return into[0..len];
    }
};

/// One kept state's codec side. Blocks [0, done) are compacted, their payloads at stream bytes [0, out); the raw
/// bytes from `rawFrom()` on are where the keep put them. `busy`, `pinned`, `cancel`, `done` and `out` change under
/// `Shared.m`; the compactor alone writes the pages while `busy`, and the engine reads them only while `pinned`.
pub const Entry = struct {
    id: u64,
    len: usize,
    base: []u8,
    ids: []u32, // the lease's page ids at keep (its first pages stay the lease's after a trim)
    cuts: []codec.Cut,
    blocks: []codec.Block,
    done: usize = 0,
    out: usize = 0,
    kept: u64, // when the keep returned
    touched: u64, // the keep or the last restore: the compactor takes the least recently touched first
    busy: bool = false,
    pinned: bool = false,
    cancel: bool = false,
    remembered: bool = false, // its blocks count toward `duplicate`
    agreed: bool = false, // with peers: every rank trimmed it to the most pages any rank's copy needs
    tries: u32 = 0,
    unclaimed: usize = 0, // pages it gave back that the store's charge has not seen yet

    pub fn full(x: *const Entry) bool {
        return x.done == x.cuts.len;
    }

    pub fn rawFrom(x: *const Entry) usize {
        return if (x.full()) x.len else x.cuts[x.done].off;
    }

    /// The pages the payloads fill (a compacted entry's lease past these can go back).
    pub fn pages(x: *const Entry) usize {
        return @max(1, (x.out + page - 1) / page);
    }

    fn view(x: *const Entry) View {
        return .{ .base = x.base, .ids = x.ids };
    }
};

pub const Totals = struct {
    keeps: u64 = 0,
    compacted: u64 = 0,
    restores: u64 = 0,
    raw_restores: u64 = 0, // served from raw pages (resumed before its compaction began)
    partial_restores: u64 = 0, // part compacted, part raw (a compaction stopped for the restore)
    duplicate: u64 = 0, // raw bytes of blocks another held state already holds (a dedup tier's ceiling)
    raw: u64 = 0,
    stored: u64 = 0,
    raw_pages: u64 = 0,
    kept_pages: u64 = 0,
    encode_ns: u64 = 0, // inside a keep, on the prompt pass
    lazy_ns: u64 = 0, // on the compactor, off the prompt pass
    decode_ns: u64 = 0,
    cancels: u64 = 0,
};

/// Everything the engine's thread and the compactor share, at a fixed address (the Codec value moves).
pub const Shared = struct {
    gpa: std.mem.Allocator,
    rank: usize,
    m: c.pthread_mutex_t = .{},
    cv: c.pthread_cond_t = .{},
    work: c.pthread_mutex_t = .{}, // held while `scratch` and the workers encode or decode
    pool: workers.Pool,
    scratch: []codec.Scratch, // one a block of a wave
    payloads: [][]const u8,
    options: codec.Options,
    entries: std.ArrayList(*Entry) = .empty,
    held: std.AutoHashMapUnmanaged(u64, u32) = .empty, // block identities of the compacted states held, with counts
    totals: Totals = .{},
    unclaimed: usize = 0,
    lazy: bool,
    delay_ns: u64,
    thread: ?std.Thread = null,
    stop: bool = false,
    force: bool = false, // `flush`: every entry is due now
    cpus: ?workers.Cpus,
    wave_blocks: usize, // blocks a compactor wave takes (tests make it 1 to stop between blocks)
    hold_after: ?usize = null, // tests: the compactor waits after this many waves of an entry, until cancelled
    paused: bool = false,

    pub fn lock(sh: *Shared) void {
        _ = c.pthread_mutex_lock(&sh.m);
    }

    pub fn unlock(sh: *Shared) void {
        _ = c.pthread_mutex_unlock(&sh.m);
    }

    fn wait(sh: *Shared) void {
        _ = c.pthread_cond_wait(&sh.cv, &sh.m);
    }

    fn wake(sh: *Shared) void {
        _ = c.pthread_cond_broadcast(&sh.cv);
    }

    pub fn start(sh: *Shared) !void {
        if (sh.lazy) sh.thread = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, compactor, .{sh});
    }

    /// The compactor stops at its next wave's end; every entry must be gone (`remove`) before this.
    pub fn deinit(sh: *Shared) void {
        sh.lock();
        sh.stop = true;
        sh.wake();
        sh.unlock();
        if (sh.thread) |th| th.join();
        std.debug.assert(sh.entries.items.len == 0);
        sh.entries.deinit(sh.gpa);
        sh.held.deinit(sh.gpa);
        sh.pool.deinit();
        for (sh.scratch) |*s| s.deinit(sh.gpa);
        sh.gpa.free(sh.scratch);
        sh.gpa.free(sh.payloads);
    }

    /// A raw entry joins: eager, compacted here before the keep returns; lazy, queued for the compactor.
    pub fn add(sh: *Shared, x: *Entry) !void {
        sh.lock();
        defer sh.unlock();
        try sh.entries.append(sh.gpa, x);
        sh.totals.keeps += 1;
        sh.wake();
    }

    /// `x` leaves (its state was dropped): a compaction under way stops first, so its pages can go back.
    pub fn remove(sh: *Shared, x: *Entry) void {
        sh.lock();
        defer sh.unlock();
        settleBusy(sh, x);
        if (std.mem.indexOfScalar(*Entry, sh.entries.items, x)) |i| _ = sh.entries.swapRemove(i);
        if (x.remembered) forget(sh, x.blocks);
        sh.unclaimed -= x.unclaimed;
    }

    /// Hold `x` for a restore: a compaction under way stops at its wave's end, and none starts until `unpin`.
    pub fn pin(sh: *Shared, x: *Entry) void {
        sh.lock();
        defer sh.unlock();
        settleBusy(sh, x);
    }

    pub fn unpin(sh: *Shared, x: *Entry) void {
        sh.lock();
        defer sh.unlock();
        x.pinned = false;
        x.touched = now();
        sh.wake();
    }

    /// Pinned first, so the compactor cannot take `x` again between its stop and this thread's wake-up.
    fn settleBusy(sh: *Shared, x: *Entry) void {
        x.pinned = true;
        while (x.busy) {
            x.cancel = true;
            sh.wake();
            sh.wait();
        }
    }

    /// Every entry compacted now, whatever its delay (tests, and the end of a run); returns when none is left raw.
    pub fn flush(sh: *Shared) void {
        if (!sh.lazy) return;
        sh.lock();
        defer sh.unlock();
        sh.force = true;
        sh.wake();
        while (for (sh.entries.items) |x| {
            if (!x.full() and !x.pinned) break true;
        } else false) sh.wait();
        sh.force = false;
    }

    /// Tests: until the compactor stands still after `hold_after` waves.
    pub fn waitPaused(sh: *Shared) void {
        sh.lock();
        defer sh.unlock();
        while (!sh.paused) sh.wait();
    }

    /// Compact the whole of `x` on this thread (the eager keep).
    pub fn compactNow(sh: *Shared, x: *Entry) void {
        while (!x.full()) {
            const r = sh.step(x, sh.scratch.len);
            sh.lock();
            x.done += r.blocks;
            x.out = r.out;
            if (x.full()) finish(sh, x);
            sh.unlock();
        }
    }

    /// Encode `x`'s next blocks (at most `most`) and write their payloads after the last ones: each payload is at
    /// most its raw bytes, so a write never reaches a block not yet read. Neither `done` nor `out` moves here.
    fn step(sh: *Shared, x: *Entry, most: usize) struct { blocks: usize, out: usize, ns: u64 } {
        _ = c.pthread_mutex_lock(&sh.work);
        defer _ = c.pthread_mutex_unlock(&sh.work);
        const t0 = now();
        const from = x.done;
        const k = @min(most, sh.scratch.len, x.cuts.len - from);
        const Job = struct {
            sh: *Shared,
            view: View,
            cuts: []const codec.Cut,
            blocks: []codec.Block,
            fn run(j: *@This(), i: usize, _: usize) void {
                const cut = j.cuts[i];
                const sc = &j.sh.scratch[i];
                const r = codec.encodeBlock(sc, j.view.read(cut.off, cut.len, sc.input), cut.class, cut.row, j.sh.options);
                j.blocks[i] = r.block;
                j.sh.payloads[i] = r.payload;
            }
        };
        var job: Job = .{ .sh = sh, .view = x.view(), .cuts = x.cuts[from..][0..k], .blocks = x.blocks[from..][0..k] };
        sh.pool.run(k, &job, Job.run);
        var out = x.out;
        for (sh.payloads[0..k]) |p| {
            job.view.write(out, p);
            out += p.len;
        }
        return .{ .blocks = k, .out = out, .ns = now() - t0 };
    }

    /// Under `m`: `x` became compacted.
    fn finish(sh: *Shared, x: *Entry) void {
        sh.totals.compacted += 1;
        sh.totals.raw += x.len;
        sh.totals.stored += x.out;
        sh.totals.raw_pages += (x.len + page - 1) / page;
        sh.totals.kept_pages += x.pages();
        sh.totals.duplicate += remember(sh, x.blocks);
        x.remembered = true;
        if (sh.lazy) std.log.info("KVCODEC rank={d} compacted keep={d} raw={d} stored={d} pages {d} -> {d} after_ms={d:.1}", .{
            sh.rank, x.id, x.len, x.out, (x.len + page - 1) / page, x.pages(), @as(f64, @floatFromInt(now() - x.kept)) / 1e6,
        });
    }

    /// The due entry the compactor takes next (oldest touch first), or when the next one falls due.
    fn next(sh: *Shared, t: u64) union(enum) { take: *Entry, at: ?u64 } {
        var best: ?*Entry = null;
        var soonest: ?u64 = null;
        for (sh.entries.items) |x| {
            if (x.full() or x.busy or x.pinned) continue;
            const due = x.touched +| sh.delay_ns;
            if (sh.force or due <= t) {
                if (best == null or x.touched < best.?.touched) best = x;
            } else soonest = @min(soonest orelse due, due);
        }
        return if (best) |x| .{ .take = x } else .{ .at = soonest };
    }

    fn compactor(sh: *Shared) void {
        if (sh.cpus) |*set| {
            var cpu: usize = 0;
            workers.pin(set, &cpu) catch std.log.warn("KVCODEC rank={d}: the compactor could not be pinned to CPU {d}", .{ sh.rank, cpu });
        }
        sh.lock();
        defer sh.unlock();
        while (!sh.stop) {
            const x = switch (sh.next(now())) {
                .take => |x| x,
                .at => |at| {
                    if (at) |due| sleepUntil(sh, due) else sh.wait();
                    continue;
                },
            };
            x.busy = true;
            var waves: usize = 0;
            while (true) {
                sh.unlock();
                const r = sh.step(x, sh.wave_blocks);
                sh.lock();
                sh.totals.lazy_ns += r.ns;
                x.done += r.blocks;
                x.out = r.out;
                waves += 1;
                if (x.full()) {
                    finish(sh, x);
                    break;
                }
                if (sh.hold_after == waves) {
                    sh.paused = true;
                    sh.wake();
                    while (!x.cancel and !sh.stop and sh.hold_after != null) sh.wait();
                    sh.paused = false;
                }
                if (x.cancel or sh.stop) {
                    sh.totals.cancels += 1;
                    break;
                }
            }
            x.busy = false;
            x.cancel = false;
            sh.wake();
        }
    }

    fn sleepUntil(sh: *Shared, due: u64) void {
        const left = @min(due -| now(), 3600 * std.time.ns_per_s); // a wake-up re-reads the delay
        const at = clock(0) + left; // CLOCK_REALTIME, as pthread_cond_timedwait reads it
        const ts: c.timespec = .{ .sec = @intCast(at / std.time.ns_per_s), .nsec = @intCast(at % std.time.ns_per_s) };
        _ = c.pthread_cond_timedwait(&sh.cv, &sh.m, &ts);
    }

    /// With peers, the compacted entry the leader proposes next for the page agreement: the least tried, oldest
    /// first; 0 when none waits.
    pub fn candidate(sh: *Shared) u64 {
        sh.lock();
        defer sh.unlock();
        var best: ?*Entry = null;
        for (sh.entries.items) |x| if (x.full() and !x.busy and !x.agreed) {
            if (best == null or x.tries < best.?.tries or (x.tries == best.?.tries and x.id < best.?.id)) best = x;
        };
        return if (best) |x| x.id else 0;
    }

    /// The pages this rank's copy of `x` needs if it is compacted, else null.
    pub fn compactedPages(sh: *Shared, x: *Entry) ?usize {
        sh.lock();
        defer sh.unlock();
        return if (x.full() and !x.busy) x.pages() else null;
    }

    /// The store saw the entry's current pages: what it gave back is no longer set aside from the spare count.
    pub fn claim(sh: *Shared, x: *Entry) void {
        sh.lock();
        defer sh.unlock();
        sh.unclaimed -= x.unclaimed;
        x.unclaimed = 0;
    }

    /// Engine thread: `lease` keeps its first `n` pages; the rest go back (unclaimed until the store's next charge).
    pub fn trim(sh: *Shared, pool: *pages.Pool, lease: *pages.Lease, x: *Entry, n: usize) void {
        const before = lease.ids.items.len;
        if (n >= before) return;
        pool.trim(lease, n);
        sh.lock();
        defer sh.unlock();
        x.unclaimed += before - n;
        sh.unclaimed += before - n;
        if (sh.lazy) std.log.info("KVCODEC rank={d} trimmed keep={d} pages {d} -> {d} free={d}", .{ sh.rank, x.id, before, n, pool.pages.available() });
    }
};

fn identity(b: codec.Block) u64 {
    return b.hash ^ (@as(u64, b.raw) *% 0x9E37_79B9_7F4A_7C15);
}

/// Count `blocks` among the held states; the raw bytes of those another state already held. A full map only stops
/// the count: it is a measurement, and a compaction never fails for it.
fn remember(sh: *Shared, blocks: []const codec.Block) u64 {
    var dup: u64 = 0;
    for (blocks) |b| {
        const slot = sh.held.getOrPut(sh.gpa, identity(b)) catch return dup;
        if (slot.found_existing) {
            dup += b.raw;
            slot.value_ptr.* += 1;
        } else slot.value_ptr.* = 1;
    }
    return dup;
}

fn forget(sh: *Shared, blocks: []const codec.Block) void {
    for (blocks) |b| if (sh.held.getPtr(identity(b))) |n| {
        n.* -= 1;
        if (n.* == 0) _ = sh.held.remove(identity(b));
    };
}

/// With peers and TF_GLM_KV_CODEC_LAZY, at every gate: the leader names one entry it holds compacted (0: none);
/// every rank answers with its copy's compacted pages, or `not_ready` while its copy is raw or part compacted. When
/// all are compacted, every rank keeps the most pages any copy needs, so page use stays equal across the ranks (the
/// leader's free pages, which its store decides on, are every rank's); otherwise the leader asks again later.
pub fn agree(leader: bool, snaps: anytype, pool: *pages.Pool, sh: *Shared, ctx: anytype) !void {
    const id = try word(try ctx.shared(@floatFromInt(if (leader) sh.candidate() else 0)));
    if (id == 0) return;
    const s = snaps.getPtr(id);
    const x: ?*Entry = if (s) |v| v.compact else null;
    const mine: u64 = if (x) |e| (if (sh.compactedPages(e)) |n| n else not_ready) else 0;
    const most = try word(try ctx.shared(@floatFromInt(mine)));
    const e = x orelse return;
    if (most == not_ready) {
        e.tries += 1;
        return;
    }
    e.agreed = true;
    sh.trim(pool, &s.?.lease, e, @intCast(most));
}

pub const not_ready: u64 = std.math.maxInt(u32);

fn word(v: f64) !u64 {
    if (!std.math.isFinite(v) or v < 0 or v > @as(f64, std.math.maxInt(u32)) or @floor(v) != v) return error.BadSnapshotControl;
    return @intFromFloat(v);
}
