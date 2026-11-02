//! TF_GLM_KV_CODEC=1 (with TF_GLM_HOST_CACHE=1): a kept state is compacted in place in its host pages, losslessly,
//! and the pages past it go back; a restore decodes through pinned staging into the same device copies.
//! TF_GLM_KV_CODEC_LAZY=1: the keep stores the raw state and returns; the compaction runs later, off the prompt pass
//! (cuda_snapshot_lazy.zig). The workers are made once, optionally pinned (TF_GLM_KV_CODEC_CPUS).
//! TF_GLM_KV_LEDGER=1 logs each keep's byte classes (Phase 0); TF_GLM_KV_DUMP=dir writes each keep's raw stream.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const codec = core.kv_codec;
const workers = codec.worker_pool;
const ledger = core.kv_ledger;
const pages = @import("cuda_snapshot_pages.zig");
const layout = @import("cuda_snapshot_layout.zig");
const lazy = @import("cuda_snapshot_lazy.zig");
const Plan = layout.Plan;

const page = pages.page;
const block = codec.block_size;
pub const now = lazy.now;
const View = lazy.View;
pub const Entry = lazy.Entry;
pub const Compact = *Entry; // a kept state's codec side, in any form (cuda_engine_types.Snap.compact)

/// With peers and TF_GLM_KV_CODEC_LAZY, at every gate: one compacted entry's pages agreed (cuda_snapshot_lazy.agree).
pub fn agreeLazy(leader: bool, snaps: anytype, pool: *pages.Pool, c: *Codec, ctx: anytype) !void {
    return lazy.agree(leader, snaps, pool, c.sh, ctx);
}

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: [*]const u8, size: usize, n: usize, f: *anyopaque) usize;
extern "c" fn fclose(f: *anyopaque) c_int;

pub fn enabled() bool {
    return flag("TF_GLM_KV_CODEC");
}

/// TF_GLM_KV_CODEC_LAZY=1: compaction after the keep returns, off the prompt pass (unset or 0: inside the keep).
pub fn lazyEnabled() bool {
    return flag("TF_GLM_KV_CODEC_LAZY");
}

fn flag(name: [:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn env(name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

/// TF_GLM_KV_CODEC_CANDIDATES: plain (default: byte planes only), atlas (the Atlas column codecs by model) or
/// atlas-exhaustive (smallest of all candidates, the cost of each); anything else keeps plain.
fn optionsFromEnv() codec.Options {
    const v = env("TF_GLM_KV_CODEC_CANDIDATES") orelse return .plain;
    return codec.Options.parse(v) orelse .plain;
}

/// TF_GLM_KV_CODEC_THREADS (1 to 64), else up to 8 of the CPUs the workers may use.
fn threadCount(cpus: ?workers.Cpus) usize {
    if (env("TF_GLM_KV_CODEC_THREADS")) |v| {
        const n = std.fmt.parseInt(usize, v, 10) catch 0;
        if (n >= 1 and n <= 64) return n;
    }
    return @min(8, if (cpus) |s| s.count() else std.Thread.getCpuCount() catch 1);
}

/// TF_GLM_KV_CODEC_LAZY_MS: how long a kept or resumed state stays raw before the compactor takes it (default 2000).
pub fn delayMs() !u64 {
    const v = env("TF_GLM_KV_CODEC_LAZY_MS") orelse return 2000;
    const n = std.fmt.parseInt(u64, v, 10) catch return refuse("TF_GLM_KV_CODEC_LAZY_MS={s}: milliseconds, 0 to 3600000", .{v}, error.KvCodecLazyMs);
    return if (n > 3_600_000) refuse("TF_GLM_KV_CODEC_LAZY_MS={s}: milliseconds, 0 to 3600000", .{v}, error.KvCodecLazyMs) else n;
}

/// TF_GLM_KV_CODEC_CPUS ("0-4,10-14"): the CPUs the workers and the compactor are pinned to; every one must be a
/// CPU this process may run on. Unset: unpinned. Pinning is Linux only; elsewhere the list is checked and noted.
pub fn cpusFromEnv() !?workers.Cpus {
    const v = env("TF_GLM_KV_CODEC_CPUS") orelse return null;
    var bad: []const u8 = undefined;
    const s = workers.parse(v, &bad) catch |err| return refuse("TF_GLM_KV_CODEC_CPUS={s}: {s} at \"{s}\" (a taskset -c list such as 0-4,10-14)", .{ v, @errorName(err), bad }, error.KvCodecCpus);
    if (!workers.pinnable) {
        std.log.info("KVCODEC: TF_GLM_KV_CODEC_CPUS={s} is ignored on this system (pinning needs Linux)", .{v});
        return null;
    }
    if (workers.allowed(&s)) |cpu| return refuse("TF_GLM_KV_CODEC_CPUS={s}: CPU {d} is not one this process may run on (offline, absent or outside its cpuset)", .{ v, cpu }, error.KvCodecCpuNotAllowed);
    return s;
}

/// A refused setting, named in the log (warn: the error itself stops the start) and returned.
fn refuse(comptime fmt: []const u8, args: anytype, err: anyerror) anyerror {
    std.log.warn("KVCODEC refused: " ++ fmt, args);
    return err;
}

pub const Codec = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    rank: usize,
    threads: usize,
    lazy: bool,
    sh: *lazy.Shared,
    staging: cuda.HostBuffer, // two waves of decoded blocks, mapped
    staging_dev: u64,
    ledger_on: bool,
    seen: ledger.Ledger = .{},
    dump: ?[]const u8,

    /// At prepare time, beside the snapshot pages: no keep or restore allocates device or pinned memory or a thread.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, rank: usize) !Codec {
        const cpus = try cpusFromEnv();
        const delay = try delayMs();
        const threads = threadCount(cpus);
        const count = @max(8, threads);
        const scratch = try gpa.alloc(codec.Scratch, count);
        var made: usize = 0;
        errdefer {
            for (scratch[0..made]) |*s| s.deinit(gpa);
            gpa.free(scratch);
        }
        while (made < count) : (made += 1) scratch[made] = try codec.Scratch.init(gpa);
        const payloads = try gpa.alloc([]const u8, count);
        errdefer gpa.free(payloads);
        var pool = try workers.Pool.init(gpa, threads, cpus);
        errdefer pool.deinit();
        const sh = try gpa.create(lazy.Shared);
        errdefer gpa.destroy(sh);
        sh.* = .{ .gpa = gpa, .rank = rank, .pool = pool, .scratch = scratch, .payloads = payloads, .options = optionsFromEnv(), .lazy = lazyEnabled(), .delay_ns = delay * std.time.ns_per_ms, .cpus = cpus, .wave_blocks = count };
        var staging = try cuda.HostBuffer.allocMapped(d, 2 * count * block);
        errdefer staging.free();
        try sh.start();
        logStart(sh, rank, threads, delay);
        return .{ .gpa = gpa, .d = d, .rank = rank, .threads = threads, .lazy = sh.lazy, .sh = sh, .staging = staging, .staging_dev = try staging.device(), .ledger_on = flag("TF_GLM_KV_LEDGER"), .dump = env("TF_GLM_KV_DUMP") };
    }

    pub fn deinit(c: *Codec) void {
        c.log();
        c.sh.deinit();
        c.gpa.destroy(c.sh);
        c.staging.free();
    }

    fn wave(c: *const Codec) usize {
        return c.sh.scratch.len;
    }

    /// After `move` copied the raw state into `lease` and the stream synchronized: the state's codec entry, compacted
    /// in place now (eager) or queued raw (lazy). The lease keeps every page; `settle` or the gate trims it. Fails only
    /// before any byte moves.
    pub fn keep(c: *Codec, pool: *const pages.Pool, lease: *const pages.Lease, plan: *const Plan, id: u64) !*Entry {
        const base = pool.hostBytes() orelse return error.KvCodecNeedsHostCache;
        var spans: [@typeInfo(@FieldType(Plan, "parts")).array.len]codec.Span = undefined;
        for (plan.parts[0..plan.count], spans[0..plan.count]) |part, *s| s.* = .{ .len = part.len, .class = @backingInt(part.class), .row = part.row };
        const n = codec.cutCount(spans[0..plan.count]);
        const x = try c.gpa.create(Entry);
        errdefer c.gpa.destroy(x);
        const cuts = try c.gpa.alloc(codec.Cut, n);
        errdefer c.gpa.free(cuts);
        codec.cut(spans[0..plan.count], cuts);
        const blocks = try c.gpa.alloc(codec.Block, n);
        errdefer c.gpa.free(blocks);
        const ids = try c.gpa.dupe(u32, lease.ids.items);
        errdefer c.gpa.free(ids);
        const view: View = .{ .base = base, .ids = ids };
        if (c.ledger_on or c.dump != null) try c.observe(view, cuts, id, plan.len);
        const t = now();
        x.* = .{ .id = id, .len = plan.len, .base = base, .ids = ids, .cuts = cuts, .blocks = blocks, .kept = t, .touched = t };
        try c.sh.add(x);
        if (!c.lazy) c.sh.compactNow(x); // encode_ns the workers' waves; the write-back copy_ns, a wait wait_ns
        return x;
    }

    /// Alone (no peers to agree with): a compacted entry's pages past its payloads go back now.
    pub fn settle(c: *Codec, pool: *pages.Pool, lease: *pages.Lease, x: *Entry) void {
        if (c.sh.compactedPages(x)) |n| c.sh.trim(pool, lease, x, n);
    }

    /// The pages the store was last told of: nothing set aside for `x` any more.
    pub fn claim(c: *Codec, x: *Entry) void {
        c.sh.claim(x);
    }

    /// Pages given back that no store charge has seen yet (held out of the spare count until one does).
    pub fn unclaimed(c: *Codec) usize {
        c.sh.lock();
        defer c.sh.unlock();
        return c.sh.unclaimed;
    }

    /// The state's codec entry goes (its state was dropped); call before its pages go back.
    pub fn drop(c: *Codec, x: *Entry) void {
        c.sh.remove(x);
        c.gpa.free(x.ids);
        c.gpa.free(x.blocks);
        c.gpa.free(x.cuts);
        c.gpa.destroy(x);
    }

    /// The kept bytes into the live state, whatever form `x` is in: its compacted blocks decode through the staging
    /// waves (a wave decodes while the last one copies), its raw rest copies straight from the pages.
    pub fn restore(c: *Codec, pool: *const pages.Pool, stream: cuda.Stream, lease: *const pages.Lease, x: *Entry, plan: *const Plan) !void {
        const base = pool.hostBytes() orelse return error.KvCodecNeedsHostCache;
        const t_pin = now();
        c.sh.pin(x); // waits for a compaction of `x` under way to stop at its wave's end
        const pinned = now();
        defer c.sh.unpin(x);
        if (x.len != plan.len) return error.SnapshotSize;
        if ((x.done > 0 and x.pages() > lease.ids.items.len) or (!x.full() and lease.ids.items.len * page < x.len)) return error.SnapshotSize;
        const view: View = .{ .base = base, .ids = lease.ids.items };
        var tm: Split = .{ .wait = pinned - t_pin };
        if (x.done > 0) try c.decode(stream, view, x.blocks[0..x.done], plan, &tm);
        const t0 = now();
        if (!x.full()) try pool.moveFrom(stream, lease, plan, false, x.rawFrom());
        try stream.synchronize(); // the last wave's copies and the raw rest have landed: copy_ms ends here
        tm.copy += now() - t0;
        c.sh.lock();
        defer c.sh.unlock();
        const t = &c.sh.totals;
        t.restores += 1;
        if (x.done == 0) t.raw_restores += 1 else if (!x.full()) t.partial_restores += 1;
        t.decode_ns += tm.decode;
        t.copy_ns += tm.copy;
        t.wait_ns += tm.wait;
    }

    /// A restore's time on the engine thread: the workers' decode waves, this thread's copies, its waits.
    const Split = struct { decode: u64 = 0, copy: u64 = 0, wait: u64 = 0 };

    fn decode(c: *Codec, stream: cuda.Stream, view: View, blocks: []const codec.Block, plan: *const Plan, tm: *Split) !void {
        tm.wait += c.sh.hold();
        defer _ = std.c.pthread_mutex_unlock(&c.sh.work);
        const Job = struct {
            c: *Codec,
            blocks: []const codec.Block,
            inputs: []const []const u8,
            slot: usize,
            failed: std.atomic.Value(bool) = .init(false),
            fn run(j: *@This(), i: usize, _: usize) void {
                const b = j.blocks[i];
                const into = j.c.staging.bytes[j.slot + i * block ..][0..b.raw];
                codec.decodeBlock(&j.c.sh.scratch[i], b, j.inputs[i], into) catch j.failed.store(true, .monotonic);
            }
        };
        const inputs = c.sh.payloads; // as payload slices: the encoder does not run while `work` is held
        var at: usize = 0; // payload offset in the pages
        var raw: usize = 0; // raw offset in the plan
        var w: usize = 0;
        var round: usize = 0;
        while (w < blocks.len) : ({
            w += c.wave();
            round += 1;
        }) {
            const n = @min(c.wave(), blocks.len - w);
            const items = blocks[w..][0..n];
            const t0 = now();
            if (round >= 2) try stream.synchronize(); // this slot's last copies have landed
            for (items, inputs[0..n], 0..) |b, *in, i| {
                in.* = view.read(at, b.len, c.sh.scratch[i].input);
                at += b.len;
            }
            const slot = (round % 2) * c.wave() * block;
            var job: Job = .{ .c = c, .blocks = items, .inputs = inputs[0..n], .slot = slot };
            const t1 = now();
            c.sh.pool.run(n, &job, Job.run);
            const t2 = now();
            if (job.failed.load(.monotonic)) return error.KvCodecCorrupt;
            for (items, 0..) |b, i| {
                try copyRange(c.d, stream, plan, raw, c.staging_dev + slot + i * block, b.raw);
                raw += b.raw;
            }
            tm.decode += t2 - t1;
            tm.copy += (t1 - t0) + (now() - t2);
        }
    }

    /// Phase 0's ledger for one keep: its classes must cover the plan's bytes exactly (delta 0), or the keep fails.
    /// Timed as keep_io_ns, its wait for the compactor's wave (`work`) as wait_ns.
    fn observe(c: *Codec, view: View, cuts: []const codec.Cut, id: u64, len: usize) !void {
        const t0 = now();
        var wait: u64 = 0;
        defer {
            c.sh.count("wait_ns", wait);
            c.sh.count("keep_io_ns", now() - t0 - wait);
        }
        var one: ledger.Ledger = .{};
        var dump: ?ledger.Dump = null;
        if (c.dump) |dir| {
            var a: [512]u8 = undefined;
            var b: [512]u8 = undefined;
            const data = try std.fmt.bufPrintSentinel(&a, "{s}/glm-rank{d}-keep{d}.bin", .{ dir, c.rank, id }, 0);
            const parts = try std.fmt.bufPrintSentinel(&b, "{s}/glm-rank{d}-keep{d}.parts", .{ dir, c.rank, id }, 0);
            dump = try ledger.Dump.open(data, parts);
        }
        errdefer if (dump) |*d| d.close() catch {};
        wait = c.sh.hold(); // scratch[0] is the encoder's too
        defer _ = std.c.pthread_mutex_unlock(&c.sh.work);
        for (cuts) |x| {
            const bytes = view.read(x.off, x.len, c.sh.scratch[0].input);
            one.observe(@fromBackingInt(@intCast(x.class)), bytes);
            if (dump) |*d| try d.span(@fromBackingInt(@intCast(x.class)), x.row, bytes);
        }
        if (dump) |*d| try d.close();
        if (one.total() != len) return error.LedgerMismatch;
        if (!c.ledger_on) return;
        c.seen.add(&one);
        var buf: [4096]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        var tag: [48]u8 = undefined;
        one.print(&w, try std.fmt.bufPrint(&tag, "rank={d} keep={d}", .{ c.rank, id })) catch {};
        c.seen.print(&w, try std.fmt.bufPrint(&tag, "rank={d} all", .{c.rank})) catch {};
        std.log.info("{s}", .{w.buffered()});
    }

    /// TF_GLM_KV_DUMP: the keep's manifest (layout.manifest) beside its bytes and cuts; nothing without a dump.
    /// Timed as keep_io_ns.
    pub fn writePlan(c: *const Codec, id: u64, plan: *const Plan, st: anytype, cfg: anytype, world: usize) !void {
        const dir = c.dump orelse return;
        const t0 = now();
        defer c.sh.count("keep_io_ns", now() - t0);
        var text: std.Io.Writer.Allocating = .init(c.gpa);
        defer text.deinit();
        try layout.manifest(plan, st, cfg, c.rank, world, &text.writer);
        var a: [512]u8 = undefined;
        const path = try std.fmt.bufPrintSentinel(&a, "{s}/glm-rank{d}-keep{d}.plan", .{ dir, c.rank, id }, 0);
        const f = fopen(path.ptr, "w") orelse return error.DumpOpen;
        const bytes = text.written();
        const n = fwrite(bytes.ptr, 1, bytes.len, f);
        if (fclose(f) != 0 or n != bytes.len) return error.DumpWrite;
    }

    /// The totals line once a state was kept. The timers (Totals; "Timers" in tools/kvprobe/STATUS-lazy.md):
    /// encode_ms the workers' encode inside keeps (eager, on the prompt pass), lazy_encode_ms on the compactor,
    /// decode_ms the workers' decode in restores; then keep_io_ms (ledger and dump), copy_ms (the engine thread's
    /// page copies and staging), lazy_copy_ms (the compactor's write-back), wait_ms (the engine thread's waits).
    pub fn log(c: *const Codec) void {
        var buf: [640]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        if (c.totalsLine(&w)) std.log.info("{s}", .{w.buffered()});
    }

    /// The totals line `log` prints; false (nothing written) before any keep.
    pub fn totalsLine(c: *const Codec, w: *std.Io.Writer) bool {
        c.sh.lock();
        const t = c.sh.totals;
        c.sh.unlock();
        if (t.keeps == 0) return false;
        const ratio = if (t.raw == 0) 0 else @as(f64, @floatFromInt(t.stored)) / @as(f64, @floatFromInt(t.raw));
        w.print("KVCODEC rank={d} mode={s} keeps={d} compacted={d} restores={d} raw_restores={d} partial_restores={d} raw={d} stored={d} ratio={d:.4} duplicate={d} pages {d} -> {d} encode_ms={d:.1} lazy_encode_ms={d:.1} decode_ms={d:.1} cancels={d} threads={d} keep_io_ms={d:.1} copy_ms={d:.1} lazy_copy_ms={d:.1} wait_ms={d:.1}", .{
            c.rank,          if (c.lazy) "lazy" else "eager", t.keeps,         t.compacted, t.restores, t.raw_restores,   t.partial_restores, t.raw,              t.stored,      ratio, t.duplicate, t.raw_pages, t.kept_pages,
            ms(t.encode_ns), ms(t.lazy_ns),                   ms(t.decode_ns), t.cancels,   c.threads,  ms(t.keep_io_ns), ms(t.copy_ns),      ms(t.lazy_copy_ns), ms(t.wait_ns),
        }) catch return false;
        return true;
    }
};

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn logStart(sh: *const lazy.Shared, rank: usize, threads: usize, delay: u64) void {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    if (sh.cpus) |*s| {
        w.writeAll(" cpus=") catch {};
        s.format(&w) catch {};
        if (workers.current()) |mine| {
            w.writeAll(if (s.outside(&mine) == null) " (inside the engine's " else " (outside the engine's ") catch {};
            mine.format(&w) catch {};
            w.writeAll(")") catch {};
        }
    }
    std.log.info("KVCODEC rank={d} mode={s} delay_ms={d} threads={d}{s}", .{ rank, if (sh.lazy) "lazy" else "eager", delay, threads, w.buffered() });
}

/// Copy `n` staged bytes to the live state at the plan's raw offset `from`, piece by piece as `Pool.move` walks it.
fn copyRange(d: *const cuda.Driver, stream: cuda.Stream, plan: *const Plan, from: usize, src: u64, n: usize) !void {
    var off: usize = 0;
    for (plan.parts[0..plan.count]) |part| {
        const end = off + part.len;
        defer off = end;
        if (end <= from) continue;
        if (off >= from + n) return;
        var done = @max(from, off) - off;
        const stop = @min(from + n, end) - off;
        while (done < stop) {
            const live = plan.piece(part, done);
            const k = @min(live.len, stop - done);
            try d.check(d.api.cuMemcpyDtoDAsync_v2(live.live, src + (off + done - from), k, stream.handle), "cuMemcpyDtoDAsync");
            done += k;
        }
    }
}
