//! The kept-state codec's persistent workers: a parallel for on threads made once (no spawn per wave), optionally
//! pinned to a CPU list (TF_GLM_KV_CODEC_CPUS, `taskset -c` syntax) that names only CPUs the process may run on.
const std = @import("std");
const builtin = @import("builtin");
const c = std.c;
const linux = std.os.linux;

pub const max_cpus = 1024;
const word_bits = @bitSizeOf(usize);

/// A set of CPU numbers, as `taskset -c` writes it ("0-4,10-14").
pub const Cpus = struct {
    words: [max_cpus / word_bits]usize = @splat(0),

    pub fn has(s: *const Cpus, cpu: usize) bool {
        return cpu < max_cpus and s.words[cpu / word_bits] >> @intCast(cpu % word_bits) & 1 != 0;
    }

    pub fn add(s: *Cpus, cpu: usize) void {
        s.words[cpu / word_bits] |= @as(usize, 1) << @intCast(cpu % word_bits);
    }

    pub fn count(s: *const Cpus) usize {
        var n: usize = 0;
        for (s.words) |w| n += @popCount(w);
        return n;
    }

    /// The first CPU of `s` that `other` lacks, or null when `other` holds all of `s`.
    pub fn outside(s: *const Cpus, other: *const Cpus) ?usize {
        for (0..max_cpus) |cpu| if (s.has(cpu) and !other.has(cpu)) return cpu;
        return null;
    }

    /// The set as ranges, the form `parse` reads.
    pub fn format(s: *const Cpus, w: *std.Io.Writer) std.Io.Writer.Error!void {
        var cpu: usize = 0;
        var first = true;
        while (cpu < max_cpus) : (cpu += 1) {
            if (!s.has(cpu)) continue;
            var end = cpu;
            while (end + 1 < max_cpus and s.has(end + 1)) end += 1;
            if (!first) try w.writeByte(',');
            first = false;
            if (end == cpu) try w.print("{d}", .{cpu}) else try w.print("{d}-{d}", .{ cpu, end });
            cpu = end;
        }
    }
};

pub const ParseError = error{ CpuListEmpty, CpuListSyntax, CpuListReversed, CpuListTooLarge, CpuListRepeated };

/// "0-4,10-14": single CPUs and inclusive ranges, comma separated, no spaces, each CPU once. On a refusal `bad` is
/// the offending item.
pub fn parse(text: []const u8, bad: *[]const u8) ParseError!Cpus {
    var s: Cpus = .{};
    bad.* = text;
    if (text.len == 0) return error.CpuListEmpty;
    var items = std.mem.splitScalar(u8, text, ',');
    while (items.next()) |item| {
        bad.* = item;
        const dash = std.mem.indexOfScalar(u8, item, '-');
        const lo = try number(if (dash) |d| item[0..d] else item);
        const hi = if (dash) |d| try number(item[d + 1 ..]) else lo;
        if (hi < lo) return error.CpuListReversed;
        for (lo..hi + 1) |cpu| {
            if (s.has(cpu)) return error.CpuListRepeated;
            s.add(cpu);
        }
    }
    return s;
}

fn number(text: []const u8) ParseError!usize {
    if (text.len == 0) return error.CpuListSyntax;
    for (text) |ch| if (ch < '0' or ch > '9') return error.CpuListSyntax;
    const n = std.fmt.parseInt(usize, text, 10) catch return error.CpuListTooLarge;
    return if (n >= max_cpus) error.CpuListTooLarge else n;
}

pub const pinnable = builtin.os.tag == .linux;

/// The CPUs the calling thread may run on now (Linux), else null.
pub fn current() ?Cpus {
    if (!pinnable) return null;
    var set: linux.cpu_set_t = @splat(0);
    if (linux.sched_getaffinity(0, @sizeOf(linux.cpu_set_t), &set) != 0) return null;
    return fromSet(&set);
}

fn fromSet(set: *const linux.cpu_set_t) Cpus {
    var s: Cpus = .{};
    const n = @min(s.words.len, set.len);
    @memcpy(s.words[0..n], set[0..n]);
    return s;
}

/// Pin the calling thread to `s` (Linux); the first CPU of `s` the kernel would not give it is refused by number.
pub fn pin(s: *const Cpus, refused: *usize) error{CpuNotAllowed}!void {
    if (!pinnable) return;
    var set: linux.cpu_set_t = @splat(0);
    const n = @min(s.words.len, set.len);
    @memcpy(set[0..n], s.words[0..n]);
    const rc = linux.syscall3(.sched_setaffinity, 0, @sizeOf(linux.cpu_set_t), @intFromPtr(&set));
    if (linux.errno(rc) != .SUCCESS) {
        refused.* = for (0..max_cpus) |cpu| (if (s.has(cpu)) break cpu) else 0;
        return error.CpuNotAllowed; // no CPU of the set is allowed (the cpuset, or none online)
    }
    const got = current() orelse return;
    if (s.outside(&got)) |cpu| { // the kernel dropped a CPU it will not run this process on
        refused.* = cpu;
        return error.CpuNotAllowed;
    }
}

/// Whether every CPU of `s` is one this process may run on, probed on a short-lived thread so the caller's own
/// affinity never changes; the first refused CPU otherwise. A thread may run outside the mask it inherited
/// (`taskset`), but not outside its cpuset or the online CPUs.
pub fn allowed(s: *const Cpus) ?usize {
    if (!pinnable) return null;
    const Probe = struct {
        fn run(set: *const Cpus, out: *?usize) void {
            var cpu: usize = 0;
            pin(set, &cpu) catch {
                out.* = cpu;
            };
        }
    };
    var out: ?usize = null;
    const th = std.Thread.spawn(.{ .stack_size = 1 << 20 }, Probe.run, .{ s, &out }) catch return null;
    th.join();
    return out;
}

/// Threads made once that run a parallel for (`run`) at a time; callers wait for their own.
pub const Pool = struct {
    gpa: std.mem.Allocator,
    sh: *Shared,
    threads: []std.Thread,

    const Call = *const fn (ctx: *anyopaque, i: usize, worker: usize) void;

    const Shared = struct {
        m: c.pthread_mutex_t = .{},
        work: c.pthread_cond_t = .{},
        done: c.pthread_cond_t = .{},
        serial: c.pthread_mutex_t = .{}, // one `run` at a time
        gen: u64 = 0,
        left: usize = 0, // workers still in the current run
        stop: bool = false,
        ctx: *anyopaque = undefined,
        call: Call = undefined,
        count: usize = 0,
        next: std.atomic.Value(usize) = .init(0),
        cpus: ?Cpus = null,
        refused: std.atomic.Value(usize) = .init(0), // a refused CPU + 1 (0: none)
    };

    /// `n` workers (at least one), each pinned to `cpus` when given. `cpus` must already pass `allowed`.
    pub fn init(gpa: std.mem.Allocator, n: usize, cpus: ?Cpus) !Pool {
        const sh = try gpa.create(Shared);
        errdefer gpa.destroy(sh);
        sh.* = .{ .cpus = cpus };
        const threads = try gpa.alloc(std.Thread, @max(1, n));
        errdefer gpa.free(threads);
        var made: usize = 0;
        errdefer {
            stopAll(sh);
            for (threads[0..made]) |t| t.join();
        }
        while (made < threads.len) : (made += 1) threads[made] = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, worker, .{ sh, made });
        return .{ .gpa = gpa, .sh = sh, .threads = threads };
    }

    pub fn deinit(p: *Pool) void {
        stopAll(p.sh);
        for (p.threads) |t| t.join();
        p.gpa.free(p.threads);
        p.gpa.destroy(p.sh);
    }

    pub fn size(p: *const Pool) usize {
        return p.threads.len;
    }

    /// A CPU a worker could not be pinned to (+1; 0: none), for the caller's log.
    pub fn refused(p: *const Pool) usize {
        return p.sh.refused.load(.monotonic);
    }

    fn stopAll(sh: *Shared) void {
        _ = c.pthread_mutex_lock(&sh.m);
        sh.stop = true;
        _ = c.pthread_cond_broadcast(&sh.work);
        _ = c.pthread_mutex_unlock(&sh.m);
    }

    /// `job(ctx, i, worker)` for every i below `count`, on the workers; returns when all are done.
    pub fn run(p: *Pool, count: usize, ctx: anytype, comptime job: fn (@TypeOf(ctx), usize, usize) void) void {
        const C = @TypeOf(ctx);
        const Thunk = struct {
            fn call(raw: *anyopaque, i: usize, w: usize) void {
                job(@as(C, @ptrCast(@alignCast(raw))), i, w);
            }
        };
        if (count == 0) return;
        const sh = p.sh;
        _ = c.pthread_mutex_lock(&sh.serial);
        defer _ = c.pthread_mutex_unlock(&sh.serial);
        _ = c.pthread_mutex_lock(&sh.m);
        sh.ctx = @ptrCast(@constCast(ctx));
        sh.call = Thunk.call;
        sh.count = count;
        sh.next.store(0, .monotonic);
        sh.left = p.threads.len;
        sh.gen +%= 1;
        _ = c.pthread_cond_broadcast(&sh.work);
        while (sh.left > 0) _ = c.pthread_cond_wait(&sh.done, &sh.m);
        _ = c.pthread_mutex_unlock(&sh.m);
    }

    fn worker(sh: *Shared, w: usize) void {
        if (sh.cpus) |*set| {
            var cpu: usize = 0;
            pin(set, &cpu) catch sh.refused.store(cpu + 1, .monotonic);
        }
        var seen: u64 = 0;
        _ = c.pthread_mutex_lock(&sh.m);
        while (true) {
            while (sh.gen == seen and !sh.stop) _ = c.pthread_cond_wait(&sh.work, &sh.m);
            if (sh.stop) break;
            seen = sh.gen;
            const ctx = sh.ctx;
            const call = sh.call;
            const count = sh.count;
            _ = c.pthread_mutex_unlock(&sh.m);
            while (true) {
                const i = sh.next.fetchAdd(1, .monotonic);
                if (i >= count) break;
                call(ctx, i, w);
            }
            _ = c.pthread_mutex_lock(&sh.m);
            sh.left -= 1;
            if (sh.left == 0) _ = c.pthread_cond_signal(&sh.done);
        }
        _ = c.pthread_mutex_unlock(&sh.m);
    }
};
