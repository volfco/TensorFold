//! TF_GLM_KDA_REPLAY=W: a kept state's KDA states and conv history rebuilt at restore from its last W prompt tokens.
//! Approximate by construction: the stored state's share from before the window decays by the product of the gates.
//! TF_GLM_KDA_REPLAY_INIT=fp8: the replay starts from the entry's coarse copy of its head (cuda_kda_coarse.zig).
const std = @import("std");
const cuda = @import("cuda");
const ledger = @import("core").kv_ledger;
const layout = @import("cuda_snapshot_layout.zig");
const Plan = layout.Plan;
pub const coarse = @import("cuda_kda_coarse.zig");
pub const Init = coarse.Init;

/// The most tokens one replay takes: one prompt pass on the prompt buffers.
pub const max_window = @import("cuda_state.zig").prefill_rows;

/// What runs the replay: `ids` as a prompt pass from the bound state's position, the DSA caches read as stored.
pub const Runner = struct { ctx: *anyopaque, run: *const fn (*anyopaque, []const u32) anyerror!void };

/// A kept entry's form: whole, or without its arena head (KDA states, conv history), rebuilt on restore.
pub const Form = enum { full, replay };

/// What a restore does with the KDA states: nothing more, rebuild them, or rebuild into scratch and compare.
pub const Mode = enum(u2) { none = 0, replay = 1, check = 2 };

const Entry = struct { form: Form, ids: []u32, init: Init = .zero };

/// A slot's last committed prompt tokens (at most the window, oldest first) ending before position `end`.
pub const Recent = struct {
    ids: []u32,
    have: usize = 0,
    end: usize = 0,

    pub fn note(r: *Recent, start: usize, ids: []const u32) void {
        if (start != r.end) r.have = 0;
        const cap = r.ids.len;
        if (ids.len >= cap) {
            @memcpy(r.ids, ids[ids.len - cap ..]);
            r.have = cap;
        } else {
            const old = @min(r.have, cap - ids.len);
            std.mem.copyForwards(u32, r.ids[0..old], r.ids[r.have - old .. r.have]);
            @memcpy(r.ids[old..][0..ids.len], ids);
            r.have = old + ids.len;
        }
        r.end = start + ids.len;
    }

    /// The window before `pos` (min(window, pos) tokens), or null when the tokens are not all known.
    pub fn before(r: *const Recent, pos: usize) ?[]const u32 {
        const n = @min(r.ids.len, pos);
        if (pos == 0 or r.end != pos or r.have < n) return null;
        return r.ids[r.have - n .. r.have];
    }
};

/// Per-head comparison of a replayed state against the stored one.
pub const Stat = struct { rel: f64, max_abs: f64, same: bool };

const Acc = struct {
    diff2: f64 = 0,
    ref2: f64 = 0,
    max_abs: f64 = 0,
    same: bool = true,

    fn add(a: *Acc, x: f32, y: f32) void {
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) a.same = false;
        const d = @as(f64, y) - @as(f64, x);
        a.diff2 += d * d;
        a.ref2 += @as(f64, x) * @as(f64, x);
        if (@abs(d) > a.max_abs) a.max_abs = @abs(d);
    }

    fn stat(a: Acc) Stat {
        const rel = if (a.ref2 > 0) @sqrt(a.diff2 / a.ref2) else if (a.diff2 == 0) 0 else std.math.inf(f64);
        return .{ .rel = rel, .max_abs = a.max_abs, .same = a.same };
    }
};

/// One layer's recurrent states [heads][128][128] fp32: relative L2, max abs error and bit identity a head.
pub fn recStats(stored: []const f32, replayed: []const f32, out: []Stat) void {
    const per = stored.len / out.len;
    for (out, 0..) |*o, h| {
        var a: Acc = .{};
        for (stored[h * per ..][0..per], replayed[h * per ..][0..per]) |x, y| a.add(x, y);
        o.* = a.stat();
    }
}

fn bf16(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// One layer's conv history [3][q | k | v of heads x 128] bf16: a head's three 128-channel blocks over the 3 rows.
pub fn convStats(stored: []const u16, replayed: []const u16, out: []Stat) void {
    const heads = out.len;
    const c = stored.len / 3;
    for (out, 0..) |*o, h| {
        var a: Acc = .{};
        for (0..3) |row| for (0..3) |g| {
            const at = row * c + g * heads * 128 + h * 128;
            for (stored[at..][0..128], replayed[at..][0..128]) |x, y| a.add(bf16(x), bf16(y));
        };
        o.* = a.stat();
    }
}

/// The KDA states the knobs replay: entries' windows, slots' recent tokens, and the runner.
pub const Replay = struct {
    gpa: std.mem.Allocator,
    window: usize,
    keep_form: bool,
    check: bool,
    runner: Runner,
    layers: []u16 = &.{}, // the model layer of each KDA layer, for KVREPLAY lines (empty: KDA indices only)
    recent: std.AutoHashMapUnmanaged(u32, Recent) = .empty,
    entries: std.AutoHashMapUnmanaged(u64, Entry) = .empty,
    replays: u64 = 0,
    checks: u64 = 0,
    last: Summary = .{}, // the last check's totals (host tests read them)
    last_coarse: Summary = .{}, // fp8: the last check's totals for the coarse copy alone (no replay)
    init: Init = .zero,
    stage: ?coarse.Stage = null, // fp8: the coarse copy's pinned staging, made at its first use

    pub fn create(gpa: std.mem.Allocator, window: usize, keep_form: bool, check: bool, runner: Runner) !*Replay {
        return createInit(gpa, window, .zero, keep_form, check, runner);
    }

    /// A window of 0 (the coarse copy alone, no replay) only with the fp8 init.
    pub fn createInit(gpa: std.mem.Allocator, window: usize, init: Init, keep_form: bool, check: bool, runner: Runner) !*Replay {
        if ((window == 0 and init == .zero) or window > max_window) return error.KdaReplayWindow;
        const r = try gpa.create(Replay);
        r.* = .{ .gpa = gpa, .window = window, .init = init, .keep_form = keep_form, .check = check, .runner = runner };
        return r;
    }

    pub fn destroy(r: *Replay) void {
        var it = r.recent.valueIterator();
        while (it.next()) |x| r.gpa.free(x.ids);
        r.recent.deinit(r.gpa);
        var en = r.entries.valueIterator();
        while (en.next()) |x| r.gpa.free(x.ids);
        r.entries.deinit(r.gpa);
        r.gpa.free(r.layers);
        if (r.stage) |*g| g.deinit();
        r.gpa.destroy(r);
    }

    fn slot(r: *Replay, s: u32) !*Recent {
        const g = try r.recent.getOrPut(r.gpa, s);
        if (!g.found_existing) {
            g.value_ptr.* = .{ .ids = r.gpa.alloc(u32, r.window) catch |err| {
                r.recent.removeByPtr(g.key_ptr);
                return err;
            } };
        }
        return g.value_ptr;
    }

    /// A prompt pass committed `ids` at `start` on slot `s`.
    pub fn note(r: *Replay, s: u32, start: usize, ids: []const u32) !void {
        (try r.slot(s)).note(start, ids);
    }

    /// Slot `s` began a fresh sequence: none of its tokens are known.
    pub fn forget(r: *Replay, s: u32) void {
        if (r.recent.getPtr(s)) |x| {
            x.have = 0;
            x.end = 0;
        }
    }

    /// The form a kept entry `id` takes, null when it carries no window.
    pub fn form(r: *const Replay, id: u64) ?Form {
        return if (r.entries.get(id)) |x| x.form else null;
    }

    fn stageFor(r: *Replay, e: anytype) !*const coarse.Stage {
        if (r.stage == null) r.stage = try coarse.Stage.open(e.d, try coarse.Shape.of(&e.cur.st));
        return &r.stage.?;
    }

    /// Entries whose kept state is gone (dropped, or a keep that failed after its plan) give their tokens back.
    fn prune(r: *Replay, snaps: anytype) void {
        var it = r.entries.iterator();
        while (it.next()) |x| if (!snaps.contains(x.key_ptr.*)) {
            r.gpa.free(x.value_ptr.ids);
            r.entries.removeByPtr(x.key_ptr);
            it = r.entries.iterator(); // removal invalidates the walk
        };
    }
};

fn replayOf(e: anytype) ?*Replay {
    if (comptime !@hasField(@TypeOf(e.*), "kda_replay")) return null;
    return e.kda_replay;
}

/// `plan` without its arena head (the KDA states and conv history), marked with the replay window.
pub fn strip(plan: Plan, window: usize) !Plan {
    return stripAs(plan, window, null);
}

/// `strip`, with `first` (the fp8 coarse copy, `len` bytes at a staging address) in the head's place as part 0.
pub fn stripAs(plan: Plan, window: usize, first: ?layout.Part) !Plan {
    var out = plan;
    out.count = 0;
    out.len = 0;
    out.replay = std.math.cast(u32, window) orelse return error.SnapshotShape;
    if (first) |x| {
        out.parts[0] = .{ .live = x.live, .len = x.len, .class = .prefix };
        out.count = 1;
        out.len = x.len;
        out.replay_init = .fp8;
    }
    var heads: usize = 0;
    for (plan.parts[0..plan.count]) |p| {
        if (p.class == .prefix) {
            heads += 1;
            continue;
        }
        out.parts[out.count] = p;
        out.count += 1;
        out.len += p.len;
    }
    if (heads != 1 or (window == 0 and first == null)) return error.SnapshotShape;
    return out;
}

/// The arena head's bytes: the plan's first part (from the arena's start).
fn headLen(plan: *const Plan) !usize {
    if (plan.count == 0 or plan.parts[0].class != .prefix or plan.parts[0].span != 0) return error.SnapshotShape;
    return plan.parts[0].len;
}

fn coarsePart(g: *const coarse.Stage) layout.Part {
    return .{ .live = g.dev, .len = g.shape.len() };
}

/// keepAs: records the entry's window and, with TF_GLM_KDA_REPLAY_KEEP=1, keeps it without its arena head (fp8: with
/// its coarse copy in the head's place).
pub fn keepPlan(e: anytype, id: u64, plan: Plan) !Plan {
    const r = replayOf(e) orelse return plan;
    r.prune(&e.snaps);
    const none: []const u32 = &.{}; // a window of 0 (fp8, the coarse copy alone) needs no tokens
    const known = if (r.window == 0) none else if (r.recent.get(e.slot)) |x| x.before(e.cur.st.pos) else null;
    const ids = known orelse return plan; // tokens past a decode commit or a fresh slot: kept whole, no window
    const vision = if (comptime @hasField(@TypeOf(e.*), "images")) e.images != null else false;
    const replay = r.keep_form and !vision; // image rows splice by position: a replay could not rebuild them
    const own = try r.gpa.dupe(u32, ids);
    errdefer r.gpa.free(own);
    if (r.entries.fetchRemove(id)) |old| r.gpa.free(old.value.ids);
    try r.entries.put(r.gpa, id, .{ .form = if (replay) .replay else .full, .ids = own, .init = r.init });
    if (!replay) return plan;
    if (r.init == .zero) return strip(plan, own.len);
    const g = try r.stageFor(e);
    try coarse.capture(g, &e.cur.st, e.stream); // the pages copy it from the stage before keepAs's sync
    return stripAs(plan, own.len, coarsePart(g));
}

/// restore: an entry kept without its head restores its attention classes only; without the knob it is refused.
pub fn restorePlan(e: anytype, id: u64, plan: Plan) !Plan {
    const s = e.snaps.get(id) orelse return plan;
    const r = replayOf(e) orelse {
        const head = headLen(&plan) catch return plan;
        if (s.len < plan.len and plan.len - s.len == head) return error.KdaReplayNeedsWindow;
        const c = (coarse.Shape.of(&e.cur.st) catch return plan).len();
        if (s.len < plan.len and c < head and plan.len - s.len == head - c) return error.KdaReplayNeedsCoarseInit;
        return plan;
    };
    const x = r.entries.get(id) orelse return plan;
    if (x.form != .replay) return plan;
    if (x.init != r.init) return error.KdaReplayInitDiffers; // kept under another TF_GLM_KDA_REPLAY_INIT
    if (x.init == .zero) return strip(plan, x.ids.len);
    return stripAs(plan, x.ids.len, coarsePart(try r.stageFor(e))); // the pages land in the stage
}

/// restore, after the attention classes and host fields are back: the replay or the check, on every rank alike.
pub fn afterRestore(e: anytype, id: u64) !void {
    const r = replayOf(e) orelse return;
    const s = e.snaps.get(id) orelse return error.NoKeptState;
    const x = r.entries.get(id);
    const mode: Mode = if (x) |y| (if (y.form == .replay) .replay else if (r.check) .check else .none) else .none;
    if (e.share.world > 1) try agree(e, mode);
    switch (mode) {
        .none => {},
        .replay => {
            const from: ?[]const u8 = if (x.?.init == .fp8) r.stage.?.coarse() else null;
            const ns = try rebuild(e, r, x.?.ids, s.pos, from);
            std.log.info("KVREPLAY rank={d} keep={d} pos={d} window={d} form=replay init={s} ms={d:.2}", .{ e.share.rank, id, s.pos, x.?.ids.len, @tagName(x.?.init), ms(ns) });
        },
        .check => try checkRestore(e, r, id, x.?.ids, s),
    }
    const y = try r.slot(e.slot);
    y.have = 0;
    y.end = s.pos;
    if (x) |z| y.note(s.pos - z.ids.len, z.ids);
}

/// Every rank restores alike: a peer that would replay while another would not is refused by name on all of them.
fn agree(e: anytype, mode: Mode) !void {
    if (comptime !@hasDecl(@TypeOf(e.*), "shared")) return;
    const v: f64 = @floatFromInt(@backingInt(mode));
    const most = try e.shared(v);
    const least = 2 - try e.shared(2 - v);
    if (most != v or least != v) return error.KdaReplayRanksDisagree;
}

/// The bound state's arena head zeroed (fp8: then `from`, the coarse copy, decoded into its conv history and each
/// layer's half 0), then `ids` (the tokens before `at`) run from at - ids.len back to `at` (none: `from` alone).
fn rebuild(e: anytype, r: *Replay, ids: []const u32, at: usize, from: ?[]const u8) !u64 {
    const st = &e.cur.st;
    if ((ids.len == 0 and from == null) or ids.len > at or ids.len > max_window) return error.KdaReplayWindow;
    const full = try Plan.init(st, &e.c, at, null);
    const head = try headLen(&full);
    const d = st.arena.buf.d;
    try d.check(d.api.cuMemsetD8Async(st.arena.buf.ptr, 0, head, e.stream.handle), "cuMemsetD8Async");
    @memset(st.cur, 0);
    if (from) |c| try coarse.apply(&r.stage.?, st, e.stream, c);
    try st.setPos(e.stream, at - ids.len);
    const t0 = now();
    if (ids.len > 0) try r.runner.run(r.runner.ctx, ids);
    try e.stream.synchronize();
    if (st.pos != at or st.pending != 0) return error.KdaReplayPosition;
    r.replays += 1;
    return now() - t0;
}

/// The live KDA states (each layer's current half) and the conv history, on the host.
const States = struct {
    conv: []u16,
    rec: []f32,

    fn read(gpa: std.mem.Allocator, st: anytype) !States {
        const conv = try gpa.alloc(u16, st.kda_layers * 3 * st.conv_c);
        errdefer gpa.free(conv);
        const rec = try gpa.alloc(f32, st.kda_layers * st.rec_layer / 4);
        errdefer gpa.free(rec);
        const base = st.arena.buf.ptr;
        try st.arena.buf.download(st.conv - base, std.mem.sliceAsBytes(conv));
        const per = st.rec_layer / 4;
        for (0..st.kda_layers) |li| try st.arena.buf.download(st.recAt(st.cur[li], li) - base, std.mem.sliceAsBytes(rec[li * per ..][0..per]));
        return .{ .conv = conv, .rec = rec };
    }

    fn deinit(s: States, gpa: std.mem.Allocator) void {
        gpa.free(s.conv);
        gpa.free(s.rec);
    }
};

/// A check's totals over every KDA layer and head.
pub const Summary = struct {
    heads: usize = 0,
    rec_same: usize = 0,
    conv_same: usize = 0,
    rec_rel_max: f64 = 0,
    conv_rel_max: f64 = 0,
    rec_abs_max: f64 = 0,
    conv_abs_max: f64 = 0,
};

/// TF_GLM_KDA_REPLAY_CHECK on a full entry: replay into the live head, compare, then put the stored head back. fp8:
/// first the stored head's coarse copy alone (window 0), then the copy plus the replay of the window.
fn checkRestore(e: anytype, r: *Replay, id: u64, ids: []const u32, s: anytype) !void {
    const st = &e.cur.st;
    const full = try Plan.init(st, &e.c, s.pos, null);
    const head = try headLen(&full);
    const d = st.arena.buf.d;
    var backup = try cuda.DeviceBuffer.alloc(d, head);
    defer backup.free();
    try backup.copyFrom(0, st.arena.buf.ptr, head, e.stream.handle);
    try e.stream.synchronize();
    const stored = try States.read(r.gpa, st);
    defer stored.deinit(r.gpa);
    const dir: ?[]const u8 = if (std.c.getenv("TF_GLM_KV_DUMP")) |v| std.mem.span(v) else null;
    const at: Here = .{ .id = id, .w = ids.len, .pos = s.pos, .dir = dir, .full = &full, .head = head };
    if (dir) |x| try dump(e, r.gpa, x, at, r.init, "stored");
    const back = struct {
        fn put(e_: anytype, s_: anytype, from: u64, n: usize) !void {
            const t = &e_.cur.st;
            try t.arena.buf.d.check(t.arena.buf.d.api.cuMemcpyDtoDAsync_v2(t.arena.buf.ptr, from, n, e_.stream.handle), "cuMemcpyDtoDAsync");
            try t.restoreHost(e_.stream, s_.pos, s_.mtp_len, s_.cur, s_.bases);
            try e_.stream.synchronize();
        }
    };
    measureAll(e, r, ids, at, stored) catch |err| {
        cuda.completion.drain(d, e.stream, "a failed KDA replay check did not drain");
        back.put(e, s, backup.ptr, head) catch {};
        return err;
    };
    try back.put(e, s, backup.ptr, head);
    r.checks += 1;
}

/// A check's keep: its id, window, position, dump directory and full plan with its head's bytes.
const Here = struct { id: u64, w: usize, pos: usize, dir: ?[]const u8, full: *const Plan, head: usize };

fn measureAll(e: anytype, r: *Replay, ids: []const u32, at: Here, stored: States) !void {
    if (r.init == .zero) {
        r.last = try measure(e, r, ids, at, stored, null, "replayed");
        return;
    }
    const g = try r.stageFor(e);
    coarse.encodeStates(g.shape, g.coarse(), stored.rec, stored.conv);
    r.last_coarse = try measure(e, r, &.{}, at, stored, g.coarse(), "coarse");
    r.last = if (ids.len > 0) try measure(e, r, ids, at, stored, g.coarse(), "replayed") else r.last_coarse;
}

/// One rebuild of the live head (from zero or `from`, then `ids`), read back, dumped as `tag` and reported.
fn measure(e: anytype, r: *Replay, ids: []const u32, at: Here, stored: States, from: ?[]const u8, tag: []const u8) !Summary {
    const ns = try rebuild(e, r, ids, at.pos, from);
    const got = try States.read(r.gpa, &e.cur.st);
    defer got.deinit(r.gpa);
    if (at.dir) |x| try dump(e, r.gpa, x, at, r.init, tag);
    return report(r, e.share.rank, at.id, at.pos, ids.len, r.init, e.cur.st.heads, stored, got, ns);
}

/// KVREPLAY lines: one a KDA layer and head, one a layer, one in all (rec: the recurrent state, conv: its history).
fn report(r: *Replay, rank: usize, id: u64, pos: usize, w: usize, init: Init, heads: usize, a: States, b: States, ns: u64) !Summary {
    const layers = a.rec.len / (heads * 128 * 128);
    const rec = try r.gpa.alloc(Stat, heads);
    defer r.gpa.free(rec);
    const conv = try r.gpa.alloc(Stat, heads);
    defer r.gpa.free(conv);
    const rels = try r.gpa.alloc(f64, heads);
    defer r.gpa.free(rels);
    const all = try r.gpa.alloc(f64, layers * heads);
    defer r.gpa.free(all);
    var sum: Summary = .{};
    const per = heads * 128 * 128;
    const cper = a.conv.len / @max(layers, 1);
    for (0..layers) |li| {
        recStats(a.rec[li * per ..][0..per], b.rec[li * per ..][0..per], rec);
        convStats(a.conv[li * cper ..][0..cper], b.conv[li * cper ..][0..cper], conv);
        const layer: isize = if (li < r.layers.len) r.layers[li] else -1;
        var same: usize = 0;
        var csame: usize = 0;
        var abs: f64 = 0;
        var cabs: f64 = 0;
        var crel: f64 = 0;
        for (rec, conv, 0..) |x, c, h| {
            std.log.info("KVREPLAY head rank={d} keep={d} window={d} init={s} kda={d} layer={d} head={d} rec_rel={e:.3} rec_maxabs={e:.3} rec_same={d} conv_rel={e:.3} conv_maxabs={e:.3} conv_same={d}", .{ rank, id, w, @tagName(init), li, layer, h, x.rel, x.max_abs, @intFromBool(x.same), c.rel, c.max_abs, @intFromBool(c.same) });
            rels[h] = x.rel;
            all[li * heads + h] = x.rel;
            same += @intFromBool(x.same);
            csame += @intFromBool(c.same);
            abs = @max(abs, x.max_abs);
            cabs = @max(cabs, c.max_abs);
            crel = @max(crel, c.rel);
        }
        std.mem.sort(f64, rels, {}, std.sort.asc(f64));
        std.log.info("KVREPLAY layer rank={d} keep={d} pos={d} window={d} init={s} kda={d} layer={d} heads={d} rec_same={d} rec_rel_p50={e:.3} rec_rel_p90={e:.3} rec_rel_max={e:.3} rec_maxabs={e:.3} conv_same={d} conv_rel_max={e:.3} conv_maxabs={e:.3}", .{ rank, id, pos, w, @tagName(init), li, layer, heads, same, quantile(rels, 0.5), quantile(rels, 0.9), rels[heads - 1], abs, csame, crel, cabs });
        sum.heads += heads;
        sum.rec_same += same;
        sum.conv_same += csame;
        sum.rec_rel_max = @max(sum.rec_rel_max, rels[heads - 1]);
        sum.conv_rel_max = @max(sum.conv_rel_max, crel);
        sum.rec_abs_max = @max(sum.rec_abs_max, abs);
        sum.conv_abs_max = @max(sum.conv_abs_max, cabs);
    }
    std.mem.sort(f64, all, {}, std.sort.asc(f64));
    std.log.info("KVREPLAY total rank={d} keep={d} pos={d} window={d} form=check init={s} heads={d} rec_same={d} rec_rel_p50={e:.3} rec_rel_p90={e:.3} rec_rel_max={e:.3} rec_maxabs={e:.3} conv_same={d} conv_rel_max={e:.3} conv_maxabs={e:.3} ms={d:.2}", .{ rank, id, pos, w, @tagName(init), sum.heads, sum.rec_same, quantile(all, 0.5), quantile(all, 0.9), sum.rec_rel_max, sum.rec_abs_max, sum.conv_same, sum.conv_rel_max, sum.conv_abs_max, ms(ns) });
    return sum;
}

fn quantile(sorted: []const f64, q: f64) f64 {
    if (sorted.len == 0) return 0;
    const at: usize = @intFromFloat(@floor(q * @as(f64, @floatFromInt(sorted.len - 1)) + 0.5));
    return sorted[at];
}

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: [*]const u8, size: usize, n: usize, f: *anyopaque) usize;
extern "c" fn fclose(f: *anyopaque) c_int;

/// TF_GLM_KV_DUMP: the live arena head as a head-only keep kvdump.py reads, kda-replay-rank<r>-keep<id>-w<W>-<tag>
/// (zero init: stored, replayed; fp8: fp8-stored, fp8-coarse, fp8-replayed, all three named with the check's W).
fn dump(e: anytype, gpa: std.mem.Allocator, dir: []const u8, at: Here, init: Init, tag: []const u8) !void {
    const id, const w, const full, const head = .{ at.id, at.w, at.full, at.head };
    const st = &e.cur.st;
    const bytes = try gpa.alloc(u8, head);
    defer gpa.free(bytes);
    try st.arena.buf.download(0, bytes);
    var one = full.*;
    one.count = 1;
    one.len = head;
    var names: [3][512]u8 = undefined;
    var paths: [3][:0]const u8 = undefined;
    for (&paths, &names, [_][]const u8{ "bin", "parts", "plan" }) |*p, *buf, ext|
        p.* = try std.fmt.bufPrintSentinel(buf, "{s}/kda-replay-rank{d}-keep{d}-w{d}-{s}{s}.{s}", .{ dir, e.share.rank, id, w, if (init == .fp8) "fp8-" else "", tag, ext }, 0);
    var out = try ledger.Dump.open(paths[0], paths[1]);
    out.span(.prefix, 0, bytes) catch |err| {
        out.close() catch {};
        return err;
    };
    try out.close();
    var text: std.Io.Writer.Allocating = .init(gpa);
    defer text.deinit();
    try layout.manifest(&one, st, &e.c, e.share.rank, e.share.world, &text.writer);
    const f = fopen(paths[2].ptr, "w") orelse return error.DumpOpen;
    const written = text.written();
    const n = fwrite(written.ptr, 1, written.len, f);
    if (fclose(f) != 0 or n != written.len) return error.DumpWrite;
}

const Timespec = extern struct { sec: isize, nsec: isize };
extern "c" fn clock_gettime(clock: c_int, ts: *Timespec) c_int;

fn now() u64 {
    var ts: Timespec = undefined;
    _ = clock_gettime(1, &ts); // CLOCK_MONOTONIC
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// The knobs (cuda_ranks.Settings.read): true when one is refused, `problem` naming it.
pub fn readSettings(get: *const fn ([:0]const u8) ?[]const u8, s: anytype, problem: *[]const u8) !bool {
    if (get("TF_GLM_KDA_REPLAY")) |v| {
        s.kda_replay = std.fmt.parseInt(u32, v, 10) catch max_window + 1;
        if (s.kda_replay > max_window) {
            problem.* = "TF_GLM_KDA_REPLAY: 0 (off) to 2048 tokens";
            return true;
        }
    }
    if (get("TF_GLM_KDA_REPLAY_INIT")) |v| s.kda_replay_init = std.meta.stringToEnum(Init, v) orelse {
        problem.* = "TF_GLM_KDA_REPLAY_INIT: zero or fp8";
        return true;
    };
    inline for (.{ .{ "TF_GLM_KDA_REPLAY_KEEP", "kda_replay_keep" }, .{ "TF_GLM_KDA_REPLAY_CHECK", "kda_replay_check" } }) |x| if (get(x[0])) |v| {
        if (std.mem.eql(u8, v, "1")) @field(s, x[1]) = true else if (!std.mem.eql(u8, v, "0")) {
            problem.* = x[0] ++ ": 0 or 1";
            return true;
        }
        if (@field(s, x[1]) and s.kda_replay == 0 and s.kda_replay_init == .zero) {
            problem.* = x[0] ++ "=1 requires TF_GLM_KDA_REPLAY (the window in tokens)";
            return true;
        }
    };
    if (s.kda_replay_init != .zero and !s.kda_replay_keep and !s.kda_replay_check) {
        problem.* = "TF_GLM_KDA_REPLAY_INIT=fp8 requires TF_GLM_KDA_REPLAY_KEEP=1 or TF_GLM_KDA_REPLAY_CHECK=1";
        return true;
    }
    return false;
}

/// The engine's replay object (null with TF_GLM_KDA_REPLAY off and no fp8 init), its runner the engine's own prompt pass.
pub fn open(e: anytype) !?*Replay {
    const s = e.settings;
    if (s.kda_replay == 0 and s.kda_replay_init == .zero) return null;
    const r = try Replay.createInit(e.gpa, s.kda_replay, s.kda_replay_init, s.kda_replay_keep, s.kda_replay_check, .{ .ctx = e, .run = enginePass });
    errdefer r.destroy();
    var n: usize = 0;
    for (e.w.layers) |l| n += @intFromBool(l.mixer == .kda);
    r.layers = try e.gpa.alloc(u16, n);
    n = 0;
    for (e.w.layers, 0..) |l, j| if (l.mixer == .kda) {
        r.layers[n] = @intCast(j);
        n += 1;
    };
    return r;
}

/// The replay pass: a prompt pass of `ids` from the state's position, its DSA rows read as stored, never written.
fn enginePass(ptr: *anyopaque, ids: []const u32) anyerror!void {
    const fw = @import("cuda_forward.zig");
    const e: *@import("cuda_engine.zig").Engine = @ptrCast(@alignCast(ptr));
    var one: [1]fw.Rank = .{.{ .w = &e.w, .st = &e.cur.st, .b = &e.bp }};
    const f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .pieces = &e.pieces, .split = e.settings.split, .ring = e.settings.ring, .kda_wide = e.settings.kda_wide, .fixed_rows = true };
    try f.run(@ptrCast(ids));
    try f.commit(ids.len);
}

test {
    _ = @import("cuda_kda_replay_test.zig");
    _ = @import("cuda_kda_coarse.zig");
}
