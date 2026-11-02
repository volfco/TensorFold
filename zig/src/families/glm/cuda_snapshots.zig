//! Prompt-sized snapshots on each rank; the existing chunk grid and the live sequence's layout do not change.
const std = @import("std");
const cuda = @import("cuda");
const types = @import("cuda_engine_types.zig");
pub const pages = @import("cuda_snapshot_pages.zig");
pub const layout = @import("cuda_snapshot_layout.zig");
pub const codec = @import("cuda_snapshot_codec.zig");
const kda_replay = @import("cuda_kda_replay.zig");

pub const Setup = struct { ptr: *anyopaque, run: *const fn (*anyopaque, u64) anyerror!void };

pub fn prepare(e: anytype, budget: u64) !void {
    if (e.snap_pages != null or e.snaps.count() != 0) return error.SnapshotCacheMade;
    if (codec.enabled() and !pages.onHost()) return error.KvCodecNeedsHostCache; // compaction runs on the CPU
    if (codec.lazyEnabled() and !codec.enabled()) return error.KvCodecLazyNeedsCodec;
    const count = std.math.cast(u32, budget / pages.page) orelse return error.SnapshotBudget;
    try e.snaps.ensureTotalCapacity(e.gpa, count);
    try e.dropped.ensureTotalCapacity(e.gpa, @as(usize, count) + 1);
    e.snap_pages = try pages.Pool.init(e.gpa, e.d, budget, pages.onHost());
    // whenever enabled, as on every rank (the fingerprint agrees on it): the gate's page agreement keys on it
    if (codec.enabled()) e.snap_codec = try codec.Codec.init(e.gpa, e.d, e.share.rank);
}

/// A failed allocation still joins the startup agreement, so a peer never waits for a missing participant.
pub fn agree(e: anytype, budget: u64) !void {
    var failed: ?anyerror = null;
    e.prepareSnapshots(budget) catch |err| {
        failed = err;
    };
    if (try e.shared(if (failed != null) 1 else 0) > 0) return failed orelse error.PeerSnapshotCache;
}

pub fn bytes(e: anytype, at: usize) u64 {
    const head: ?usize = if (e.mtp) @min(e.capacity, std.math.add(usize, at, @import("cuda_state.zig").window_rows) catch return std.math.maxInt(u64)) else null;
    // a paged sequence holds its request's rows: its prefix is counted to the capacity, in a window's bytes
    const n = layout.Plan.size(&e.cur.st, &e.c, e.capacity, at, head) catch return std.math.maxInt(u64);
    return pages.rounded(n) catch std.math.maxInt(u64);
}

pub fn keep(e: anytype) !u64 {
    if (e.next_snap > std.math.maxInt(u32)) return error.SnapshotIdLimit;
    const id = e.next_snap;
    try keepAs(e, id);
    e.next_snap += 1;
    e.announce = id;
    return id;
}

pub fn keepAs(e: anytype, id: u64) !void {
    if (e.snaps.contains(id)) return error.StateKept;
    if (e.cur.st.pending != 0 or e.cur.st.mtp_len > e.cur.st.pos + @import("cuda_state.zig").window_rows)
        return error.NotAtMark;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    const plan = try kda_replay.keepPlan(e, id, try layout.Plan.init(&e.cur.st, &e.c, e.cur.st.pos, if (e.mtp) e.cur.st.mtp_len else null));
    var lease = try pool.take(plan.len);
    errdefer {
        cuda.completion.drain(e.d, e.stream, "a failed GLM snapshot copy did not drain");
        pool.give(&lease);
    }
    const cur = try e.gpa.dupe(u1, e.cur.st.cur);
    errdefer e.gpa.free(cur);
    try pool.move(e.stream, &lease, &plan, true);
    try e.stream.synchronize();
    if (e.snap_codec) |*c| try c.writePlan(id, &plan, &e.cur.st, &e.c, e.share.world);
    const compact: ?codec.Compact = if (e.snap_codec) |*c| try c.keep(pool, &lease, &plan, id) else null;
    // alone, the pages past a compacted state go back now (lazy: when compacted); with peers, once `gate` agrees
    if (compact) |x| if (e.share.world == 1) {
        e.snap_codec.?.settle(pool, &lease, x);
        e.snap_codec.?.claim(x); // before the store's charge, which reads the lease
    };
    e.snaps.putAssumeCapacityNoClobber(id, .{ .lease = lease, .len = plan.len, .pos = e.cur.st.pos, .mtp_len = e.cur.st.mtp_len, .cur = cur, .bases = .{ e.cur.st.base, e.cur.st.mtp_base }, .compact = compact });
}

pub fn restore(e: anytype, id: u64) !void {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    const plan = try kda_replay.restorePlan(e, id, try layout.Plan.init(&e.cur.st, &e.c, s.pos, if (e.mtp) s.mtp_len else null));
    if (plan.len != s.len) return error.SnapshotSize;
    errdefer cuda.completion.drain(e.d, e.stream, "a failed GLM snapshot restore did not drain");
    if (s.compact) |x| try e.snap_codec.?.restore(pool, e.stream, &s.lease, x, &plan) else try pool.move(e.stream, &s.lease, &plan, false);
    // TF_GLM_INDEX_PRUNE: the restored pooled keys' strips, from the rows themselves (an entry carries none)
    if (e.m.prune) |*k| try k.refresh(e.stream, &e.cur.st, s.pos, if (e.mtp) s.mtp_len else 0);
    try e.cur.st.restoreHost(e.stream, s.pos, s.mtp_len, s.cur, s.bases);
    e.cur.in_head = 0;
    try e.stream.synchronize();
    try kda_replay.afterRestore(e, id); // TF_GLM_KDA_REPLAY: KDA states rebuilt (or checked) from the entry's tokens
}

/// The pages a kept state holds for the store's charge: its lease (raw, or trimmed once compacted), or with peers and
/// an eager codec its compacted pages (this rank's own; peers may hold more until the gate agrees, so the leader's
/// charge can only understate them).
pub fn charged(e: anytype, id: u64) u64 {
    const s = e.snaps.get(id).?;
    const eager_peers = e.share.world > 1 and e.snap_codec != null and !e.snap_codec.?.lazy;
    return @as(u64, if (s.compact) |x| (if (eager_peers) x.pages() else s.lease.ids.items.len) else s.lease.ids.items.len) * pages.page;
}

/// The store asks again (at admission): a state compacted since its keep gives back its pages past the payloads
/// (alone: now; with peers: at the gate's agreement) and is charged its lease from here on.
pub fn recharged(e: anytype, id: u64) u64 {
    const s = e.snaps.getPtr(id) orelse return std.math.maxInt(u64); // not held: never lowers a charge
    if (s.compact) |x| {
        if (e.share.world == 1) e.snap_codec.?.settle(&e.snap_pages.?, &s.lease, x);
        e.snap_codec.?.claim(x);
    }
    return charged(e, id);
}

/// Free pages past those given back since the store last charged their states (it still counts them held).
pub fn spare(e: anytype) u64 {
    const p = if (e.snap_pages) |*p| p else return 0;
    const held_out: u64 = if (e.snap_codec) |*c| c.unclaimed() else 0;
    return p.spare() -| held_out * pages.page;
}

pub fn holds(e: anytype, id: u64) bool {
    return e.snaps.contains(id);
}

pub fn drop(e: anytype, id: u64) void {
    var kv = e.snaps.fetchRemove(id) orelse return;
    free(e, &kv.value);
    if (e.share.rank == 0 and e.announce == id) {
        e.announce = 0; // the store failed to publish this keep, so no follower has it yet
        return;
    }
    if (e.share.rank == 0 and e.share.world > 1) e.dropped.appendAssumeCapacity(@intCast(id));
}

pub fn free(e: anytype, s: *types.Snap) void {
    if (s.compact) |x| e.snap_codec.?.drop(x); // a compaction under way stops before its pages go back
    e.snap_pages.?.give(&s.lease);
    e.gpa.free(s.cur);
}

/// Odd words release an evicted id before an even word announces its replacement; an idle gate still sends one word.
pub fn gate(e: anytype) !void {
    const leader = e.share.rank == 0;
    var dropped: usize = 0;
    while (true) {
        const word: u64 = if (!leader) 0 else if (dropped < e.dropped.items.len)
            2 * @as(u64, e.dropped.items[dropped]) + 1
        else
            2 * e.announce;
        const shared = try e.shared(@floatFromInt(word));
        if (!std.math.isFinite(shared) or shared < 0 or shared > 2 * @as(f64, std.math.maxInt(u32)) + 1 or @floor(shared) != shared)
            return error.BadSnapshotControl;
        const got: u64 = @intFromFloat(shared);
        if (got & 1 != 0) {
            if (leader) dropped += 1 else e.dropState(got / 2);
            continue;
        }
        var failed: ?anyerror = null;
        if (leader) {
            e.dropped.clearRetainingCapacity();
            e.announce = 0;
        } else if (got > 0) e.keepAs(got / 2) catch |err| {
            failed = err; // it still joins the page agreement, so no peer waits on it
        };
        if (comptime @hasField(@TypeOf(e.*), "snap_codec")) // the gate's own tests drive it with a bare control
            if (e.snap_codec != null and e.share.world > 1) {
                // lazy: every copy is raw at its keep, so only a later compaction has pages to agree on
                if (e.snap_codec.?.lazy) try codec.agreeLazy(e.share.rank == 0, &e.snaps, &e.snap_pages.?, &e.snap_codec.?, e) else if (got > 0) try agreePages(e, got / 2);
            };
        if (failed) |err| return err;
        return;
    }
}

/// Every rank keeps the most pages any rank's compacted copy of `id` needs: page use stays equal across ranks, so
/// the leader's free pages (which its store decides on) are every rank's.
fn agreePages(e: anytype, id: u64) !void {
    const s = e.snaps.getPtr(id);
    const mine: usize = if (s) |x| (if (x.compact) |k| k.pages() else x.lease.ids.items.len) else 0;
    const most = try e.shared(@floatFromInt(mine));
    if (!std.math.isFinite(most) or most < 0 or most > @as(f64, std.math.maxInt(u32)) or @floor(most) != most)
        return error.BadSnapshotControl;
    const x = s orelse return;
    if (x.compact != null) e.snap_pages.?.trim(&x.lease, @intFromFloat(most));
}

/// After all ranks agree to cancel, publish the last completed state before any rank releases or rebinds its slot.
pub fn interrupted(e: anytype, marks: ?types.Marks, at: u32) !void {
    if (marks) |m| if (m.interrupted) |save_prefix| save_prefix(m.ctx, at);
    try gate(e);
}
