//! TF_GLM_KDA_REPLAY's host pieces: the token window, the per-head comparison, the knobs and the stripped plan.
const std = @import("std");
const t = std.testing;
const replay = @import("cuda_kda_replay.zig");
const ranks = @import("cuda_ranks.zig");
const layout = @import("cuda_snapshot_layout.zig");

test "GLM KDA replay: a slot's window is the last tokens before its position, and unknown after a gap" {
    var buf: [4]u32 = undefined;
    var r: replay.Recent = .{ .ids = &buf };
    try t.expect(r.before(0) == null);
    r.note(0, &.{ 1, 2 });
    try t.expectEqualSlices(u32, &.{ 1, 2 }, r.before(2).?); // shorter than the window: from position 0
    try t.expect(r.before(3) == null); // not where the tokens end
    r.note(2, &.{ 3, 4, 5 });
    try t.expectEqualSlices(u32, &.{ 2, 3, 4, 5 }, r.before(5).?);
    r.note(5, &.{ 6, 7, 8, 9, 10, 11 }); // longer than the window: its last four
    try t.expectEqualSlices(u32, &.{ 8, 9, 10, 11 }, r.before(11).?);
    r.note(20, &.{12}); // a gap (a decode commit in between): only what follows it is known
    try t.expect(r.before(21) == null);
    r.note(21, &.{ 13, 14, 15 });
    try t.expectEqualSlices(u32, &.{ 12, 13, 14, 15 }, r.before(24).?);
}

test "GLM KDA replay: per-head relative error, max abs error and bit identity, conv by its q, k and v blocks" {
    const heads = 3;
    const per = 128 * 128;
    const a = try t.allocator.alloc(f32, heads * per);
    defer t.allocator.free(a);
    const b = try t.allocator.alloc(f32, heads * per);
    defer t.allocator.free(b);
    for (a, 0..) |*x, i| x.* = @floatFromInt(i % 7 + 1);
    @memcpy(b, a);
    b[per + 5] += 0.5; // head 1: one element off
    for (b[2 * per ..][0..per]) |*x| x.* *= 1.1; // head 2: 10 % everywhere
    var s: [heads]replay.Stat = undefined;
    replay.recStats(a, b, &s);
    try t.expect(s[0].same and s[0].rel == 0 and s[0].max_abs == 0);
    try t.expect(!s[1].same and s[1].max_abs == 0.5);
    var ref: f64 = 0;
    for (a[per..][0..per]) |x| ref += @as(f64, x) * x;
    try t.expectApproxEqRel(0.5 / @sqrt(ref), s[1].rel, 1e-9);
    try t.expectApproxEqRel(0.1, s[2].rel, 1e-6);
    // conv [3][q | k | v of heads x 128] bf16: head 1's channels in all three blocks and rows, nothing of the others
    const c = 3 * heads * 128;
    var ca: [3 * c]u16 = undefined;
    for (&ca, 0..) |*x, i| x.* = 0x3f80 + @as(u16, @intCast(i % 3)); // about 1.0
    var cb = ca;
    for (0..3) |row| for (0..3) |g| {
        cb[row * c + g * heads * 128 + 128 + 7] = 0x4000; // 2.0
    };
    replay.convStats(&ca, &cb, &s);
    try t.expect(s[0].same and s[2].same and !s[1].same);
    try t.expect(s[1].max_abs > 0.9 and s[1].max_abs < 1.1);
    // a zero stored head: 0 when both are zero, infinite otherwise
    @memset(a[0..per], 0);
    @memset(b[0..per], 0);
    replay.recStats(a, b, &s);
    try t.expectEqual(@as(f64, 0), s[0].rel);
    b[0] = 1;
    replay.recStats(a, b, &s);
    try t.expect(std.math.isInf(s[0].rel));
}

const Env = struct {
    var w: ?[]const u8 = null;
    var keep: ?[]const u8 = null;
    var check: ?[]const u8 = null;
    fn get(key: [:0]const u8) ?[]const u8 {
        if (std.mem.eql(u8, key, "TF_GLM_KDA_REPLAY")) return w;
        if (std.mem.eql(u8, key, "TF_GLM_KDA_REPLAY_KEEP")) return keep;
        if (std.mem.eql(u8, key, "TF_GLM_KDA_REPLAY_CHECK")) return check;
        return null;
    }
};

test "GLM KDA replay: the knobs are refused by name, default off, and must agree between ranks" {
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(&Env.get, &problem);
    try t.expectEqual(@as(u32, 0), off.kda_replay);
    try t.expect(!off.kda_replay_keep and !off.kda_replay_check);
    Env.keep = "1";
    try t.expectError(error.BadSetting, ranks.Settings.read(&Env.get, &problem));
    try t.expectEqualStrings("TF_GLM_KDA_REPLAY_KEEP=1 requires TF_GLM_KDA_REPLAY (the window in tokens)", problem);
    Env.keep = null;
    Env.check = "1";
    try t.expectError(error.BadSetting, ranks.Settings.read(&Env.get, &problem));
    try t.expectEqualStrings("TF_GLM_KDA_REPLAY_CHECK=1 requires TF_GLM_KDA_REPLAY (the window in tokens)", problem);
    for ([_][]const u8{ "2049", "-1", "many" }) |bad| {
        Env.w = bad;
        try t.expectError(error.BadSetting, ranks.Settings.read(&Env.get, &problem));
        try t.expectEqualStrings("TF_GLM_KDA_REPLAY: 0 (off) to 2048 tokens", problem);
    }
    Env.w = "256";
    Env.check = "yes";
    try t.expectError(error.BadSetting, ranks.Settings.read(&Env.get, &problem));
    try t.expectEqualStrings("TF_GLM_KDA_REPLAY_CHECK: 0 or 1", problem);
    Env.check = "1";
    const on = try ranks.Settings.read(&Env.get, &problem);
    defer {
        Env.w = null;
        Env.check = null;
    }
    try t.expect(on.kda_replay == 256 and on.kda_replay_check and !on.kda_replay_keep);
    const h = ranks.hashBytes("x");
    var all = [_]ranks.Fingerprint{
        ranks.Fingerprint.of(2, 0, .{ .kda_replay = 256 }, 8192, true, h, h, h),
        ranks.Fingerprint.of(2, 1, on, 8192, true, h, h, h),
    };
    try t.expectEqual(ranks.Mismatch{ .field = .kda_replay, .rank = 1 }, ranks.mismatch(&all).?);
    all[0] = ranks.Fingerprint.of(2, 0, on, 8192, true, h, h, h);
    try t.expect(ranks.mismatch(&all) == null);
    try t.expectError(error.KdaReplayWindow, replay.Replay.create(t.allocator, 0, false, false, undefined));
    try t.expectError(error.KdaReplayWindow, replay.Replay.create(t.allocator, 2049, false, false, undefined));
}

test "GLM KDA replay: a stripped plan is the attention classes alone, marked with its window" {
    var p: layout.Plan = .{};
    const classes = [_]layout.Class{ .prefix, .latent, .ring, .ring, .pooled, .tail };
    for (classes, 0..) |c, i| {
        p.parts[i] = .{ .live = 4096 * i, .len = 100 + i, .class = c };
        p.len += 100 + i;
    }
    p.count = classes.len;
    const s = try replay.strip(p, 64);
    try t.expectEqual(@as(usize, 5), s.count);
    try t.expectEqual(p.len - 100, s.len);
    try t.expectEqual(@as(u32, 64), s.replay);
    for (s.parts[0..s.count], classes[1..]) |x, c| try t.expectEqual(c, x.class);
    try t.expectError(error.SnapshotShape, replay.strip(s, 64)); // no head left to take
    try t.expectError(error.SnapshotShape, replay.strip(p, 0));
}
