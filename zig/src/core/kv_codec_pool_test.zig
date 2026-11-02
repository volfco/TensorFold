//! kv_codec_pool: the CPU list parser and its refusals, the affinity probe, and the persistent parallel for.
const std = @import("std");
const t = std.testing;
const wp = @import("kv_codec_pool.zig");

fn text(s: *const wp.Cpus, buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    s.format(&w) catch unreachable;
    return w.buffered();
}

test "CPU lists: taskset ranges parse to their CPUs and print back the same" {
    var bad: []const u8 = undefined;
    const s = try wp.parse("0-4,10-14", &bad);
    try t.expectEqual(@as(usize, 10), s.count());
    for ([_]usize{ 0, 4, 10, 14 }) |cpu| try t.expect(s.has(cpu));
    for ([_]usize{ 5, 9, 15, 1023 }) |cpu| try t.expect(!s.has(cpu));
    var buf: [64]u8 = undefined;
    try t.expectEqualStrings("0-4,10-14", text(&s, &buf));
    const one = try wp.parse("7,3,1023", &bad);
    try t.expectEqualStrings("3,7,1023", text(&one, &buf));
    var other = one;
    other.words[0] = 0;
    try t.expectEqual(@as(?usize, 3), one.outside(&other));
    try t.expectEqual(@as(?usize, null), other.outside(&one));
}

test "CPU lists: every malformed list is refused with the item that broke it" {
    const Case = struct { in: []const u8, err: wp.ParseError, bad: []const u8 };
    const cases = [_]Case{
        .{ .in = "", .err = error.CpuListEmpty, .bad = "" },
        .{ .in = "1,,2", .err = error.CpuListSyntax, .bad = "" },
        .{ .in = "0-4,", .err = error.CpuListSyntax, .bad = "" },
        .{ .in = "x", .err = error.CpuListSyntax, .bad = "x" },
        .{ .in = "1, 2", .err = error.CpuListSyntax, .bad = " 2" },
        .{ .in = "3-", .err = error.CpuListSyntax, .bad = "3-" },
        .{ .in = "-3", .err = error.CpuListSyntax, .bad = "-3" },
        .{ .in = "1-2-3", .err = error.CpuListSyntax, .bad = "1-2-3" },
        .{ .in = "0,9-5", .err = error.CpuListReversed, .bad = "9-5" },
        .{ .in = "1024", .err = error.CpuListTooLarge, .bad = "1024" },
        .{ .in = "1000-1030", .err = error.CpuListTooLarge, .bad = "1000-1030" },
        .{ .in = "99999999999999999999999", .err = error.CpuListTooLarge, .bad = "99999999999999999999999" },
        .{ .in = "0-3,2", .err = error.CpuListRepeated, .bad = "2" },
        .{ .in = "5,5", .err = error.CpuListRepeated, .bad = "5" },
    };
    for (cases) |x| {
        var bad: []const u8 = "unset";
        try t.expectError(x.err, wp.parse(x.in, &bad));
        try t.expectEqualStrings(x.bad, bad);
    }
}

test "CPU lists: the process's own CPUs are allowed; a CPU it cannot run on is refused by number" {
    if (!wp.pinnable) return error.SkipZigTest;
    const mine = wp.current() orelse return error.SkipZigTest;
    try t.expect(mine.count() > 0);
    try t.expectEqual(@as(?usize, null), wp.allowed(&mine));
    try t.expectEqualDeep(mine, wp.current().?); // the probe ran on its own thread: this one's mask is unchanged
    const cpus = std.Thread.getCpuCount() catch return error.SkipZigTest;
    if (cpus + 1 >= wp.max_cpus) return error.SkipZigTest;
    var bad: []const u8 = undefined;
    var buf: [16]u8 = undefined;
    const absent = try wp.parse(try std.fmt.bufPrint(&buf, "{d}", .{wp.max_cpus - 1}), &bad);
    try t.expectEqual(@as(?usize, wp.max_cpus - 1), wp.allowed(&absent)); // no allowed CPU at all
    var mixed = mine;
    mixed.add(wp.max_cpus - 1);
    try t.expectEqual(@as(?usize, wp.max_cpus - 1), wp.allowed(&mixed)); // one allowed, one dropped by the kernel
}

const Count = struct {
    hits: []std.atomic.Value(u32),
    seen: []?wp.Cpus,
    fn run(x: *Count, i: usize, w: usize) void {
        _ = x.hits[i].fetchAdd(1, .monotonic);
        if (w < x.seen.len) x.seen[w] = wp.current();
    }
};

test "worker pool: every index runs exactly once per run, on threads made once" {
    var p = try wp.Pool.init(t.allocator, 5, null);
    defer p.deinit();
    try t.expectEqual(@as(usize, 5), p.size());
    var hits: [1000]std.atomic.Value(u32) = @splat(.init(0));
    var seen: [5]?wp.Cpus = @splat(null);
    var x: Count = .{ .hits = &hits, .seen = &seen };
    for (0..20) |_| p.run(hits.len, &x, Count.run);
    p.run(0, &x, Count.run);
    for (hits) |h| try t.expectEqual(@as(u32, 20), h.load(.monotonic));
    // the same threads served every run
    try t.expectEqual(@as(usize, 5), p.threads.len);
}

test "worker pool: pinned workers run only on their list" {
    if (!wp.pinnable) return error.SkipZigTest;
    const mine = wp.current() orelse return error.SkipZigTest;
    var one: wp.Cpus = .{};
    for (0..wp.max_cpus) |cpu| if (mine.has(cpu)) {
        one.add(cpu);
        break;
    };
    var p = try wp.Pool.init(t.allocator, 3, one);
    defer p.deinit();
    var hits: [64]std.atomic.Value(u32) = @splat(.init(0));
    var seen: [3]?wp.Cpus = @splat(null);
    var x: Count = .{ .hits = &hits, .seen = &seen };
    p.run(hits.len, &x, Count.run);
    try t.expectEqual(@as(usize, 0), p.refused());
    var any = false;
    for (seen) |s| if (s) |got| {
        any = true;
        try t.expectEqualDeep(one, got);
    };
    try t.expect(any);
}

test "worker pool: two callers at once are served one run at a time, each complete" {
    var p = try wp.Pool.init(t.allocator, 4, null);
    defer p.deinit();
    const Caller = struct {
        fn go(pool: *wp.Pool, hits: []std.atomic.Value(u32)) void {
            var x: Count = .{ .hits = hits, .seen = &.{} };
            for (0..50) |_| pool.run(hits.len, &x, Count.run);
        }
    };
    var a: [300]std.atomic.Value(u32) = @splat(.init(0));
    var b: [200]std.atomic.Value(u32) = @splat(.init(0));
    const th = try std.Thread.spawn(.{}, Caller.go, .{ &p, &a });
    Caller.go(&p, &b);
    th.join();
    for (a) |h| try t.expectEqual(@as(u32, 50), h.load(.monotonic));
    for (b) |h| try t.expectEqual(@as(u32, 50), h.load(.monotonic));
}
