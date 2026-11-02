//! Focused N256 replacement gates and alternating product timings over a weight bank larger than GB10's L2.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const product = @import("contract_exl3_n256.zig");

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var loaded = try product.Loaded.init(gpu);
    defer loaded.deinit();
    var up: product.Product = .{ .rig = .{ .gpu = gpu, .s = s }, .loaded = &loaded };
    defer up.deinit();
    var down: product.Product = .{ .rig = .{ .gpu = gpu, .s = s }, .loaded = &loaded, .sk = 1, .mats = 1 };
    defer down.deinit();
    var bad: product.Product = .{ .rig = .{ .gpu = gpu, .s = s }, .loaded = &loaded, .fault = true };
    defer bad.deinit();
    var small: product.Product = .{ .rig = .{ .gpu = gpu, .s = s }, .loaded = &loaded, .sk = 1, .mats = 1, .small = true };
    defer small.deinit();
    var bad_small: product.Product = .{ .rig = .{ .gpu = gpu, .s = s }, .loaded = &loaded, .sk = 1, .mats = 1, .small = true, .fault = true };
    defer bad_small.deinit();
    const kernels = [_]core.contract.gates.Kernel{ up.kernel(), down.kernel(), bad.kernel(), small.kernel(), bad_small.kernel() };
    var tally: core.contract.gates.Tally = .{};
    try core.contract.gates.runAll(gpu.gpa, &kernels, 0x256, &tally);
    std.debug.print("SUMMARY N256 focused: {d} kernels pass, {d} fail; {d} controls red, {d} missed; limits not run\n", .{ tally.passed, tally.failed, tally.caught, tally.missed });
    try check.expect(tally.problems() == 0, "N256 replacement gates", .{});
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for ([_]usize{ 1, 4, 8, 16 }) |rows| {
        try bench(&up, .{ .rows = rows, .n = 1024, .k = 4096 }, start, end);
        try bench(&down, .{ .rows = rows, .n = 4096, .k = 1024 }, start, end);
        if (rows <= 8) try bench(&small, .{ .rows = rows, .n = 4096, .k = 1024 }, start, end);
    }
}

fn measure(p: *product.Product, candidate: bool, banks: []const u64, start: cuda.Event, end: cuda.Event) !f32 {
    const original = p.args.tp0;
    defer {
        p.args.tp0 = original;
        p.args.tp1 = original;
    }
    try start.record(p.rig.s);
    for (0..10) |_| for (banks) |bank| {
        p.args.tp0 = bank;
        p.args.tp1 = bank;
        try p.launch(candidate, p.out);
    };
    try end.record(p.rig.s);
    try end.synchronize();
    return (try start.elapsedMs(end)) / @as(f32, @floatFromInt(10 * banks.len));
}

fn bench(p: *product.Product, shape: core.contract.spec.Shape, start: cuda.Event, end: cuda.Event) !void {
    try check.expect(try p.prepare(shape, .random, 0x6256), "N256 benchmark shape admitted", .{});
    try p.run();
    const got = try p.rig.gpu.gpa.dupe(u8, try p.rig.outputBytes());
    defer p.rig.gpu.gpa.free(got);
    try check.sameBytes("N256 benchmark output", got, try p.replaced());
    const tr = std.mem.bytesToValue(u64, try p.rig.bytes(p.args.tp0, 8));
    const raw = try p.rig.bytes(tr, shape.k * shape.n / 2);
    var banks: [16]u64 = undefined;
    banks[0] = p.args.tp0;
    for (banks[1..]) |*bank| {
        const weights = try p.rig.input(raw);
        bank.* = try p.rig.input(std.mem.asBytes(&weights));
    }
    var baseline: [6]f32 = undefined;
    var candidate: [6]f32 = undefined;
    for (0..6) |rep| {
        if (rep % 2 == 0) {
            baseline[rep] = try measure(p, false, &banks, start, end);
            candidate[rep] = try measure(p, true, &banks, start, end);
        } else {
            candidate[rep] = try measure(p, true, &banks, start, end);
            baseline[rep] = try measure(p, false, &banks, start, end);
        }
        std.debug.print("BENCH N256 small={s} weight_bank_bytes={d} rows={d} n={d} k={d} sk={d} rep={d} warmup={s} n128_ms={d:.6} n256_ms={d:.6}\n", .{ if (p.small) "true" else "false", banks.len * raw.len, shape.rows, shape.n, shape.k, p.sk, rep, if (rep == 0) "true" else "false", baseline[rep], candidate[rep] });
    }
    try check.sameBytes("N256 benchmark retained bytes", try p.rig.outputBytes(), got);
}
