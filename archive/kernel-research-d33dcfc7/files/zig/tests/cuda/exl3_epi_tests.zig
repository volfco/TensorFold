//! Common EXL3 fused tails against the unfused served slot path, including pruned routes and graph replay.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const ref = @import("exl3_experts_f64_tests.zig");
const guards = @import("contract_rig.zig");
const ex = cuda.exl3_experts;
const D = ref.D;
const N = 256;
const slots = ref.SLOTS;

fn same(r: *qr.RigT, a: ex.Scratch, b: ex.Scratch, rows: usize) !void {
    try check.sameBytes("EXL3 fused Xd", try r.back(a.xd, rows * slots * N * 2), try r.back(b.xd, rows * slots * N * 2));
    try check.sameBytes("EXL3 fused per-slot y", try r.back(a.y, rows * slots * D * 4), try r.back(b.y, rows * slots * D * 4));
    const ticket = try r.back(a.ticket, ex.epilogues.ticketWords(.{ .gate = 0, .up = 0, .down = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .experts = ref.E, .dims = D, .width = N, .codebook = .mcg }, rows, slots) * 4);
    for (ticket) |v| try check.expect(v == 0, "EXL3 fused counters reset", .{});
}

fn timed(r: *qr.RigT, owner: *const ex.Experts, l: ex.Layer, x: u64, pick: u64, rows: usize, sc: ex.Scratch) !f32 {
    var t0 = try cuda.Event.init(r.gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(r.gpu.d, true);
    defer t1.deinit();
    try t0.record(r.s);
    for (0..10) |_| try owner.runSlots(r.s, l, x, D, .bf16, pick, rows, sc);
    try t1.record(r.s);
    try t1.synchronize();
    return (try cuda.Event.elapsedMs(t0, t1)) / 10;
}

fn cell(r: *qr.RigT, owner: *const ex.Experts, b: *const ref.Built, layer: ex.Layer, rows: usize, x: u64, nonfinite: bool) !void {
    var gr: guards.Rig = .{ .gpu = r.gpu, .s = r.s };
    defer gr.deinit();
    const bytes = try ex.scratchBytes(layer, rows, slots);
    const candidate = try ex.scratch(try gr.scratch(bytes), layer, rows, slots);
    const baseline = try ex.scratch(try gr.scratch(bytes), layer, rows, slots);
    var old = owner.*;
    old.fuse_epilogues = false;
    old.grouped256 = null;
    var fused = old;
    fused.fuse_epilogues = true;
    for (0..5) |_| {
        try old.runSlots(r.s, layer, x, D, .bf16, b.dpick, rows, baseline);
        try fused.runSlots(r.s, layer, x, D, .bf16, b.dpick, rows, candidate);
        try same(r, candidate, baseline, rows);
    }
    try cuda.graph.beginCapture(r.s, .thread_local);
    try fused.runSlots(r.s, layer, x, D, .bf16, b.dpick, rows, candidate);
    var graph = try cuda.graph.endCapture(r.s);
    defer graph.deinit();
    var exec = try graph.instantiate();
    defer exec.deinit();
    for (0..5) |_| {
        try exec.launchOn(r.s);
        try same(r, candidate, baseline, rows);
    }
    try check.expect(try gr.intact(), "EXL3 fused output and scratch guards", .{});
    if (nonfinite) {
        const raw = try r.back(candidate.y, rows * slots * D * 4);
        var found = false;
        for (0..rows * slots * D) |i| {
            const v: f32 = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
            found = found or !std.math.isFinite(v);
        }
        try check.expect(found, "EXL3 fused nonfinite plant reaches output", .{});
    }
    if (rows == 8 and !nonfinite and layer.act_mode == 0 and layer.limit > 0) {
        var old_ms: [6]f32 = undefined;
        var new_ms: [6]f32 = undefined;
        for (0..6) |i| {
            if (i % 2 == 0) {
                old_ms[i] = try timed(r, &old, layer, x, b.dpick, rows, baseline);
                new_ms[i] = try timed(r, &fused, layer, x, b.dpick, rows, candidate);
            } else {
                new_ms[i] = try timed(r, &fused, layer, x, b.dpick, rows, candidate);
                old_ms[i] = try timed(r, &old, layer, x, b.dpick, rows, baseline);
            }
        }
        std.mem.sort(f32, &old_ms, {}, std.sort.asc(f32));
        std.mem.sort(f32, &new_ms, {}, std.sort.asc(f32));
        std.debug.print("RESULT exl3-fused diagnostic-cache-resident {t}: baseline_ms={d:.6} candidate_ms={d:.6} ratio={d:.4}\n", .{ layer.codebook, old_ms[3], new_ms[3], new_ms[3] / old_ms[3] });
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var prng: std.Random.DefaultPrng = .init(0xe917);
    var r: qr.RigT = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*buf| buf.free();
        r.bufs.deinit(gpu.gpa);
    }
    var owner = try ex.Experts.load(gpu.d);
    defer owner.unload();
    const bits = [_]u32{ 4, 5, 6, 4, 5, 6, 4, 5, 6 };
    var b = try ref.buildLayer(&r, &bits);
    for (0..300) |row| b.pick[row * slots + slots - 1] = if (row % 2 == 0) -1 else ref.E;
    b.dpick = try r.dev(std.mem.sliceAsBytes(b.pick));
    const finite_x = b.dx;
    const nf = try r.a.dupe(u16, b.x);
    nf[7] = 0x7fc0;
    nf[D + 9] = 0x7f80;
    nf[2 * D + 11] = 0xff80;
    const nonfinite_x = try r.dev(std.mem.sliceAsBytes(nf));
    const zero = try r.a.alloc(u16, 4 * D);
    @memset(zero, 0x8000);
    const zero_x = try r.dev(std.mem.sliceAsBytes(zero));
    var cases: usize = 0;
    for ([_]ex.Codebook{ .inst3, .mcg, .mul1 }) |cb| for ([_]i32{ 0, 1 }) |mode| for ([_]f32{ 0, 10 }) |limit| {
        var l = b.layer;
        l.codebook = cb;
        l.act_mode = mode;
        l.limit = limit;
        for ([_]usize{ 1, 3, 8, 16, 17, 32 }) |rows| {
            try cell(&r, &owner, &b, l, rows, finite_x, false);
            cases += 1;
        }
        try cell(&r, &owner, &b, l, 4, nonfinite_x, true);
        cases += 1;
        try cell(&r, &owner, &b, l, 4, zero_x, false);
        cases += 1;
    };
    check.pass("exl3-fused: {d} cells, Xd/y byte equality, pruned slots, five eager and graph replays, reset counters, guards", .{cases});
}

pub fn contract(gpu: check.Gpu) !void {
    const core = @import("core");
    const product = @import("contract_exl3_n256.zig");
    const fused = @import("contract_exl3_epi.zig");
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var loaded = try product.Loaded.init(gpu);
    defer loaded.deinit();
    const kernels = try ex.epilogues.Kernels.resolve(loaded.module);
    var items: [6]fused.Product = undefined;
    for (&items, 0..) |*p, i| p.* = .{
        .product = .{ .rig = .{ .gpu = gpu, .s = stream }, .loaded = &loaded, .sk = if (i % 3 == 1) 1 else 4, .mats = if (i % 3 == 1) 1 else 2 },
        .kernels = kernels,
        .fault = i % 3 == 2,
        .tail = i >= 3,
    };
    defer for (&items) |*p| p.deinit();
    var registrations: [6]core.contract.gates.Kernel = undefined;
    for (&items, &registrations) |*p, *k| k.* = p.kernel();
    var tally: core.contract.gates.Tally = .{};
    try core.contract.gates.runAll(gpu.gpa, &registrations, 0xe917, &tally);
    std.debug.print("SUMMARY exl3-fused focused kernels={d} passed={d} failed={d} controls={d} caught={d} missed={d}; launch limits belong to full runner\n", .{ tally.kernels, tally.passed, tally.failed, tally.controls, tally.caught, tally.missed });
    if (tally.problems() != 0) return error.TestFailed;
}
