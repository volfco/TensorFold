//! `tf-glm-expert-prefetch-bench`: GLM's routed NVFP4 decode experts (gate-up, then fp32 down) on synthetic weights,
//! stock against TF_GLM_EXPERT_PREFETCH's L2 prefetch and TF_NVFP4_DECODE_RING's deeper load ring. Every variant's
//! outputs must hash equal to the stock run's, or the program fails. GB10 only; the two images are built by hand
//! (tools/kvprobe/STATUS-track4.md). Times are event times on one stream after an L2 flush, the first rep discarded.
//! The stock kernels come from the ring image too (it includes nvfp4_experts.cu), so this program needs no embedded
//! fatbins and stock and ring are built by the same nvcc with the same flags.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const prefetch = glm.expert_prefetch;
const ring_mod = glm.expert_ring;

const usage =
    \\usage: tf-glm-expert-prefetch-bench --ring-image F [--prefetch-image F] [--rows R] [--dims D] [--width NI]
    \\         [--experts E] [--reps N] [--gap-cycles C] [--groups G] [--seed S]
    \\  --dims, --width: the checkpoint's hidden size and its routed intermediate width on this rank (defaults 4096, 768)
    \\  --gap-cycles: a one-thread spin between the pick and the experts (the plan and shared expert a prefetch hides
    \\                behind); needs --prefetch-image, which holds the spin kernel. --reps N >= 3 (first discarded)
    \\
;

const Opts = struct {
    prefetch_image: ?[:0]const u8 = null,
    ring_image: ?[:0]const u8 = null,
    rows: usize = 1,
    dims: usize = 4096,
    width: usize = 768,
    experts: usize = 256,
    reps: usize = 5,
    gap: i64 = 20000,
    groups: usize = 2,
    seed: u64 = 1,
    top_k: usize = 8,
};

fn parseArgs(args: []const [:0]const u8) !Opts {
    var o: Opts = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) return error.MissingValue;
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--prefetch-image")) o.prefetch_image = v else if (std.mem.eql(u8, k, "--ring-image")) o.ring_image = v else if (std.mem.eql(u8, k, "--rows")) o.rows = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--dims")) o.dims = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--width")) o.width = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--experts")) o.experts = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--reps")) o.reps = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--gap-cycles")) o.gap = try std.fmt.parseInt(i64, v, 10) else if (std.mem.eql(u8, k, "--groups")) o.groups = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--seed")) o.seed = try std.fmt.parseInt(u64, v, 10) else return error.UnknownOption;
    }
    if (o.ring_image == null or o.reps < 3 or o.rows < 1 or o.rows > 64 or o.dims % 32 != 0 or o.width % 32 != 0 or o.experts < o.top_k) return error.BadOption;
    return o;
}

/// Pairs (row * slots + slot) grouped by expert into items of at most 16 (the decode plan's tile), on the host.
const HostPlan = struct { items: []i32, members: []i32, count: i32 };

fn hostPlan(a: std.mem.Allocator, picks: []const i32, experts: usize) !HostPlan {
    var members: std.ArrayList(i32) = .empty;
    var items: std.ArrayList(i32) = .empty;
    for (0..experts) |e| {
        var first: usize = members.items.len;
        var cnt: usize = 0;
        for (picks, 0..) |p, pair| if (p == @as(i32, @intCast(e))) {
            try members.append(a, @intCast(pair));
            cnt += 1;
            if (cnt == 16) {
                try items.appendSlice(a, &.{ @intCast(e), @intCast(first), 16 });
                first += 16;
                cnt = 0;
            }
        };
        if (cnt > 0) try items.appendSlice(a, &.{ @intCast(e), @intCast(first), @intCast(cnt) });
    }
    return .{ .items = items.items, .members = members.items, .count = @intCast(items.items.len / 3) };
}

/// Random NVFP4 blocks whose 16 scale words hold finite e4m3 scales in [2^-4, 2^0] (no NaN, no overflow).
fn weights(rng: std.Random, bytes: []u8) void {
    rng.bytes(bytes);
    var at: usize = 0;
    while (at + 576 <= bytes.len) : (at += 576) for (bytes[at + 512 .. at + 576]) |*b| {
        b.* = 0x20 + (b.* & 0x18); // e4m3 exponent field 4..7, mantissa 0: 2^-3 .. 2^0
    };
}

const Variant = struct { name: []const u8, pf: ?prefetch.Mode = null, ring: ?u8 = null };

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    const o = parseArgs(args) catch {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const d = &driver;
    const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
    var pf: ?prefetch.Prefetcher = null;
    if (o.prefetch_image) |p| pf = try prefetch.Prefetcher.load(d, try prefetch.readImageAt(a, p.ptr), .{ .mode = .bulk, .groups = o.groups, .rows = 64 });
    defer if (pf) |*x| x.unload();
    const spin: ?cuda.Function = if (pf) |p| try p.module.function("tf_glm_expert_prefetch_spin") else null;
    const ring_image: []const u8 = try prefetch.readImageAt(a, o.ring_image.?.ptr);
    const stock_names = [2][:0]const u8{ cuda.experts.Nvfp4Experts.symbols.gate_up, cuda.experts.Nvfp4Experts.symbols.down_f32 };

    // E: the routed experts plus a stand-in at the shared slot's id, which the routed kernels skip (cuda_moe.body)
    const E = o.experts + 1;
    const slots = o.top_k + 1;
    const pairs = o.rows * slots;
    var prng = std.Random.DefaultPrng.init(o.seed);
    const rng = prng.random();
    const up_bytes = cuda.experts.Nvfp4Experts.bytes(E, o.width, o.dims, 2);
    const down_bytes = cuda.experts.Nvfp4Experts.bytes(E, o.dims, o.width, 1);
    const host_up = try a.alloc(u8, up_bytes);
    const host_down = try a.alloc(u8, down_bytes);
    weights(rng, host_up);
    weights(rng, host_down);
    var up = try cuda.DeviceBuffer.fromHost(d, host_up);
    defer up.free();
    var down = try cuda.DeviceBuffer.fromHost(d, host_down);
    defer down.free();
    const ones = try a.alloc(f32, 2 * E);
    @memset(ones, 1.0);
    var scales = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ones));
    defer scales.free();
    const xs = try a.alloc(u16, o.rows * o.dims);
    // bf16 inputs k / 64, k in [-64, 64]: exact in bf16, so the truncation below rounds nothing
    for (xs) |*v| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(rng.intRangeAtMost(i32, -64, 64))) / 64.0)) >> 16);
    var x = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(xs));
    defer x.free();
    const picks = try a.alloc(i32, pairs);
    for (0..o.rows) |r| {
        var k: usize = 0;
        while (k < o.top_k) {
            const e: i32 = @intCast(rng.uintLessThan(usize, o.experts));
            if (std.mem.indexOfScalar(i32, picks[r * slots ..][0..k], e) != null) continue;
            picks[r * slots + k] = e;
            k += 1;
        }
        picks[r * slots + o.top_k] = @intCast(o.experts); // the shared slot
    }
    const plan = try hostPlan(a, picks, E);
    var picks_dev = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(picks));
    defer picks_dev.free();
    var items_dev = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(plan.items));
    defer items_dev.free();
    var members_dev = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(plan.members));
    defer members_dev.free();
    var count_dev = try cuda.DeviceBuffer.fromHost(d, std.mem.asBytes(&plan.count));
    defer count_dev.free();
    var eact = try cuda.DeviceBuffer.alloc(d, pairs * o.width * 2);
    defer eact.free();
    var ey = try cuda.DeviceBuffer.alloc(d, pairs * o.dims * 4);
    defer ey.free();
    var flush = try cuda.DeviceBuffer.alloc(d, 256 << 20); // ten times GB10's L2
    defer flush.free();

    const layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = up.ptr, .down = down.ptr, .up_scale = scales.ptr, .down_scale = scales.ptr, .width = o.width, .dims = o.dims, .experts = o.experts };
    const route: cuda.grouped.Plan = .{ .members = members_dev.ptr, .items = items_dev.ptr, .counts = count_dev.ptr, .rank = 0, .hist = 0 };
    const items = cuda.grouped.maxItems(pairs, E, 16);
    const skip: i32 = @intCast(o.experts);

    var variants: std.ArrayList(Variant) = .empty;
    try variants.append(a, .{ .name = "stock" });
    if (pf != null) try variants.appendSlice(a, &.{ .{ .name = "prefetch-lines", .pf = .lines }, .{ .name = "prefetch-bulk", .pf = .bulk } });
    for ([_]u8{ 2, 3, 4 }) |depth| try variants.append(a, .{ .name = try std.fmt.allocPrint(a, "ring{d}", .{depth}), .ring = depth });
    if (pf != null) try variants.append(a, .{ .name = "ring3+prefetch-bulk", .pf = .bulk, .ring = 3 });

    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var t0 = try cuda.Event.init(d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(d, true);
    defer t1.deinit();
    const out_eact = try a.alloc(u8, pairs * o.width * 2);
    const out_ey = try a.alloc(u8, pairs * o.dims * 4);
    var want: ?u64 = null;
    var failed = false;
    std.debug.print("rows={d} dims={d} width={d} experts={d} pairs={d} items={d} prefetch bytes a pick={d}\n", .{ o.rows, o.dims, o.width, o.experts, pairs, plan.count, prefetch.bytesPerPick(try prefetch.geometry(layer, o.groups)) });
    for (variants.items) |v| {
        var ring = if (v.ring) |depth| try ring_mod.Ring.load(d, ring_image, depth, sms) else try ring_mod.Ring.loadNamed(d, ring_image, stock_names, 0, sms);
        defer ring.unload();
        if (pf) |*p| p.settings.mode = v.pf orelse .bulk;
        const ms = try a.alloc(f32, o.reps);
        var hash: u64 = 0;
        for (0..o.reps) |rep| {
            try flush.fill32(@intCast(rep + 1), stream.handle);
            try eact.fill8(0, stream.handle);
            try ey.fill8(0, stream.handle);
            try t0.record(stream);
            if (v.pf != null) try pf.?.issue(stream, picks_dev.ptr, o.rows, slots, layer);
            if (spin) |f| if (o.gap > 0) {
                var sa: cuda.Args = .{};
                sa.add(@as(i64, o.gap));
                try cuda.launch.launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, stream, &sa);
            };
            const in_up: cuda.experts.Rows = .{ .x = x.ptr, .stride = o.dims, .slots = slots };
            const in_down: cuda.experts.Rows = .{ .x = eact.ptr, .stride = o.width, .slots = 0 };
            try ring.run(stream, true, in_up, layer, route, eact.ptr, items, skip);
            try ring.run(stream, false, in_down, layer, route, ey.ptr, items, skip);
            try t1.record(stream);
            if (pf) |*p| try p.join(stream);
            try stream.synchronize();
            ms[rep] = try cuda.Event.elapsedMs(t0, t1);
            try eact.download(0, out_eact);
            try ey.download(0, out_ey);
            var h = std.hash.Wyhash.init(0);
            h.update(out_eact);
            h.update(out_ey);
            const got = h.final();
            if (rep == 0) hash = got else if (got != hash) {
                std.debug.print("FAIL {s}: outputs differ between reps\n", .{v.name});
                failed = true;
            }
        }
        if (want == null) want = hash;
        const same = hash == want.?;
        if (!same) failed = true;
        const kept = ms[1..];
        std.mem.sort(f32, kept, {}, std.sort.asc(f32));
        std.debug.print("{s:<22} median_ms={d:.4} min_ms={d:.4} max_ms={d:.4} n={d} bits={s} hash={x:0>16}\n", .{ v.name, kept[kept.len / 2], kept[0], kept[kept.len - 1], kept.len, if (same) "SAME" else "DIFFER", hash });
    }
    if (failed) {
        std.debug.print("FAIL: a variant's outputs differ from the stock kernels'\n", .{});
        return 1;
    }
    return 0;
}

test "glm expert prefetch bench: the host plan groups pairs by expert in items of at most 16" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var picks: [40]i32 = undefined;
    for (&picks, 0..) |*p, i| p.* = if (i < 33) 2 else @intCast(i % 3);
    const p = try hostPlan(arena.allocator(), &picks, 4);
    // expert 0: 33,36,39 -> 3; expert 1: 34,37 -> 2; expert 2: 33 + 35,38 = 35 -> 16,16,3; expert 3: none
    try std.testing.expectEqual(@as(i32, 5), p.count);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 3, 1, 3, 2, 2, 5, 16, 2, 21, 16, 2, 37, 3 }, p.items);
    try std.testing.expectEqual(@as(usize, 40), p.members.len);
}

test "glm expert prefetch bench: synthetic scale bytes are finite e4m3 in [2^-3, 2^0]" {
    var prng = std.Random.DefaultPrng.init(7);
    var buf: [576 * 3]u8 = undefined;
    weights(prng.random(), &buf);
    for (0..3) |b| for (buf[b * 576 + 512 ..][0..64]) |s| {
        try std.testing.expect(s & 0x87 == 0); // sign 0, mantissa 0
        try std.testing.expect((s >> 3) >= 4 and (s >> 3) <= 7);
    };
}
