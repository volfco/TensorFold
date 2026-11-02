//! TF_GLM_EXPERT_PREFETCH and TF_NVFP4_DECODE_RING (unbuilt prototypes): their launch geometry is registered; no
//! compile has recorded their resources, so every served SM fails qualification until tf-cuda-test
//! kernel-contract-limits supplies them from built images.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");
const none = registry.none;

pub const records = [_]registry.CudaRecord{
    // loads only (cp.async.bulk.prefetch.L2 needs sm_90; the line form runs anywhere but is registered beside it)
    .{ .launch = .{ .name = "glm_expert_prefetch tf_glm_expert_prefetch_bulk", .symbol = "tf_glm_expert_prefetch_bulk", .image = "glm_expert_prefetch", .threads = 128, .min_sm = 90 }, .usage = none },
    .{ .launch = .{ .name = "glm_expert_prefetch tf_glm_expert_prefetch_lines", .symbol = "tf_glm_expert_prefetch_lines", .image = "glm_expert_prefetch", .threads = 128 }, .usage = none },
    // the stock decode's geometry (4 warps, no shared memory); bits must equal nvfp4_expert_kernel<M, EPI, 4>'s
    .{ .launch = .{ .name = "nvfp4_expert_ring tf_nvfp4_ring_gate_up_d2", .symbol = "tf_nvfp4_ring_gate_up_d2", .image = "nvfp4_expert_ring", .threads = 128 }, .usage = none },
    .{ .launch = .{ .name = "nvfp4_expert_ring tf_nvfp4_ring_gate_up_d3", .symbol = "tf_nvfp4_ring_gate_up_d3", .image = "nvfp4_expert_ring", .threads = 128 }, .usage = none },
    .{ .launch = .{ .name = "nvfp4_expert_ring tf_nvfp4_ring_gate_up_d4", .symbol = "tf_nvfp4_ring_gate_up_d4", .image = "nvfp4_expert_ring", .threads = 128 }, .usage = none },
    .{ .launch = .{ .name = "nvfp4_expert_ring tf_nvfp4_ring_down_f32_d2", .symbol = "tf_nvfp4_ring_down_f32_d2", .image = "nvfp4_expert_ring", .threads = 128 }, .usage = none },
    .{ .launch = .{ .name = "nvfp4_expert_ring tf_nvfp4_ring_down_f32_d3", .symbol = "tf_nvfp4_ring_down_f32_d3", .image = "nvfp4_expert_ring", .threads = 128 }, .usage = none },
    .{ .launch = .{ .name = "nvfp4_expert_ring tf_nvfp4_ring_down_f32_d4", .symbol = "tf_nvfp4_ring_down_f32_d4", .image = "nvfp4_expert_ring", .threads = 128 }, .usage = none },
};

/// The entries the replaced kernel's bits bind (the contract's "kernel whose bits it must equal").
pub const replaces = [_][2][]const u8{
    .{ "tf_nvfp4_ring_gate_up_d2", "nvfp4_expert_kernel<2, 2, 4>" },
    .{ "tf_nvfp4_ring_gate_up_d3", "nvfp4_expert_kernel<2, 2, 4>" },
    .{ "tf_nvfp4_ring_gate_up_d4", "nvfp4_expert_kernel<2, 2, 4>" },
    .{ "tf_nvfp4_ring_down_f32_d2", "nvfp4_expert_kernel<1, 0, 4>" },
    .{ "tf_nvfp4_ring_down_f32_d3", "nvfp4_expert_kernel<1, 0, 4>" },
    .{ "tf_nvfp4_ring_down_f32_d4", "nvfp4_expert_kernel<1, 0, 4>" },
};

test "GLM expert prefetch and ring records fail qualification on every applicable SM until compiles are recorded" {
    for (records) |record| {
        var applicable: usize = 0;
        for (limits.sms) |t| applicable += @intFromBool(t.sm >= record.launch.min_sm);
        try std.testing.expect(applicable > 0);
        try std.testing.expectEqual(applicable, try registry.checkCuda(std.testing.allocator, &.{record}, true));
    }
}

test "GLM expert ring records each name the stock kernel they must equal" {
    var ring: usize = 0;
    for (records) |r| if (std.mem.startsWith(u8, r.launch.symbol, "tf_nvfp4_ring_")) {
        ring += 1;
        var found = false;
        for (replaces) |x| found = found or std.mem.eql(u8, x[0], r.launch.symbol);
        try std.testing.expect(found);
    };
    try std.testing.expectEqual(replaces.len, ring);
}
