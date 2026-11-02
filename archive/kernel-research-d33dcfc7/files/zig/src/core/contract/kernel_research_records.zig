//! CUDA 13.0 compiles of the optional EXL3 and KDA experiments, with driver-visible shared memory.
const registry = @import("registry.zig");

fn entry(comptime image: []const u8, comptime symbol: []const u8, threads: u32, shared: u32, bound: u32, regs: [5]u32) registry.CudaRecord {
    var r: registry.CudaRecord = .{
        .launch = .{ .name = image ++ " " ++ symbol, .symbol = symbol, .image = image, .threads = threads, .min_sm = 80 },
        .usage = undefined,
        .later_pending = true,
    };
    for (&r.usage, regs) |*u, n| u.* = .{ .regs = n, .static = shared, .bound = bound };
    return r;
}

fn groupEpi() registry.CudaRecord {
    var r = entry("exl3_experts", "tf_exl3_group_epi", 1024, 128, 1024, .{ 32, 42, 42, 42, 42 });
    r.launch.dynamic = 64 * 1024;
    r.launch.opt_in = true;
    return r;
}

pub const records = [_]registry.CudaRecord{
    entry("exl3_experts", "tf_exl3_grouped_256_1", 128, 49152, 128, .{ 168, 200, 200, 221, 221 }),
    entry("exl3_experts", "tf_exl3_grouped_256_small_1", 128, 24576, 128, .{ 150, 150, 150, 150, 150 }),
    groupEpi(),
    entry("exl3_experts", "tf_exl3_fused_gu_0", 128, 32848, 0, .{ 167, 128, 128, 127, 127 }),
    entry("exl3_experts", "tf_exl3_fused_gu_1", 128, 32848, 0, .{ 163, 130, 130, 128, 128 }),
    entry("exl3_experts", "tf_exl3_fused_gu_2", 128, 32848, 0, .{ 157, 127, 127, 128, 128 }),
    entry("exl3_experts", "tf_exl3_fused_down_0", 128, 32832, 0, .{ 161, 128, 128, 128, 128 }),
    entry("exl3_experts", "tf_exl3_fused_down_1", 128, 32832, 0, .{ 157, 130, 130, 127, 127 }),
    entry("exl3_experts", "tf_exl3_fused_down_2", 128, 32832, 0, .{ 159, 128, 128, 128, 128 }),
    entry("glm_kda", "tf_glm_kda_chain", 1024, 2568, 1024, .{ 55, 57, 57, 59, 59 }),
    entry("glm_kda", "tf_glm_kda_replay", 1024, 512, 1024, .{ 52, 53, 53, 55, 57 }),
    entry("glm_kda", "tf_glm_kda_replay_layers", 1024, 512, 1024, .{ 56, 62, 62, 56, 62 }),
    entry("glm_kda", "tf_glm_kda_prep", 512, 1024, 512, .{ 32, 32, 32, 36, 36 }),
    entry("glm_kda", "tf_glm_kda_step", 256, 26688, 256, .{ 60, 79, 79, 68, 68 }),
    entry("glm_kda", "tf_glm_kda_step_vector8", 256, 13344, 256, .{ 64, 64, 64, 58, 58 }),
    entry("glm_kda", "tf_glm_kda_step_vector16", 256, 26688, 256, .{ 48, 80, 80, 80, 80 }),
    entry("glm_kda", "tf_glm_kda_out", 128, 4, 128, .{ 22, 22, 22, 26, 26 }),
};
