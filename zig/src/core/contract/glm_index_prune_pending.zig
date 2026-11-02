//! TF_GLM_INDEX_PRUNE's glm_index launches: geometry registered, unmeasured (no nvcc yet): every SM fails until then.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");
const none = registry.none;

pub const records = [_]registry.CudaRecord{
    // tf_glm_index_scores' geometry and its 32 KiB of key tiles, plus a CTA's 32-tile need flags
    .{ .launch = .{ .name = "glm_index tf_glm_index_scores_prune", .symbol = "tf_glm_index_scores_prune", .image = "glm_index", .threads = 256, .min_sm = 80 }, .usage = none },
    // a thread a key dimension, a block a completed page
    .{ .launch = .{ .name = "glm_index tf_glm_prune_bounds", .symbol = "tf_glm_prune_bounds", .image = "glm_index", .threads = 128, .min_sm = 80 }, .usage = none },
    // a block a query row: the row's 32 heads (16.5 KiB), eight warps' strips, 1,024 pages' bounds and plan
    .{ .launch = .{ .name = "glm_index tf_glm_prune_plan", .symbol = "tf_glm_prune_plan", .image = "glm_index", .threads = 256, .min_sm = 80 }, .usage = none },
    .{ .launch = .{ .name = "glm_index tf_glm_prune_cut", .symbol = "tf_glm_prune_cut", .image = "glm_index", .threads = 256, .min_sm = 80 }, .usage = none },
};

/// The kernel whose bits the planned scores keep on every page they score (a plan of 0s: every byte).
pub const replaces = [_][2][]const u8{.{ "tf_glm_index_scores_prune", "tf_glm_scores" }};

test "GLM index: TF_GLM_INDEX_PRUNE's launches fail qualification on every served SM until compiles are recorded" {
    for (records) |record| {
        var applicable: usize = 0;
        for (limits.sms) |t| applicable += @intFromBool(t.sm >= record.launch.min_sm);
        try std.testing.expect(applicable > 0);
        try std.testing.expectEqual(applicable, try registry.checkCuda(std.testing.allocator, &.{record}, true));
    }
    try std.testing.expectEqualStrings(records[0].launch.symbol, replaces[0][0]);
}
