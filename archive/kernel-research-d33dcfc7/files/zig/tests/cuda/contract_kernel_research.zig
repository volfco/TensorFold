//! Optional CUDA kernel experiments share the production runner's gates and controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const n256 = @import("contract_exl3_n256.zig");
const epi = @import("contract_exl3_epi.zig");
const kda = @import("contract_glm_kda.zig");
const gates = core.contract.gates;

pub const Suite = struct {
    loaded: n256.Loaded,
    products: [5]n256.Product,
    fused: [6]epi.Product,
    recurrent: kda.Registrations,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(gates.Kernel)) !*Suite {
        const original_len = kernels.items.len;
        errdefer kernels.items.len = original_len;
        const self = try a.create(Suite);
        errdefer a.destroy(self);
        self.loaded = try n256.Loaded.init(gpu);
        errdefer self.loaded.deinit();
        self.products = .{
            .{ .rig = .{ .gpu = gpu, .s = s }, .loaded = &self.loaded, .sk = 4, .mats = 2 },
            .{ .rig = .{ .gpu = gpu, .s = s }, .loaded = &self.loaded, .sk = 1, .mats = 1 },
            .{ .rig = .{ .gpu = gpu, .s = s }, .loaded = &self.loaded, .sk = 4, .mats = 2, .fault = true },
            .{ .rig = .{ .gpu = gpu, .s = s }, .loaded = &self.loaded, .sk = 1, .mats = 1, .small = true },
            .{ .rig = .{ .gpu = gpu, .s = s }, .loaded = &self.loaded, .sk = 1, .mats = 1, .small = true, .fault = true },
        };
        errdefer for (&self.products) |*p| p.deinit();
        const fns = try cuda.exl3_experts.epilogues.Kernels.resolve(self.loaded.module);
        for (&self.fused, 0..) |*p, i| p.* = .{
            .product = .{ .rig = .{ .gpu = gpu, .s = s }, .loaded = &self.loaded, .sk = if (i % 3 == 1) 1 else 4, .mats = if (i % 3 == 1) 1 else 2 },
            .kernels = fns,
            .fault = i % 3 == 2,
            .tail = i >= 3,
        };
        errdefer for (&self.fused) |*p| p.deinit();
        self.recurrent = try kda.Registrations.init(gpu, s);
        errdefer self.recurrent.deinit();
        for (&self.products) |*p| try kernels.append(a, p.kernel());
        for (&self.fused) |*p| try kernels.append(a, p.kernel());
        try self.recurrent.append(a, kernels);
        return self;
    }

    pub fn destroy(self: *Suite, a: std.mem.Allocator) void {
        self.recurrent.deinit();
        for (&self.fused) |*p| p.deinit();
        for (&self.products) |*p| p.deinit();
        self.loaded.deinit();
        a.destroy(self);
    }
};
