const std = @import("std");
const cubin = @import("cubin");

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    for ([_]u32{80,86,89,120,121}) |sm| {
        const path = try std.fmt.allocPrint(a, "/tmp/tf-allsm-exl3-cubins/n256-sm{d}.cubin", .{sm});
        defer a.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(1 << 28));
        defer a.free(bytes);
        const kernels = try cubin.kernels(a, bytes, sm);
        defer a.free(kernels);
        for (kernels) |k| {
            if (std.mem.eql(u8, k.name, "tf_exl3_group_epi") or std.mem.startsWith(u8, k.name, "tf_exl3_fused_"))
                std.debug.print("sm{d} {s} regs={d} static={d} bound={d}\n", .{ k.sm, k.name, k.usage.regs, k.usage.static, k.usage.bound });
        }
    }

}
