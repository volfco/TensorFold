//! The lossless kept-state codec (docs/kv-cache-compression-plan.md): host tests and the ReleaseFast benchmark.
const std = @import("std");

pub fn wire(b: *std.Build) void {
    const host = b.graph.host;
    const tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/kv_codec_test.zig"), .target = host, .optimize = .Debug }) });
    const step = b.step("test-kv-codec", "KV codec: rANS, block and container round trips, refusals, ledger, worker pool (CPU only)");
    step.dependOn(&b.addRunArtifact(tests).step);
    // the persistent workers and their CPU list (TF_GLM_KV_CODEC_CPUS): threads and affinity need libc
    const pool_tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/kv_codec_pool_test.zig"), .target = host, .optimize = .Debug, .link_libc = true }) });
    step.dependOn(&b.addRunArtifact(pool_tests).step);
    // cache-entry hints (tools/kvprobe/HINTS.md): the sidecar reader's refusals
    const hint_tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/kv_hints_test.zig"), .target = host, .optimize = .Debug }) });
    b.step("test-kv-hints", "KV cache-entry hint sidecars: golden format and every refusal (CPU only)").dependOn(&b.addRunArtifact(hint_tests).step);
    const compile = b.step("compile-kv-codec", "KV codec tests and benchmark compiled, nothing run");
    compile.dependOn(&tests.step);
    compile.dependOn(&pool_tests.step);
    const kv = b.createModule(.{ .root_source_file = b.path("zig/src/core/kv.zig"), .target = host, .optimize = .ReleaseFast });
    // -Dkv-bench-libs: the bench (only) links the system's libzstd and liblz4 for ClickHouse's general-purpose codecs
    const libs = b.option(bool, "kv-bench-libs", "kv-codec-bench: link system libzstd and liblz4 (ClickHouse LZ4/LZ4HC/ZSTD legs)") orelse false;
    const opts = b.addOptions();
    opts.addOption(bool, "system_codecs", libs);
    const exe = b.addExecutable(.{ .name = "tf-kv-codec-bench", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/kv_codec_bench.zig"), .target = host, .optimize = .ReleaseFast, .link_libc = true, .imports = &.{ .{ .name = "kv", .module = kv }, .{ .name = "kv_bench_options", .module = opts.createModule() } } }) });
    if (libs) {
        exe.root_module.linkSystemLibrary("zstd", .{});
        exe.root_module.linkSystemLibrary("lz4", .{});
    }
    compile.dependOn(&exe.step);
    b.step("kv-codec-bench", "Install zig-out/bin/tf-kv-codec-bench (ratio, encode/decode GB/s vs memcpy and deflate; --help)").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const run = b.addRunArtifact(exe);
    run.has_side_effects = true;
    b.step("kv-codec-bench-run", "tf-kv-codec-bench with its defaults: 64 MiB synthetic streams, up to 8 threads").dependOn(&run.step);
}
