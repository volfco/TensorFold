//! Synthetic kept-state byte classes for the codec's tests and benchmark. They stand in for captured states until
//! TF_GLM_KV_DUMP captures exist; no compression figure from them is a claim about a real checkpoint.
const std = @import("std");

pub const Kind = enum {
    random, // incompressible: the raw-fallback worst case
    zeros, // untouched capacity
    f32_from_bf16, // FP32 words widened from BF16 (low 16 bits zero): the exact-packing case
    f32_gauss, // full-mantissa FP32 (a recurrent state accumulated in FP32)
    bf16_gauss, // BF16 activations (MLA latent rows)
    bf16_rows, // BF16 rows with a per-row magnitude (latent rows of differing norms, pooled keys)
    e4m3, // FP8 E4M3 codes (the fp8 latent format)
    state_mix, // a GLM-shaped mix: 10 % FP32 state, 70 % BF16 latent rows, 15 % rings, 5 % pooled keys
};

fn bf16(x: f32) u16 {
    const w: u32 = @bitCast(x);
    return @truncate((w + 0x7FFF + ((w >> 16) & 1)) >> 16); // round to nearest even
}

fn e4m3(x: f32) u8 {
    const a = @abs(x);
    const sign: u8 = if (x < 0) 0x80 else 0;
    if (a < 1.0 / 512.0) return sign;
    const e = std.math.clamp(@as(i32, @intFromFloat(@floor(@log2(a)))), -6, 8);
    const m: i32 = std.math.clamp(@as(i32, @intFromFloat(@round((a / std.math.pow(f32, 2, @floatFromInt(e)) - 1) * 8))), 0, 7);
    return sign | @as(u8, @intCast((e + 7) << 3)) | @as(u8, @intCast(m));
}

pub fn fill(kind: Kind, out: []u8, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    switch (kind) {
        .random => r.bytes(out),
        .zeros => @memset(out, 0),
        .f32_from_bf16, .f32_gauss => {
            var i: usize = 0;
            while (i + 4 <= out.len) : (i += 4) {
                const x: f32 = @floatCast(r.floatNorm(f64) * 0.05);
                const w: u32 = if (kind == .f32_from_bf16) @as(u32, bf16(x)) << 16 else @bitCast(x);
                std.mem.writeInt(u32, out[i..][0..4], w, .little);
            }
            @memset(out[i..], 0);
        },
        .bf16_gauss, .bf16_rows => {
            var i: usize = 0;
            var scale: f32 = 1;
            while (i + 2 <= out.len) : (i += 2) {
                if (kind == .bf16_rows and i % 1024 == 0) scale = @floatCast(@exp(r.floatNorm(f64) * 1.5));
                std.mem.writeInt(u16, out[i..][0..2], bf16(@as(f32, @floatCast(r.floatNorm(f64))) * scale), .little);
            }
            @memset(out[i..], 0);
        },
        .e4m3 => for (out) |*b| {
            b.* = e4m3(@floatCast(r.floatNorm(f64) * 4));
        },
        .state_mix => {
            const cuts = [_]struct { Kind, usize }{ .{ .f32_gauss, 10 }, .{ .bf16_rows, 70 }, .{ .bf16_gauss, 15 }, .{ .bf16_gauss, 5 } };
            var at: usize = 0;
            for (cuts, 0..) |c, k| {
                const n = if (k + 1 == cuts.len) out.len - at else (out.len * c[1] / 100) & ~@as(usize, 3);
                fill(c[0], out[at..][0..n], seed +% k +% 1);
                at += n;
            }
        },
    }
}
