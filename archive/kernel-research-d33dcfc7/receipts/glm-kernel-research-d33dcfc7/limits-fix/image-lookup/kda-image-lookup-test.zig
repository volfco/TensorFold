const std = @import("std");
const registry = @import("contract").registry;
const common = [_][]const u8{ "act_rows", "attn_train", "block_attention", "ck_experts", "ck_lane", "dense_rows", "drama3", "drama3_codec", "drama3_fp8", "embed_rows", "exl3_candidate", "exl3_experts", "exl3_linear", "experts", "experts3", "experts3_prefill", "experts_rows", "flashnext_attn", "flashnext_batch", "flashnext_fact", "flashnext_fexp", "flashnext_gdn", "flashnext_hc", "flashnext_hcf", "flashnext_mtp", "flashnext_train", "fp8_experts", "gdn_prep", "gdn_train", "glm_dflash", "glm_glue", "glm_index", "glm_kda", "glm_mm", "glm_ops", "lane_gemv", "moe", "moe_glue", "nemotron_attention", "nemotron_latent", "nemotron_mamba", "nemotron_route", "norms", "nvfp4_experts", "nvfp4_experts_tile", "prefill_attention", "prompt16", "qk_rows", "qmm", "qmm_group", "qmm_prefill", "qmmf", "qwen27_train", "qwen35", "qwen35_bf16_rows", "qwen35_draft", "qwen35_keep", "qwen35_kv8", "qwen36_mtp", "qwen4_gdn", "qwen4_hc", "qwen4_ple", "qwen4_projection", "qwen_vision", "rank_sum", "route_rows", "sample", "train", "train_attn", "train_q4", "tree_attention", "tree_round", "vit", "w4a8_prefill", "window_attention", "window_pages", "window_rows" };
const deep = [_][]const u8{ "dsv41_exl3_experts", "dsv41_x3ld", "dsv41_x3pf", "dsv41_x3gm", "dsv41_linear", "dsv41_x3seg", "dsv41_dense3", "dsv41_attn", "dsv41_topk", "dsv41_mhc", "dsv41_mhc_pf", "dsv41_router_gemv", "dsv41_pfdense_k8", "dsv41_pfdense_k10", "dsv41_pfdense_k12", "dsv41_pfdense_k16", "dsv41_engram_gate", "dsv41_l2pace", "dsv41_l2pf", "dsv41_x3gm_plan", "dsv41_topk_keys", "dsv41_pointwise", "dsv41_kvsplit", "dsv41_glue", "dsv41_x3gm3", "dsv41_pfglue", "dsv41_router_glue", "dsv41_kv_glue", "dsv41_x3ld_epi", "dsv41_topk_b", "dsv41_sampling", "dsv41_lin2" };
fn known(name: []const u8, omit_fp8: bool) bool {
    if (omit_fp8 and std.mem.eql(u8,name,"fp8_experts")) return false;
    for (common ++ deep ++ [_][]const u8{"h3","contract_controls"}) |n| if (std.mem.eql(u8,name,n)) return true;
    return false;
}
test "all CUDA registry images have lookups, omitted fp8 is caught" {
    var missing: usize = 0;
    for (registry.all_cuda ++ registry.cuda_controls) |r| {
        try std.testing.expect(known(r.launch.image,false));
        if (!known(r.launch.image,true)) {
            try std.testing.expectEqualStrings("fp8_experts",r.launch.image);
            missing += 1;
        }
    }
    try std.testing.expect(missing > 0);
    try std.testing.expect(!known("unregistered-image",false));
}
