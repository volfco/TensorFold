The post-campaign live launch audit stopped with UnknownContractImage fp8_experts. contract_limits.zig now resolves fp8_experts and checks all actual CUDA registrations, including controls, against the resolver table at compile time. The common resolver and its image names share one table; DeepSeek and H3 retain their existing embedded images. Unknown names still refuse.

Host verification used Zig 0.17.0 from /tmp/tensorfold-night-tools/zig-x86_64-linux-0.17.0/zig and no GPU work:

- compile-cuda-tests + test-kernel-contract: exit 0.
- compile-cuda-tests with -Dfatbins=/tmp/kda-inert-fatbins and -Dcontract-controls=/tmp/kda-inert-fatbins/contract_controls.fatbin: exit 0. This forces image-enabled live contract paths to be semantically analyzed; inert files are never executed.
- Standalone host audit of the actual registry and copied resolver names: exit 0. Its negative control deliberately omits fp8_experts and finds registered missing lookups; an unknown name remains absent.

A full live GPU rerun belongs to the parent campaign. These host checks make no GPU qualification claim. Empty successful build logs are paired with explicit shell exit files.
