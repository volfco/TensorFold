import json, sys, numpy as np
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "/dev/null", "/dev/null"]
src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "analyze.py")).read().split("pairs = [")[0]      # reuse the loaders
exec(src)
for layer_slot in (22, 44):
    x = bf16_to_f32(taps_of("A+S", layer_slot)); y = bf16_to_f32(taps_of("B+S", layer_slot))
    d = rel_l2(x, y)
    print(f"layer {layers[layer_slot]}: rows with rel L2 > 0.5: {(d > 0.5).mean()*100:.1f}%, > 0.3: {(d > 0.3).mean()*100:.1f}%, > 0.2: {(d > 0.2).mean()*100:.1f}%; "
          f"best 20% of rows p50 {np.quantile(d, 0.2):.3f}, worst 20% start {np.quantile(d, 0.8):.3f}")
