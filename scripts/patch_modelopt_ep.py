#!/usr/bin/env python3
"""Patch modelopt_quant.py to fix EP+NVFP4 shape mismatch in the deep_gemm path.

Bug: process_weights_after_loading() else-branch computes w13_input_scale as
shape [num_experts] (256) but w13_weight_scale_2 is EP-sharded to
[num_local_experts] (32 for EP=8). The element-wise multiply at g1_alphas fails.

Fix: After computing w13_input_scale in the else-branch, EP-slice it to match
the local expert count, same as the cutedsl branch does with _slice_scale().
Also slice w2_input_scale for g2_alphas consistency.
"""
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py"

with open(TARGET) as f:
    content = f.read()

# The else branch that handles deep_gemm (and other non-flashinfer runners)
OLD = """\
        else:
            w13_input_scale = layer.w13_input_scale.max(dim=-1).values.to(torch.float32)
            w2_input_scale = layer.w2_input_scale"""

NEW = """\
        else:
            w13_input_scale = layer.w13_input_scale.max(dim=-1).values.to(torch.float32)
            w2_input_scale = layer.w2_input_scale
            # EP fix: slice input scales to local experts if EP is active
            if hasattr(layer, 'moe_ep_size') and layer.moe_ep_size > 1:
                def _ep_slice(w, layer=layer):
                    if w.dim() == 0:
                        return w
                    n = layer.num_local_experts
                    r = layer.moe_ep_rank
                    return w[r * n : (r + 1) * n]
                w13_input_scale = _ep_slice(w13_input_scale)
                w2_input_scale = _ep_slice(w2_input_scale)"""

if OLD not in content:
    print("ERROR: Could not find the target code block. File may have changed.", file=sys.stderr)
    sys.exit(1)

content = content.replace(OLD, NEW, 1)

with open(TARGET, 'w') as f:
    f.write(content)

print("Patched modelopt_quant.py: added EP-slice for deep_gemm path input scales")
