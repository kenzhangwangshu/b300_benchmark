# EP+NVFP4 Results Tracker (Driver 595.58.03)

**Node**: 8× B300 SXM6 AC, driver 595.58.03, CUDA 13.2.
**Framework**: SGLang (multiple images tested).
**Results dir**: `~/benchmark/results_595/sglang/<model>/`

## EP+NVFP4 Status: BLOCKED on ALL current SGLang images

**Confirmed 2026-04-16**: EP+NVFP4 is blocked due to two independent bugs in SGLang that together form a dead end:

1. **`flashinfer_trtllm` runner** has NO deepep fused funcs registered → `NotImplementedError` at scheduler init for any model.
2. **`deep_gemm` runner** HAS deepep fused funcs but its forward method is deprecated → `AssertionError: forward_deepgemm_contiguous is deprecated` at first inference request.
3. **modelopt weight loader** has an EP shape mismatch bug (`w13_input_scale[256] × w13_weight_scale_2[32]`) — we wrote a patch (`scripts/patch_modelopt_ep.py`) that fixes this, but hits bugs #1 or #2 after.

No other runner backends (`flashinfer_cutedsl`, `triton`, `triton_kernel`, `cutlass`) have deepep permute registrations.

### Full attempt chain on M2.7 (2026-04-16 04:15–05:43 UTC)

| # | Runner | A2A | Image | deepep-mode | Result |
|---|---|---|---|---|---|
| 1 | `flashinfer_trtllm` | `deepep` | latest-cu130-runtime | auto | **NotImplementedError**: no deepep fused func |
| 2 | `deep_gemm` | `deepep` | latest-cu130-runtime (unpatched) | auto | **modelopt shape mismatch**: w13_input_scale[256] vs w13_weight_scale_2[32] |
| 3 | `deep_gemm` | `deepep` | latest-cu130-runtime (unpatched, +HF_HUB_OFFLINE) | auto | **Same shape mismatch** (confirmed driver-independent) |
| 4 | `deep_gemm` | `deepep` | dev (CUDA 12.9, patched) | auto | **PTXAS error**: Internal Triton PTX codegen error (CUDA 12.9 incompatible with driver 595/B300) |
| 5 | `deep_gemm` | `deepep` | latest-cu130-runtime (patched) | auto | **DeepEP internode_ll.cu:391**: "Unsupported hidden" (low-latency kernel rejects M2.7's intermediate dim 1536) |
| 6 | `deep_gemm` | `deepep` | latest-cu130-runtime (patched) | **normal** | **AssertionError**: `forward_deepgemm_contiguous is deprecated` |
| 7 | `deep_gemm` | `deepep` | **dev-cu13** (CUDA 13.0, patched) | **normal** | **Same deprecated assert** (dev code deprecates deep_gemm forward) |

### R1 attempt (2026-04-16 05:43)

| # | Runner | A2A | Image | Result |
|---|---|---|---|---|
| 8 | `flashinfer_trtllm` | `deepep` | dev-cu13 (patched) | **NotImplementedError**: no deepep fused func (same as M2.7) |

**Conclusion**: The bug is NOT model-specific. All models hit the same two dead ends.

### What would fix this (upstream SGLang)

Either:
- Register deepep fused funcs for `flashinfer_trtllm` runner (the natural path — trtllm is the B300 production runner)
- Un-deprecate `deep_gemm` forward, or provide a replacement that works with deepep
- Create a new runner backend that combines deepep permutes with a non-deprecated forward

### Our patch (scripts/patch_modelopt_ep.py)

Fixes the modelopt weight loader EP shape mismatch by EP-slicing `w13_input_scale` in the `deep_gemm` else-branch of `process_weights_after_loading()`. Verified working — weights load successfully on all images. The patch is necessary but NOT sufficient; the runner-level bugs (#1 or #2) still block inference.

## Fallback: TP=8 EP=1 on driver 595

Running all 5 models with TP=8 EP=1 on `sglang:dev-cu13` (CUDA 13.0 + latest SGLang dev code) to measure driver 595 vs 590.48 throughput delta.

Order: R1 → Qwen → GLM → Kimi → M2.7
Results: `~/benchmark/results_595/sglang/<model>/{json,logs}/`
