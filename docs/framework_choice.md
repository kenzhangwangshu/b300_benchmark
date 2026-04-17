# Framework Choice: SGLang for B300 EP+NVFP4 Benchmarks

## Decision

**SGLang** is the primary (and only) framework for EP+NVFP4 benchmarks on B300.

## Reasons

1. **Official recommendation.** SGLang is recommended for DeepSeek/MoE models by
   both the DeepSeek team and NVIDIA. LMSYS (SGLang maintainers) published
   official B300 EP+NVFP4 configurations we can use directly.

2. **Native DeepEP integration.** SGLang has first-class `--moe-a2a-backend deepep`
   support with both `normal` and `low_latency` modes. DeepEP is purpose-built
   for expert-parallel all-to-all communication on NVLink-connected GPUs.

3. **Blackwell-specific MoE kernels.** SGLang ships CuTe DSL NVFP4 kernels
   (`flashinfer_cutedsl` runner backend) tuned for B300/B200 SM 103a. The
   `flashinfer_trtllm` backend is the proven production path on B300 — it
   auto-selects when CUTLASS is disabled due to TMA descriptor init issues
   on Blackwell.

4. **Proven B300 EP track record.** LMSYS blog posts document successful EP
   deployments on GB200/GB300 with NVFP4 quantization:
   - https://www.lmsys.org/blog/2026-02-19-gb300-longctx/
   - https://www.lmsys.org/blog/2025-09-25-gb200-part-2/

5. **vLLM EP is experimental.** vLLM's expert parallelism with NVFP4 is still
   marked experimental, and on driver 590.48 the `cutlass_moe_fp4` kernel
   rejected `expert_map` entirely. vLLM's `--all2all-backend` flag offers
   `deepep_high_throughput` and `deepep_low_latency` but the NVFP4+EP
   code path is less mature than SGLang's.

## EP Launch Pattern (LMSYS official for Blackwell)

```bash
python3 -m sglang.launch_server \
  --model-path <MODEL> \
  --tp 8 --ep 8 \
  --moe-a2a-backend deepep \
  --moe-runner-backend flashinfer_trtllm \
  --deepep-mode auto \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend trtllm_mla \
  --trust-remote-code \
  --disable-radix-cache \
  --context-length 16384
```

**Constraints:**
- `tp_size` MUST equal `ep_size` for DeepEP (all ranks participate in both TP and EP).
- `(num_routed_experts)` must be divisible by `ep_size=8`. All 5 models pass.
- `--attention-backend trtllm_mla` is for DeepSeek-V3-class MLA models (Kimi, GLM, R1).
  Non-MLA models (M2.7, Qwen) omit this or use `auto`.
- `--kv-cache-dtype fp8_e4m3` is for MLA models that auto-promote. Non-MLA models
  can omit (let SGLang auto-pick).
- NO `--reasoning-parser` or `--tool-call-parser` (benchmark rule, CLAUDE.md).

## SGLang Flag Reference (from 0.5.10.post1 --help)

| Flag | Values | Purpose |
|---|---|---|
| `--expert-parallel-size N` (aliases: `--ep-size`, `--ep`) | int | Expert parallel world size |
| `--moe-a2a-backend` | `none`, `deepep`, `mooncake`, `nixl`, `mori`, `ascend_fuseep`, `flashinfer` | All-to-all communication backend |
| `--moe-runner-backend` | `auto`, `deep_gemm`, `triton`, `flashinfer_trtllm`, `flashinfer_cutedsl`, `cutlass`, etc. | MoE computation kernel |
| `--deepep-mode` | `normal`, `low_latency`, `auto` | DeepEP dispatch mode |
| `--deepep-config` | JSON string | Custom DeepEP config |

## Cross-references

- vLLM B300 blog (for comparison): https://blog.vllm.ai/2026/02/13/gb300-deepseek.html
- SGLang EP docs: https://docs.sglang.io/advanced_features/expert_parallelism.html
