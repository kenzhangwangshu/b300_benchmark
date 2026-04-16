# FP8 vs NVFP4 Block Quantization & Tensor Parallelism Alignment

> **TL;DR** — Block-FP8 quantization (the format MiniMax, DeepSeek, and most
> FP8 checkpoints ship) tiles weights into 128×128 blocks. Tensor parallelism
> splits weight tensors along one axis, and the per-rank size **must** be
> divisible by the block side. NVFP4 uses 16-element groups, so it tolerates
> much smaller shards. This is why some models that run fine at TP=8 in NVFP4
> cannot run at TP=8 in FP8 — the math just doesn't divide.

## 1. The underlying rule

Both quantization schemes store weights in fixed-size blocks with one scale
per block. To compute anything, every rank must hold *complete* blocks;
partial blocks are not representable.

| Format | Block shape | Where it bites |
|---|---|---|
| **Block-FP8** (DeepSeek-V3 / vLLM / SGLang `fp8.py`) | `[128, 128]` | The **output** axis of gate/up/down projections must be a multiple of 128 per rank |
| **NVFP4** (ModelOpt) | `[1, 16]` — 16-element groups along one axis | Per-rank size must be a multiple of 16 (almost always true) |

Concretely, SGLang 0.5.10.post1 enforces this in
`sglang/srt/layers/quantization/fp8.py:819`:

```python
raise ValueError(
    f"The output_size of gate's and up's weight = {per_rank_out}"
    f" is not divisible by weight quantization block_n = {block_n}."
)
```

vLLM 0.13.0 has the same rule (same format, same block shape). It enforces
it implicitly during weight loading instead of raising a clean ValueError,
but the failure mode is the same.

## 2. Why M2.7 can do NVFP4 TP=8 but not FP8 TP=8

MiniMax M2.7 has:
- `hidden_size: 3072`
- `intermediate_size: 1536`  ← **this is the gate/up width per expert**
- `num_local_experts: 256`

At TP=`N`, the gate/up projection is split across ranks along the
`intermediate_size` axis. Each rank ends up with `1536 / N` elements.
**That per-rank value must be divisible by `block_n`.**

| TP | Per-rank elements | FP8 (`÷128`) | NVFP4 (`÷16`) |
|---:|---:|:---:|:---:|
| 1 | 1536 | 12 ✓ | 96 ✓ |
| 2 | 768 | 6 ✓ | 48 ✓ |
| **4** | **384** | **3 ✓** | **24 ✓** |
| **8** | **192** | **✗ 1.5** | **12 ✓** |
| 16 | 96 | ✗ 0.75 | 6 ✓ |

**M2.7's FP8 ceiling on this node is TP=4.** NVFP4 is fine at TP=8 because
the 16-element group divides 192 evenly.

This aligns with NVIDIA's own M2.7 FP8 blog post, which recommends TP=4 as
the production configuration.

## 3. Where this bites for the rest of the queue

Data pulled from each model's `config.json` on 2026-04-15:

| Model | `moe_intermediate_size` | `intermediate_size` (dense) | `shared_expert` | FP8 TP=8 | NVFP4 TP=8 |
|---|---:|---:|---:|:---:|:---:|
| **MiniMax M2.7** | 1536 | 1536 (all layers dense-style) | — | ✗ (192/128) | ✓ (192/16=12) |
| **Kimi K2.5** | 2048 | 18432 (layer 0 only, `first_k_dense_replace=1`) | — | ✓ (256/128=2) | ✓ (256/16=16) |
| **Qwen 3.5 397B-A17B** | 1024 | — (all-MoE, no dense FFN) | 1024 | ✓ (128/128=1) ← razor-thin | ✓ (128/16=8) |
| **GLM-5.1** | 2048 | 12288 | — | ✓ (256/128=2) | ✓ (256/16=16) |

**Notes:**
- **M2.7** has a peculiar structure: `intermediate_size = 1536` is BOTH the
  MoE expert dim and (as we saw in the crash) the gate/up width per expert.
  256 experts × 1536 intermediate × 2 projections is a deliberately compact
  expert design to keep active params at 10B out of 230B total.
- **Kimi K2.5** mixes: layer 0 is a full dense FFN with `intermediate_size=18432`
  (which TP=8-shards to 2304, 18 blocks — plenty of headroom), while layers
  1–60 are MoE with `moe_intermediate_size=2048`. Both are 128-aligned.
- **Qwen 3.5 397B-A17B** has NO dense FFN layers. Every FFN in every layer
  is a MoE block with 512 experts and 10 active per token. It also has a
  **shared expert** of intermediate 1024 in each layer (an always-on small
  expert that runs in parallel with the routed ones). Both the routed and
  shared experts have 1024 wide, and at TP=8 the per-rank slice is exactly
  128 — **one FP8 block per rank**. This works but has zero slack: at TP=16
  it would break (64/128 = 0.5).
- **GLM-5.1** has a clean split — dense intermediate 12288 and MoE
  intermediate 2048, both comfortably divisible by 128 at TP=8.

At TP=8:
- **M2.7** is the only model in the queue that fails FP8 block-alignment.
- **Qwen 3.5 397B-A17B** is on the edge — its per-rank slice at TP=8 is
  exactly 128 (one FP8 block per rank). At **TP=16** it would fail
  (64/128 = 0.5). That's a problem for larger nodes but fine for B300 NVL8.
- **Kimi** and **GLM-5.1** have 2× the headroom.

### Dense vs MoE dimensions

On MoE models, the dense (attention) layers use `hidden_size`/head-dim
shards which are typically 128-aligned, while the MoE expert projections
use `moe_intermediate_size`, which is set independently. The two can
disagree about which TP values are legal — e.g. GLM's dense layer at
`intermediate_size=12288` loves any TP up to 96 (12288 ÷ 128 = 96), but
its MoE path caps at TP=16 because `moe_intermediate_size=2048`. Always
check the **MoE intermediate** for MoE models; that's usually the binding
constraint.

## 4. Was NVFP4 TP=8 "broken after reboot"? — NO

The node was cycled between the previous session (2026-04-14) and this
session (2026-04-15). Before launching the FP8 retry, a concern was
raised: "maybe the SGLang image was auto-updated during the reboot and
now NVFP4 TP=8 is also broken, and we just haven't noticed."

**This did not happen.** Verified 2026-04-15:

```bash
sudo docker images | grep sglang
# lmsysorg/sglang:latest-cu130-runtime   715c46125862   47.2GB
```

The image ID `715c46125862` matches the prefix of the SHA recorded
in the previous session's `results/metadata/node_info.yaml`:
`sha256:715c461258624eae38124dcb1e1f620cbe307594fd3403ab92caf1b7017afd0f`

Same image, same SGLang version (0.5.10.post1), same everything. The
NVFP4 TP=8 path has not been retested in this session, but there is no
reason to believe it changed. When we do retest it, the expectation is
that it still works. (Worth a 2-minute smoke launch to confirm if we
have concerns.)

The FP8 TP=8 failure is **inherent to the M2.7 checkpoint's quantization
format**, not a framework-version regression.

## 5. Production deployment implication — dual-instance TP=4

When a model's TP ceiling is lower than the node's GPU count, the
cost-optimal deployment is to run **multiple replicas per node** rather
than underutilizing GPUs at low TP. For M2.7 FP8 on an 8-GPU node:

- **Two instances at TP=4**, one on GPUs 0–3 and one on GPUs 4–7, each
  bound to its own port.
- At cluster scale (96 nodes): 192 M2.7 FP8 replicas vs 96 NVFP4 replicas.
- This is NOT a "two copies of the same sweep" mistake — it's the real
  production topology NVIDIA recommends for this checkpoint.

See `configs/minimax-m2.7-fp8-dual.yaml` for the full deployment spec
and `scripts/serve_minimax-m2.7_fp8_vllm_inst{0,1}.sh` for the launch
commands.

The same logic will apply to any future model whose FP8 TP ceiling is
below 8. From the table above, that's only M2.7 in our current queue,
but it's the prototype for the pattern.

## 6. Practical checklist when bringing up a new model

Before launching a new FP8 (or any block-quantized) checkpoint:

1. `cat config.json` → look up `quantization_config.quant_method` and
   `quantization_config.weight_block_size`.
2. For MoE models, look up `moe_intermediate_size` (the binding axis).
3. Compute `moe_intermediate_size / TP`. Must be divisible by `block_n`
   (usually 128).
4. If the answer is less than 1 block per rank, reduce TP or switch
   to a non-block-quantized checkpoint (NVFP4, AWQ, GPTQ, or a
   per-tensor FP8).
5. For multi-replica deployments, plan `N = 8 / TP` instances per node
   to fully saturate the GPUs.

## 7. References

- SGLang FP8 block validation: `sglang/srt/layers/quantization/fp8.py:819`
- vLLM FP8 MoE kernel choice: `sglang/srt/layers/quantization/fp8.py:186-199`
  (Cutlass BlockScaled GroupedGemm fallback when DeepGEMM unavailable)
- NVIDIA MiniMax M2.7 FP8 blog post (recommends TP=4)
- `configs/minimax-m2.7.yaml` `constraints:` section
- `configs/minimax-m2.7-fp8-dual.yaml` (this session's dual-instance spec)
