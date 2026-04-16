#!/bin/bash
# serve_deepseek-r1_nvfp4_sglang.sh
#
# DeepSeek R1 NVFP4 served via SGLang on B300 TP=8 (full node).
#
# Model: nvidia/DeepSeek-R1-NVFP4 (395 GB on disk, 80 shards)
#   - Architecture: DeepseekV3ForCausalLM (DeepSeek V3 / R1 class, MLA)
#   - 61 layers, hidden=7168, 128 heads, kv_lora_rank=512, q_lora_rank=1536
#   - 256 routed experts + 1 shared, 8 experts/token, moe_intermediate=2048
#   - max_position_embeddings=163840 (YaRN-scaled 128K context)
#   - Custom modeling code (auto_map → modeling_deepseek.DeepseekV3Model) →
#     --trust-remote-code required
#
# Flags chosen from the superset of:
#   - NVIDIA model card: TP=8, TRT-LLM primary, recommends enable_attention_dp
#   - vLLM GB300 blog (2026-02-13): VLLM_USE_FLASHINFER_MOE_FP4=1, EP2 for prefill
#     peak 22476 TGS, TP2 for mixed 3072 TGS. Only tested 2x B300 — we go 8x.
#   - verda SGLang deploy guide: attention-backend trtllm_mla, disable-radix-cache,
#     moe-runner-backend flashinfer_trtllm, kv-cache-dtype fp8_e4m3 (their test
#     was 4x B300 single-batch, not a concurrency sweep)
#
# Notes:
# - TP=8 per project rule: benchmark the full node. Do NOT fall back to TP=4 or
#   TP=2 without explicit user approval.
# - EP=1 (no --expert-parallel-size). EP+NVFP4 is broken on SGLang 0.5.10.post1
#   modelopt loader (see CLAUDE.md known issues) — same loader path used by all
#   NVFP4 MoE models.
# - No --reasoning-parser, no --tool-call-parser (benchmark rule — see CLAUDE.md).
# - Explicit --kv-cache-dtype fp8_e4m3 matches what SGLang auto-promotes on
#   DeepSeek V3 class MLA models on this node (observed on Kimi K2.5 and GLM-5.1).
# - Explicit --attention-backend trtllm_mla matches what Kimi auto-selected;
#   making it explicit removes ambiguity and matches the verda guide.
# - Explicit --moe-runner-backend flashinfer_trtllm matches Kimi+GLM auto-select.
# - --disable-radix-cache per verda. Random-dataset benchmarks have no prefix
#   overlap so radix cache adds memory overhead with zero benefit.
# - VLLM_USE_FLASHINFER_MOE_FP4=1 is a vLLM env var. Passing it to SGLang is
#   harmless (SGLang ignores it) and matches the user's template.

set -u

NAME=deepseek-r1
IMAGE=lmsysorg/sglang:latest-cu130-runtime
MODEL=nvidia/DeepSeek-R1-NVFP4

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then
  echo "ERROR: container(s) already running: $RUNNING" >&2
  echo "Kill them first with: docker rm -f <name>" >&2
  exit 2
fi

mkdir -p ~/benchmark/results/sglang/deepseek-r1/{json,logs}

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=8 EP=1"
# TORCHINDUCTOR_COMPILE_THREADS=1 workaround: torch.inductor spawns compile
# worker subprocesses (--kind=fork --workers=32) to compile Triton kernels in
# parallel. Those subprocesses fail `triton_helpers.set_driver_to_gpu()` with
# "Could not find an active GPU backend" on R1 specifically — the fork'd
# workers don't inherit CUDA context, and R1's vocab_parallel_embedding
# get_masked_input_and_mask triggers on-demand inductor compile during CUDA
# graph capture (dynamo-decorated function, compiles regardless of
# --enable-torch-compile=False). Setting this to 1 forces in-process
# compilation, bypassing the subprocess issue. Kimi K2.5 and GLM-5.1 don't
# hit it due to flaky kernel compile ordering.
docker run --gpus all --shm-size 32g --ipc=host --ulimit memlock=-1 \
  -v ~/hf_hub_cache:/root/.cache/huggingface \
  -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
  -v ~/benchmark/results:/results \
  -p 30000:30000 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
  -e TORCHINDUCTOR_COMPILE_THREADS=1 \
  --name "$NAME" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp 8 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend trtllm_mla \
  --moe-runner-backend flashinfer_trtllm \
  --disable-radix-cache \
  --context-length 16384 \
  --host 0.0.0.0 --port 30000
