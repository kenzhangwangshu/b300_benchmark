#!/bin/bash
# serve_qwen3.5-397b_nvfp4_sglang.sh
#
# Qwen 3.5 397B-A17B NVFP4 served via SGLang on B300 TP=8 (full node).
#
# Model: nvidia/Qwen3.5-397B-A17B-NVFP4 (234 GB on disk, 11 shards)
#   - Architecture: Qwen3_5MoeForConditionalGeneration (multimodal class,
#     benchmarked as text-only)
#   - model_type: qwen3_5_moe   ← NEW type in SGLang; support version unverified
#   - 60 layers: 45 linear_attention + 15 full_attention (hybrid Mamba-style,
#     full_attention_interval=4)
#   - 512 experts, 10 experts/token, moe_intermediate_size=1024
#   - num_attention_heads=32, num_key_value_heads=2 (aggressive GQA),
#     head_dim=256, hidden_size=4096
#   - attn_output_gate=True, max_position_embeddings=262144, vocab=248320
#   - Total 397B / 17B active per nvidia model name
#
# Flags:
# - TP=8 per project rule (full node).
# - EP=1 because EP+NVFP4 is broken on SGLang 0.5.10.post1 modelopt loader.
# - --quantization modelopt_fp4: NVFP4 weights loader.
# - --trust-remote-code: safer for a new architecture class.
# - --kv-cache-dtype NOT set explicitly — Qwen 3.5 isn't MLA so the auto path
#   is less surprising than on Kimi/GLM/R1. Let SGLang pick.
# - --attention-backend NOT set explicitly — hybrid linear+full attention is
#   new; forcing trtllm_mla would be wrong (that's MLA-specific), and forcing
#   flashinfer may not understand the linear_attention layers. Let SGLang auto
#   pick. Expect --linear-attn-backend triton (default) for the linear layers.
# - --moe-runner-backend flashinfer_trtllm: matches what SGLang auto-selects
#   on B300 for other models; explicit keeps the config auditable.
# - --disable-radix-cache: per benchmark rule (random dataset, no prefix reuse).
# - --context-length 16384: standard benchmark cap (model native 256K).
# - NO --reasoning-parser, NO --tool-call-parser (benchmark rule, CLAUDE.md).
# - -e TORCHINDUCTOR_COMPILE_THREADS=1: preventative — not strictly required
#   (Qwen isn't DeepSeek V3 class, so the specific vocab_parallel_embedding
#   compile path we hit on R1 may not trigger) but harmless and future-proofs
#   us against any similar inductor subprocess issue on a new arch.

set -u

NAME=qwen35-397b
IMAGE=lmsysorg/sglang:latest-cu130-runtime
MODEL=nvidia/Qwen3.5-397B-A17B-NVFP4

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then
  echo "ERROR: container(s) already running: $RUNNING" >&2
  echo "Kill them first with: docker rm -f <name>" >&2
  exit 2
fi

# bench_sweep.sh derives MODEL_SHORT=qwen3.5-397b-a17b (strips -NVFP4 suffix,
# retains -A17B), so results live under that path. Keep this in sync.
mkdir -p ~/benchmark/results/sglang/qwen3.5-397b-a17b/{json,logs}

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=8 EP=1"
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
  --moe-runner-backend flashinfer_trtllm \
  --disable-radix-cache \
  --context-length 16384 \
  --host 0.0.0.0 --port 30000
