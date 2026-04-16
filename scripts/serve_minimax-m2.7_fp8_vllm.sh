#!/bin/bash
# serve_minimax-m2.7_fp8_vllm.sh
# Status: PENDING VERIFICATION
#
# MiniMax M2.7 FP8 served via vLLM NGC 26.01 with TP=8 + EP=8.
#
# Rationale: SGLang FP8 CANNOT run this model at TP=8 because the FP8
# quantization path requires per-rank output dim divisible by 128, and
# M2.7's MoE intermediate 1536/TP=8 = 192 breaks that constraint. vLLM's
# FP8 loader uses the same block-FP8 format but WITH expert parallelism
# the per-rank dim constraint is relaxed (experts are sharded across
# ranks rather than the intermediate dim being sliced), so EP=8 makes
# the load succeed. Also: FP8+EP works on vLLM (unlike NVFP4+EP which
# is broken in the cutlass_moe_fp4 kernel dispatcher).
#
# This gives us TP=8 FP8 data that's directly comparable (same hardware
# utilization) to the existing TP=8 NVFP4 SGLang data. Not a perfect
# controlled experiment (vLLM vs SGLang framework + EP vs no-EP both
# differ) but the closest honest FP4-vs-FP8 data we can get on this
# checkpoint.

set -u

NAME=m27-fp8
IMAGE=nvcr.io/nvidia/vllm:26.01-py3
MODEL=MiniMaxAI/MiniMax-M2.7

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then
  echo "ERROR: container(s) already running: $RUNNING" >&2
  echo "Kill them first with: docker rm -f <name>" >&2
  exit 2
fi

mkdir -p ~/benchmark/results/vllm/minimax-m2.7/{json,logs}

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=8 EP=8 precision=fp8"
docker run --gpus all --ipc=host --ulimit memlock=-1 \
  --shm-size=16g \
  -v ~/hf_hub_cache:/root/.cache/huggingface \
  -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
  -v ~/benchmark/results:/results \
  -p 8000:8000 \
  --name "$NAME" \
  "$IMAGE" \
  vllm serve "$MODEL" \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --trust-remote-code \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  --port 8000
