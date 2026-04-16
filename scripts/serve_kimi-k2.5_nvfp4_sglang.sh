#!/bin/bash
# serve_kimi-k2.5_nvfp4_sglang.sh
# Status: PENDING VERIFICATION
#
# Kimi K2.5 NVFP4 served via SGLang. Architecture is DeepSeek-V3-class
# (multi-head latent attention / MLA), so KV cache per token is much
# smaller than M2.7 — high concurrency should be easier.
#
# Notes:
# - Tool/reasoning parser flags (--tool-call-parser, --reasoning-parser)
#   are NOT included. They affect output post-processing for agentic
#   workflows but are unnecessary for random-input benchmark sweeps.
#   Add them later if/when we run agentic evals.
# - --trust-remote-code is required because Kimi ships modeling_deepseek.py
#   and modeling_kimi_k25.py as custom python modules.
# - EP=1 (no --expert-parallel-size). EP+NVFP4 is broken on SGLang
#   0.5.10.post1 modelopt loader as of 2026-04-14 — see CLAUDE.md.

set -u

NAME=kimi25-sglang
IMAGE=lmsysorg/sglang:latest-cu130-runtime
MODEL=nvidia/Kimi-K2.5-NVFP4

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then
  echo "ERROR: container(s) already running: $RUNNING" >&2
  echo "Kill them first with: docker rm -f <name>" >&2
  exit 2
fi

mkdir -p ~/benchmark/results/sglang/kimi-k2.5/{json,logs}

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=8 EP=1"
docker run --gpus all --shm-size 32g --ipc=host --ulimit memlock=-1 \
  -v ~/hf_hub_cache:/root/.cache/huggingface \
  -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
  -v ~/benchmark/results:/results \
  -p 30000:30000 \
  --name "$NAME" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp 8 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --host 0.0.0.0 --port 30000
