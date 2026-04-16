#!/bin/bash
# serve_glm-5.1_nvfp4_sglang.sh
#
# GLM-5.1 NVFP4 served via SGLang on B300 TP=8.
#
# GLM-5.1 is a post-training upgrade of GLM-5 focused on agentic coding:
#   - 744B total params / 40B active / 256 experts
#   - Same architecture as GLM-5 (serving flags should match GLM-5)
#   - License: MIT
#
# Flags:
# - --trust-remote-code: required because GLM ships custom modeling code
# - --quantization modelopt_fp4: NVFP4 weights loader
# - **NO --tool-call-parser and NO --reasoning-parser** — SGLang's glm45
#   reasoning parser buffers <think>...</think> content before streaming,
#   which inflates TTFT by the entire reasoning-phase duration (observed
#   3.9 s TTFT at conc=1 on the first GLM-5.1 run; parser-buffered
#   reasoning was ~430 tokens). Per project rule, no parsers in benchmark
#   sweeps — agentic parsers belong in production, not throughput tests.
#   See CLAUDE.md.
# - EP=1 (no --expert-parallel-size). EP+NVFP4 is broken on SGLang 0.5.10.post1
#   modelopt loader as of 2026-04-14 — see CLAUDE.md.

set -u

NAME=glm51
IMAGE=lmsysorg/sglang:latest-cu130-runtime
MODEL=lukealonso/GLM-5.1-NVFP4

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then
  echo "ERROR: container(s) already running: $RUNNING" >&2
  echo "Kill them first with: docker rm -f <name>" >&2
  exit 2
fi

mkdir -p ~/benchmark/results/sglang/glm-5.1/{json,logs}

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
