#!/bin/bash
# serve_minimax-m2.7_fp8_sglang.sh
# Status: PENDING VERIFICATION
#
# MiniMax M2.7 FP8 served via SGLang. Goal: direct FP4 vs FP8 comparison on
# the same model + same hardware, to confirm whether B300's NVFP4 tensor
# cores actually pay off vs the previous-gen FP8 path.
#
# Checkpoint format (from config.json):
#   quant_method:       fp8                       # vLLM-style block-FP8,
#                                                   not modelopt_fp8
#   fmt:                float8_e4m3fn
#   activation_scheme:  dynamic
#   weight_block_size:  [128, 128]                # DeepSeek-V3 / vLLM standard
#   modules_to_not_convert: [gate, e_score_correction_bias, lm_head]
# SGLang supports this natively with --quantization fp8.
#
# Mount trick: the FP8 checkpoint lives under ~/hf_hub_cache/ (non-standard
# HF cache location from an earlier pre-SOP run), while our normal NVFP4
# cache is under ~/.cache/huggingface/hub/. We mount the NVFP4 hub dir as
# the base and then overlay the FP8 model's dir into the correct sub-path
# inside the container. Docker allows this nested bind mount.

set -u

NAME=m27fp8-sglang
IMAGE=lmsysorg/sglang:latest-cu130-runtime
MODEL=MiniMaxAI/MiniMax-M2.7

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then
  echo "ERROR: container(s) already running: $RUNNING" >&2
  echo "Kill them first with: docker rm -f <name>" >&2
  exit 2
fi

mkdir -p ~/benchmark/results/sglang/minimax-m2.7/{json,logs}

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=8 EP=1 precision=fp8"
docker run --gpus all --shm-size 32g --ipc=host --ulimit memlock=-1 \
  -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
  -v ~/hf_hub_cache/models--MiniMaxAI--MiniMax-M2.7:/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.7 \
  -v ~/benchmark/results:/results \
  -p 30000:30000 \
  --name "$NAME" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp 8 \
  --trust-remote-code \
  --quantization fp8 \
  --host 0.0.0.0 --port 30000
