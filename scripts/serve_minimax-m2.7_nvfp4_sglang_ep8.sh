#!/bin/bash
# serve_minimax-m2.7_nvfp4_sglang_ep8.sh
# Status: PENDING VERIFICATION
#
# Variant of serve_minimax-m2.7_nvfp4_sglang.sh that enables expert parallelism
# (--expert-parallel-size 8) on top of TP=8. SGLang routes MoE experts across
# ranks instead of replicating them; non-MoE layers still use TP. This is the
# exact NVFP4 + EP path that is broken on vLLM 0.13.0 (ModelOptNvFp4FusedMoE's
# cutlass_moe_fp4 kernel rejects EP). SGLang uses a newer flashinfer cutlass
# MoE allgather kernel that supports NVFP4 + EP together.
#
# Container name: m27-sglang-ep8 (distinct from the TP-only variant so we can
# compare results/sglang JSONs cleanly — tags will carry the container name
# already via the framework split.)

set -u

NAME=m27-sglang-ep8
IMAGE=lmsysorg/sglang:latest-cu130-runtime
MODEL=lukealonso/MiniMax-M2.7-NVFP4

RUNNING=$(sudo docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then
  echo "ERROR: container(s) already running: $RUNNING" >&2
  echo "Kill them first with: sudo docker rm -f <name>" >&2
  exit 2
fi

mkdir -p ~/benchmark/results/sglang

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=8 EP=8"
sudo docker run --gpus all --shm-size 32g --ipc=host --ulimit memlock=-1 \
  -v ~/hf_hub_cache:/root/.cache/huggingface \
  -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
  -v ~/benchmark/results:/results \
  -p 30000:30000 \
  --name "$NAME" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp 8 \
  --expert-parallel-size 8 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --host 0.0.0.0 --port 30000
