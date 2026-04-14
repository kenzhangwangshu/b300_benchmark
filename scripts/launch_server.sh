#!/bin/bash
# launch_server.sh — unified vLLM container launcher for B300 NVFP4 benchmarks
#
# Usage:
#   launch_server.sh <container_name> <model_id> [extra vllm serve args...]
#
# Example:
#   launch_server.sh m27nvfp4 lukealonso/MiniMax-M2.7-NVFP4 --quantization modelopt_fp4
#
# Encapsulates the hard rules:
#   - Only one container at a time (refuses to launch if anything is running)
#   - Standard B300 mounts (HF cache read dirs + results write dir)
#   - TP=8, max-model-len=16384, GPU mem 0.9
#   - Image nvcr.io/nvidia/vllm:26.01-py3 (driver-compatible on 590.48)
#
# Why max-model-len=16384 (not 8192): the InferenceX benchmark profiles
# include 1k8k (ISL=1024, OSL=8192) and 8k1k (ISL=8192, OSL=1024), both
# of which need ≥ 9216 tokens of context. 8192 rejects those requests with
# HTTP 400 and the sweep silently produces zero data. 16384 leaves comfortable
# headroom for all three InferenceX profiles with a single warm container,
# trading a little KV cache capacity on 1k1k for correctness everywhere.
#
# Do NOT add --enable-expert-parallel when serving NVFP4 MoE models —
# ModelOptNvFp4FusedMoE's cutlass_moe_fp4 kernel does not support EP in
# vLLM 0.13.0. Pass --quantization modelopt_fp4 in extra args for NVFP4.

set -u

if [ $# -lt 2 ]; then
  echo "usage: $0 <container_name> <model_id> [extra vllm serve args...]" >&2
  exit 1
fi

NAME="$1"
MODEL="$2"
shift 2

IMAGE="nvcr.io/nvidia/vllm:26.01-py3"

# Hard rule: only one container at a time.
RUNNING=$(sudo docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then
  echo "ERROR: container(s) already running: $RUNNING" >&2
  echo "Kill them first with: sudo docker rm -f <name>" >&2
  exit 2
fi

mkdir -p ~/benchmark/results/raw

echo "Launching $NAME  model=$MODEL  image=$IMAGE"
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 \
  --shm-size=16g \
  -v ~/hf_hub_cache:/root/.cache/huggingface \
  -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
  -v ~/benchmark/results:/results \
  -p 8000:8000 \
  --name "$NAME" \
  "$IMAGE" \
  vllm serve "$MODEL" \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  --port 8000 \
  "$@"
