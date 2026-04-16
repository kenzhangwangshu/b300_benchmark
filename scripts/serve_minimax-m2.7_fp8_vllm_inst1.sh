#!/bin/bash
# serve_minimax-m2.7_fp8_vllm_inst1.sh
# Instance 1 of a dual-replica deployment on this 8-GPU node.
# GPUs 4-7, port 8001, TP=4 + EP=4.
#
# SOP exception: this is ONE of TWO concurrent M2.7 FP8 containers on
# this node. See configs/minimax-m2.7-fp8-dual.yaml for the rationale.

set -u

NAME=m27-fp8-inst1
IMAGE=nvcr.io/nvidia/vllm:26.01-py3
MODEL=MiniMaxAI/MiniMax-M2.7

mkdir -p ~/benchmark/results/vllm/minimax-m2.7/{json,logs}

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=4 EP=4 GPUs=4-7 port=8001"
docker run --gpus '"device=4,5,6,7"' --ipc=host --ulimit memlock=-1 \
  --shm-size=16g \
  -v ~/hf_hub_cache:/root/.cache/huggingface \
  -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
  -v ~/benchmark/results:/results \
  -p 8001:8001 \
  --name "$NAME" \
  "$IMAGE" \
  vllm serve "$MODEL" \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --trust-remote-code \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  --port 8001
