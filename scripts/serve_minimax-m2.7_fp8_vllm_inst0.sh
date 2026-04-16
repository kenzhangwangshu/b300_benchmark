#!/bin/bash
# serve_minimax-m2.7_fp8_vllm_inst0.sh
# Instance 0 of a dual-replica deployment on this 8-GPU node.
# GPUs 0-3, port 8000, TP=4 + EP=4.
#
# SOP exception: this is ONE of TWO concurrent M2.7 FP8 containers on
# this node. See configs/minimax-m2.7-fp8-dual.yaml for the rationale
# (TP=4 is architecturally required; dual instances fill the other 4
# GPUs to get full node utilization).

set -u

NAME=m27-fp8-inst0
IMAGE=nvcr.io/nvidia/vllm:26.01-py3
MODEL=MiniMaxAI/MiniMax-M2.7

# Note: intentionally NOT running the single-container safety check
# because the dual-instance layout launches two. The check lives in
# launch_server.sh (which we're not using for this one-off).

mkdir -p ~/benchmark/results/vllm/minimax-m2.7/{json,logs}

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=4 EP=4 GPUs=0-3 port=8000"
docker run --gpus '"device=0,1,2,3"' --ipc=host --ulimit memlock=-1 \
  --shm-size=16g \
  -v ~/hf_hub_cache:/root/.cache/huggingface \
  -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
  -v ~/benchmark/results:/results \
  -p 8000:8000 \
  --name "$NAME" \
  "$IMAGE" \
  vllm serve "$MODEL" \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --trust-remote-code \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  --port 8000
