#!/bin/bash
# serve_minimax-m2.7_nvfp4_sglang.sh
# Status: PENDING VERIFICATION
#
# SGLang serve wrapper for MiniMax M2.7 NVFP4. SGLang is being evaluated as
# a replacement for vLLM on B300 because:
#   - NVFP4 + EP works on SGLang (vLLM 0.13.0 cutlass_moe_fp4 kernel has no EP)
#   - Faster cold start (no torch.compile)
#   - Same OpenAI-compatible API — bench_sweep.sh works with PORT=30000
#
# Image: lmsysorg/sglang:latest-cu130-runtime (fallback: lmsysorg/sglang:dev-cu13)
# Port: 30000 (SGLang default). Container name: m27-sglang.
#
# Known issue: some cu13 images throw libcudart.so.12 errors. If stable fails,
# try `lmsysorg/sglang:dev-cu13`. A Triton ptxas workaround may be needed:
#   export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

set -u

NAME=m27-sglang
IMAGE=lmsysorg/sglang:latest-cu130-runtime
MODEL=lukealonso/MiniMax-M2.7-NVFP4

RUNNING=$(sudo docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then
  echo "ERROR: container(s) already running: $RUNNING" >&2
  echo "Kill them first with: sudo docker rm -f <name>" >&2
  exit 2
fi

mkdir -p ~/benchmark/results/sglang ~/benchmark/results/vllm

echo "Launching $NAME  model=$MODEL  image=$IMAGE"
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
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --host 0.0.0.0 --port 30000
