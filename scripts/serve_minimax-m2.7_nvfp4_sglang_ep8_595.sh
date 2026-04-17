#!/bin/bash
# serve_minimax-m2.7_nvfp4_sglang_ep8_595.sh
#
# MiniMax M2.7 NVFP4 with EP+NVFP4 enabled (TP=8 EP=8) on driver 595.58.03.
# Uses DeepEP all-to-all backend for expert-parallel communication.
# Runner must be deep_gemm — only backend with deepep permute registrations.
# flashinfer_trtllm does NOT have deepep fused funcs (NotImplementedError).
#
# Model: lukealonso/MiniMax-M2.7-NVFP4 (126 GB, 256 routed experts, 0 shared)
#   - 256 ÷ 8 = 32 experts per rank. EP=8 divisibility OK.
#   - NOT MLA — no --attention-backend trtllm_mla, no --kv-cache-dtype fp8_e4m3.
#
# Results go to ~/benchmark/results_595/sglang/minimax-m2.7/{json,logs}/

set -u

NAME=m27-ep8
IMAGE=lmsysorg/sglang:latest-cu130-runtime
MODEL=lukealonso/MiniMax-M2.7-NVFP4

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then
  echo "ERROR: container(s) already running: $RUNNING" >&2
  echo "Kill them first with: docker rm -f <name>" >&2
  exit 2
fi

mkdir -p ~/benchmark/results_595/sglang/minimax-m2.7/{json,logs}

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=8 EP=8 (DeepEP)"
docker run --gpus all --shm-size 32g --ipc=host --ulimit memlock=-1 \
  -v ~/hf_hub_cache:/root/.cache/huggingface \
  -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
  -v ~/benchmark/results_595:/results \
  -p 30000:30000 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
  -e TORCHINDUCTOR_COMPILE_THREADS=1 \
  -e HF_HUB_OFFLINE=1 \
  --name "$NAME" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp 8 \
  --ep 8 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --deepep-mode auto \
  --disable-radix-cache \
  --context-length 16384 \
  --host 0.0.0.0 --port 30000
