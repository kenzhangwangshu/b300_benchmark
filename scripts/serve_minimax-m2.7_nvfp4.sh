#!/bin/bash
# serve_minimax-m2.7_nvfp4.sh
# Tested: 2026-04-13, NGC 26.01, B300 NVL8, driver 590.48
# Status: WORKING (server reached "Application startup complete.",
#          warmup curl returned a valid completion, vllm bench serve confirmed)
# Note: --enable-expert-parallel REMOVED — ModelOptNvFp4FusedMoE's cutlass_moe_fp4
# kernel does not support EP in vLLM 0.13.0 (26.01). Use TP-only sharding.

exec bash ~/benchmark/scripts/launch_server.sh \
  m27nvfp4 \
  lukealonso/MiniMax-M2.7-NVFP4 \
  --quantization modelopt_fp4
