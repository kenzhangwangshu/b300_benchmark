#!/bin/bash
# serve_kimi-k2.5_nvfp4.sh
# Status: PENDING VERIFICATION
# Architecture: Kimi K2.5 uses the DeepSeek-V3-class transformer block
# (modeling_deepseek.py / modeling_kimi_k25.py) loaded via --trust-remote-code.

exec bash ~/benchmark/scripts/launch_server.sh \
  kimi25nvfp4 \
  nvidia/Kimi-K2.5-NVFP4 \
  --quantization modelopt_fp4 \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2
