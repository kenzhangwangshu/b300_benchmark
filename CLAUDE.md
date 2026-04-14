# B300 Benchmark Project — Claude Code Instructions

## Rules (always follow)
- Only 1 Docker container at a time. Kill before starting new.
- Never Ctrl+C a running server. Use `docker rm -f` from another session to stop it. Use `kill <pid>` (not Ctrl+C) for bench loops and other non-server processes.
- Warmup once per container launch, not between concurrency levels. The per-container curl flushes CUDA graphs / JIT for all subsequent sweep levels against the same container.
- Use tmux sessions: `server`, `bench`, `download`, `claudecode` only. No extra sessions or panes. Do not cross-pollute (e.g. no bench probes in `download`).
- Save results under `~/benchmark/results/<framework>/<model_short>/{json,logs}/`. `bench_sweep.sh` derives `model_short` automatically (lowercased basename with `-NVFP4`/`-FP8` stripped — e.g. `lukealonso/MiniMax-M2.7-NVFP4` → `minimax-m2.7`). Override with `MODEL_SHORT=...` env var.
- Do not touch the `download` session unless explicitly told to — active downloads live there.
- Queue downloads one at a time (network is shared with the bench node).
- When making important changes, update CLAUDE.md, SOP.md, STATUS.md, and `configs/*.yaml` immediately. Persistent memory in `~/.claude/projects/-home-howell/memory/` holds cross-session gotchas that are too deep for a rulebook line.

## Node
- Hardware: 8× NVIDIA B300 SXM6 AC, 288 GB HBM3e each, SM 103a (detected as B200-class by some tooling)
- Driver: 590.48.01, CUDA runtime 13.1
- Host: Intel Xeon 6776P, 2× 64-core × 2 threads, 2.0 TiB RAM
- OS: Ubuntu 22.04.5 LTS, kernel 5.15.0-119-generic, Python 3.10.12, user `howell`
- Full detail: `~/benchmark/results/metadata/node_info.yaml` (canonical source — this section is a summary only)
- Driver 590.48 limits: **NGC 26.03 crashes at high concurrency** (needs 595+). Use NGC 26.01 for vLLM until driver upgrade.

## Framework
- **Primary: SGLang** — `lmsysorg/sglang:latest-cu130-runtime`, version 0.5.10.post1, port 30000
- Secondary: vLLM NGC 26.01 — `nvcr.io/nvidia/vllm:26.01-py3`, version 0.13.0, port 8000 (only for historical/fallback runs)
- SGLang bench tool: `python3 -m sglang.bench_serving --backend sglang-oai-chat`
- vLLM bench tool: `vllm bench serve --backend openai-chat`
- **Critical sglang bench flag: `--random-range-ratio 1.0`** — the default 0.0 silently samples per-prompt output lengths uniformly from `[1, random_output_len]`, producing ~half the expected total tokens. 1.0 pins every prompt to exactly `random_output_len`. `ignore_eos=true` is already set by default in the chat-backend payload, so no `--extra-request-body` is needed. Both already baked into `bench_sweep.sh`.
- Always pass `--ready-check-timeout-sec 0` to skip the hardcoded per-invocation readiness probe (the bench tool's own JIT flush that decodes `random_output_len` tokens). We curl-warmup once at container launch per SOP.

## Benchmark Config
- TP: 8 for all models (8 GPUs)
- EP: **broken on both frameworks** for NVFP4 at this snapshot. vLLM 0.13.0 rejects EP at kernel dispatch (`cutlass_moe_fp4` won't accept `expert_map`). SGLang 0.5.10.post1 accepts the flag but **crashes during model load** in `sglang/srt/layers/quantization/modelopt_quant.py:1754` with a shape mismatch — `w13_input_scale` is loaded full-size [256, ...] but `w13_weight_scale_2` is EP-sharded to [32, ...] per rank, then their element-wise multiply explodes. Verified on M2.7 NVFP4 2026-04-14 00:52. Use TP-only sharding everywhere until the loader is fixed upstream. Possible workarounds (untried): `--moe-runner-backend flashinfer_trtllm`, or re-quantize the checkpoint with EP-aware ModelOpt scales.
- Sequence profiles: **1k1k (1024/1024), 1k4k (1024/4096), 4k1k (4096/1024)**. Previous 1k8k/8k1k retired (required max-model-len ≥ 9216).
- Concurrency sweep: 1, 2, 4, 8, 16, 32, 64, 128, 256 (extend to 512 if throughput is still climbing at 256)
- max-model-len: **16384** (fits all three profiles with headroom; avoids per-profile restart cost)
- num_prompts = max(concurrency × 10, 40) — 40 is the P99 stability floor
- Stop sweep on any of: non-zero `Failed requests`, `ValueError: Initial test run failed`, `Error: Bad Request`, `Traceback`, `OCI runtime exec failed`, or missing JSON output file. All layered into `bench_sweep.sh`.

## Known Issues
- **EP+NVFP4 broken on BOTH vLLM 0.13.0 AND SGLang 0.5.10.post1.** vLLM rejects at the kernel dispatcher; SGLang crashes during ModelOpt weight post-processing with a w13_input_scale [256] vs w13_weight_scale_2 [32] shape mismatch. Use TP-only on B300 NVFP4 until upstream fixes land. (Verified on M2.7 2026-04-14 00:52.)
- **DeepGEMM not available on NGC 26.01** (26.03 had it, but 26.03 crashes on this driver).
- **Missing B300 MoE config file in vLLM:** `E=64,N=1536,device_name=NVIDIA_B300_SXM6_AC,dtype=fp8_w8a8,block_shape=[128,128].json` — affects FP8 MoE tuning; NVFP4 path is unaffected.
- **SGLang auto-disables CUTLASS MoE on B300** with warning: *"CUTLASS backend is disabled when piecewise cuda graph is enabled due to TMA descriptor initialization issues on B200. Using auto backend instead for stability."* MoE runs through the `auto` backend instead. Stable but possibly not peak perf — follow-up: A/B test `--moe-runner-backend flashinfer_trtllm`.
- **SGLang bench `--random-range-ratio 0.0` is a trap:** it samples per-prompt output lengths uniformly from `[1, random_output_len]` instead of pinning to `random_output_len`. Always pass `--random-range-ratio 1.0`. (Baked into `bench_sweep.sh` — only relevant if you ever call `sglang.bench_serving` directly.)
- **Cold start:** vLLM ~7 min (FlashInfer JIT + torch.compile + CUDA graphs). SGLang ~2–5 min (weight load ~45 s + CUDA graph capture ~1–2 min; no torch.compile phase).

## Key Files
- `~/benchmark/SOP.md` — how to benchmark (process, step-by-step, lessons learned)
- `~/benchmark/STATUS.md` — what's happening now (live state, change log)
- `~/benchmark/configs/*.yaml` — per-model configs
- `~/benchmark/scripts/launch_server.sh` — unified vLLM launch wrapper (single-container guard, standard mounts)
- `~/benchmark/scripts/bench_sweep.sh` — concurrency sweep script (framework-aware, layered stop guards)
- `~/benchmark/scripts/serve_*.sh` — per-model server launch scripts
- `~/benchmark/scripts/check_kimi_download.sh` — read-only download progress check
- `~/benchmark/scripts/patch_bench*.py` — (one-off, reverted) bench_serving.py diagnostic patches from the 2026-04-13 debug session
- `~/benchmark/results/<framework>/<model>/json/` — benchmark result JSONs (e.g. `sglang/minimax-m2.7/json/`, `vllm/kimi-k2.5/json/`)
- `~/benchmark/results/<framework>/<model>/logs/` — per-run stdout tees
- `~/benchmark/results/metadata/` — node_info.yaml, per-container startup yamls, GPU stats

## Model Queue (NVFP4)
Status as of session-end 2026-04-14 02:30 UTC. Always cross-check `STATUS.md` for the latest sweep state before relying on this.

1. **MiniMax M2.7** — `lukealonso/MiniMax-M2.7-NVFP4` (115 GB, downloaded). 1k1k complete on **both** vLLM and SGLang TP=8 (conc 1–512). 1k4k and 4k1k pending. EP=8 confirmed broken on both frameworks.
2. **Kimi K2.5** — `nvidia/Kimi-K2.5-NVFP4` (~591 GB, downloaded). 1k1k SGLang partial (conc 1–128, plateaued at ~2600 t/s). 1k4k and 4k1k pending. DeepSeek-V3-class architecture. Tokenizer slow-path caveat documented in `configs/kimi-k2.5.yaml` — TTFT inflated 10–30%.
3. **Qwen 3.5 397B** — `nvidia/Qwen3.5-397B-A17B-NVFP4` (~251 GB, downloaded). Serve script not yet written — copy `serve_kimi-k2.5_nvfp4_sglang.sh` and swap NAME/MODEL.
4. **GLM-5.1** — `lukealonso/GLM-5.1-NVFP4` (~434 GB, **NOT downloaded**)
5. DeepSeek R1 — `nvidia/DeepSeek-R1-NVFP4` (~400 GB, **TRT-LLM only currently**, deferred)
6. DeepSeek V3.2 — `nvidia/DeepSeek-V3.2-NVFP4` (~415 GB, **TRT-LLM only currently**, deferred)
