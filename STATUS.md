# B300 Benchmark — Live Status

Last updated: 2026-04-14 02:30 UTC (session-end snapshot)

## Currently Running
- Container: **none** (all containers killed for clean handover)
- Benchmark: idle
- Downloads: see [Downloads](#downloads-state) below

## Completed Results

### MiniMax M2.7 (lukealonso/MiniMax-M2.7-NVFP4)
| Framework | Profile | EP | conc range | Levels | Workload validated | JSON path |
|---|---|---|---|---|---|---|
| vLLM 0.13.0 (NGC 26.01) | 1k1k | 1 | 1–512 | 10 | ✓ (40960 tokens × 40 prompts at conc=1, etc.) | `~/benchmark/results/vllm/minimax-m2.7/json/` |
| SGLang 0.5.10.post1 | 1k1k | 1 | 1–512 | 10 | ✓ (after `--random-range-ratio 1.0` fix) | `~/benchmark/results/sglang/minimax-m2.7/json/` |
| SGLang 0.5.10.post1 | 1k1k | **8** | — | — | **✗ FAILED** at model load (modelopt loader bug, see below) | n/a |

**Plus 2 historical FP8 JSONs** at `~/benchmark/results/vllm/minimax-m2.7/json/m27_fp8_tp4_conc{4,16}_1k1k.json` from a pre-SOP NGC 26.03 run. Retained for reference; not part of the canonical NVFP4 sweep.

### Kimi K2.5 (nvidia/Kimi-K2.5-NVFP4)
| Framework | Profile | EP | conc range | Levels | Workload validated | JSON path |
|---|---|---|---|---|---|---|
| SGLang 0.5.10.post1 | 1k1k | 1 | 1–128 | 8 | ✓ | `~/benchmark/results/sglang/kimi-k2.5/json/` |

**Note:** Sweep was stopped at conc=128 because Kimi's throughput plateaued around 2600 t/s (conc=32 → 64 = +4.3%, conc=64 → 128 = +0.6%). Pushing to conc=256/512 would only add latency. **conc=256 and conc=512 are intentionally not collected** for Kimi 1k1k. If we revisit, document in writeup that the curve is complete to its knee.

## Headline Findings (this session)

1. **SGLang vs vLLM on M2.7 1k1k (TP-only):**
   - At conc=1: SGLang **+42.7%** throughput (104 vs 73 t/s), **-63%** TTFT (45 vs 121 ms), **-30%** TPOT (9.5 vs 13.5 ms).
   - At conc=512 the throughput gap vanishes (+2.4%), and SGLang's TTFT goes 4.4× **worse** (3484 ms vs 791 ms). SGLang's TTFT scales linearly with concurrency on B300; vLLM scales sublinearly.
   - **Sweet spots:** SGLang dominates conc≤32 (interactive), vLLM wins conc≥128 (high-load batch).

2. **EP+NVFP4 broken on BOTH frameworks** (verified by trying):
   - vLLM 0.13.0 — `cutlass_moe_fp4` kernel rejects EP at dispatch.
   - SGLang 0.5.10.post1 — `modelopt_quant.py:1754` shape mismatch during weight post-processing: `w13_input_scale[256] × w13_weight_scale_2[32]`. Loader replicates input_scale across all experts but EP-shards weight_scale_2 to per-rank 32 — the multiply explodes. Both fail at runtime regardless of `--moe-runner-backend` (which only affects forward kernel, not loader).

3. **Kimi vs M2.7 (both SGLang):**
   - Kimi has dramatically lower TPOT (5.6 vs 9.5 ms at conc=1) thanks to MLA's compact KV.
   - Kimi prefills ~6× slower (260 vs 45 ms TTFT at conc=1) — partially due to bigger model, partially due to a **slow tokenizer fallback path** (`tokenization_kimi.py:178` warns "Calling super().encode" whenever sglang passes `add_special_tokens=False`, which it does on every chat-template encoding). Kimi's reported TTFT is ~10–30% pessimistic; documented in `configs/kimi-k2.5.yaml`.
   - Kimi peaks at ~2600 t/s by conc=64 (saturates), M2.7 climbs all the way to ~10300 t/s at conc=512.

4. **`--random-range-ratio 0.0` is a trap** in `sglang.bench_serving`. It samples per-prompt output_len uniformly from `[1, full_len]`, NOT "no variation". You must pass `--random-range-ratio 1.0` for fixed-length outputs. Cost me ~1 hour of misdiagnosis (chasing `ignore_eos` paths that weren't broken). Now baked into `bench_sweep.sh`.

5. **`max_model_len=8192` was too small** for the original 1k8k/8k1k profiles (1024+8192 = 9216 > 8192). Sweep was silently failing for hours because `bench_sweep.sh`'s only stop guard was `"Failed requests"`, which doesn't match `ValueError: Initial test run failed ... Error: Bad Request`. Both fixed: max-model-len now 16384, and `bench_sweep.sh` now layers in `ValueError|Bad Request|Traceback|missing-JSON` stop guards.

## Pending

### Higher priority
- **M2.7 SGLang 1k4k profile** (decode-heavy, ISL=1024 OSL=4096) — never run
- **M2.7 SGLang 4k1k profile** (prefill-heavy, ISL=4096 OSL=1024) — never run
- **Kimi K2.5 SGLang 1k4k profile** — never run (will likely also peak at low conc due to throughput saturation)
- **Kimi K2.5 SGLang 4k1k profile** — never run (prefill-heavy, will be where Kimi's prefill weakness shows the worst)
- **Qwen 3.5 397B-A17B SGLang all profiles** — model downloaded (per session start), serve script needed (`serve_qwen3.5-397b_nvfp4_sglang.sh`)

### Lower priority
- **GLM-5.1 SGLang all profiles** — model NOT downloaded yet; need `huggingface-cli download lukealonso/GLM-5.1-NVFP4` (~434 GB)
- **DeepSeek-R1 / DeepSeek-V3.2** — TRT-LLM only stack; deferred until a different serving stack is set up
- **Try `--moe-runner-backend flashinfer_trtllm` on M2.7 SGLang TP-only** — might fix the high-concurrency TTFT regression we saw (3.5 s at conc=512). Cheap follow-up: kill+relaunch with the flag, run conc=128/256/512 only, compare.

## Blockers & Open Issues

- **EP+NVFP4 broken on both frameworks** (see above). `--moe-runner-backend` swap will NOT fix it (the failure is in the weight loader, not the MoE kernel). Real fix requires either an upstream patch to `modelopt_quant.process_weights_after_loading` or re-quantizing the checkpoint with EP-aware ModelOpt.
- **SGLang TTFT scales linearly with concurrency on B300** — likely tied to the CUTLASS MoE auto-disable (`auto` fallback backend) plus prefill scheduling. `flashinfer_trtllm` MoE backend is the next thing to try.
- **Kimi tokenizer slow path** — every `encode()` call from sglang's chat preprocessor takes the `super().encode()` slow path because sglang passes `add_special_tokens=False` and Kimi's fast path bails on any kwargs. ~10–30% TTFT penalty, documented in `configs/kimi-k2.5.yaml`. Real fix is on the SGLang side.
- **Driver 590.48 blocks NGC 26.03** — driver upgrade scheduled in a future maintenance window.
- **No canonical preflight** for `max_model_len ≥ max(ISL+OSL)` — relies on the SOP rule and the multi-layer stop guards in `bench_sweep.sh`. Fine for now.
- **Sudo cache is per-tty** (`tty_tickets`). Each tmux pane needs its own password. Documented option to fix: `usermod -aG docker howell` + relogin (deferred to driver upgrade window — relogin would kill all tmux sessions).

## Downloads State

- ✅ `lukealonso/MiniMax-M2.7-NVFP4` (~115 GB) — `~/.cache/huggingface/hub/models--lukealonso--MiniMax-M2.7-NVFP4/`
- ✅ `nvidia/Kimi-K2.5-NVFP4` (~591 GB, all 140 files) — `~/.cache/huggingface/hub/models--nvidia--Kimi-K2.5-NVFP4/`
- ✅ `nvidia/Qwen3.5-397B-A17B-NVFP4` (~251 GB) — user reported complete during session
- ❌ `lukealonso/GLM-5.1-NVFP4` (~434 GB) — not started
- ⏸ `nvidia/DeepSeek-R1-NVFP4` (~400 GB) — TRT-LLM only, deferred
- ⏸ `nvidia/DeepSeek-V3.2-NVFP4` (~415 GB) — TRT-LLM only, deferred

(Also in HF cache: a stale FP8 `MiniMaxAI/MiniMax-M2.7` ~215 GB from before the NVFP4 pivot — can be deleted to reclaim disk if needed.)

## What to Do Next Session

The cheapest, highest-information next step is **one of**:
1. **M2.7 SGLang 1k4k sweep** — fills out the matrix for the model we already understand. ~20 min cold start + sweep.
2. **`--moe-runner-backend flashinfer_trtllm` on M2.7 1k1k** — A/B test on the same model. If it fixes the high-conc TTFT regression that's a big win for the writeup. ~15 min.
3. **Qwen 3.5 397B SGLang launch** — third model in the queue, cleanest comparison story. Needs `serve_qwen3.5-397b_nvfp4_sglang.sh` (template ready: copy `serve_kimi-k2.5_nvfp4_sglang.sh`, swap NAME and MODEL).

I'd lean **(2)** as the next move because TTFT is the open performance question on SGLang and the test is dirt cheap. But **(3)** has the most queue-progress value.

## Change Log

- 2026-04-13 15:45 — Started M2.7 FP8 on vLLM 26.01 TP=4 (historical, retained)
- 2026-04-13 17:01 — Started Kimi K2.5 NVFP4 download
- 2026-04-13 18:30 — Launched M2.7 NVFP4 on vLLM 26.01 TP=8 (EP broken with FP4)
- 2026-04-13 19:00 — Completed M2.7 vLLM 1k1k sweep (conc 1–512)
- 2026-04-13 19:30 — Discovered 1k8k/8k1k context overflow; wasted ~2 h on silently-failing runs
- 2026-04-13 20:00 — Switched profiles to 1k1k/1k4k/4k1k, max-model-len 16384
- 2026-04-13 20:45 — Switched framework to SGLang (lmsysorg/sglang:latest-cu130-runtime), pulled image
- 2026-04-13 23:04 — M2.7 SGLang container ready; first sweep attempt
- 2026-04-13 23:18 — First SGLang 1k1k sweep finished but produced wrong total_output_tokens (range_ratio bug)
- 2026-04-13 23:50 — Root-caused to `--random-range-ratio 0.0` (samples uniformly from [1, full_len])
- 2026-04-14 00:00 — Re-launched M2.7 SGLang 1k1k sweep with fix; correct workload confirmed
- 2026-04-14 00:30 — M2.7 SGLang 1k1k sweep complete (conc 1–256)
- 2026-04-14 00:42 — Reorganized `results/` into framework/json+logs split
- 2026-04-14 00:48 — One-off conc=512 against same warm container; throughput tied with vLLM, TTFT 4.4× worse
- 2026-04-14 00:52 — **EP=8 attempt FAILED** — modelopt loader shape mismatch during weight post-processing
- 2026-04-14 01:14 — Reorganized `results/` into per-model subfolders (`<framework>/<model>/json,logs`)
- 2026-04-14 01:15 — Kimi K2.5 SGLang container up (~2 min cold start), warmup successful
- 2026-04-14 01:26 — Kimi K2.5 1k1k sweep started
- 2026-04-14 01:35 — Identified Kimi tokenizer slow-path warning (`Calling super().encode` from `tokenization_kimi.py:178`) — documented as caveat, not patched
- 2026-04-14 02:15 — Stopped Kimi sweep at conc=128 (throughput plateaued at ~2600 t/s, no point pushing further)
- 2026-04-14 02:30 — Killed Kimi container, session-end snapshot written
