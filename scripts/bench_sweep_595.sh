#!/bin/bash
# bench_sweep_595.sh — Concurrency sweep for driver 595 cycle
# Results go to ~/benchmark/results_595/<framework>/<model_short>/
#
# Usage: bench_sweep_595.sh <model_id> <precision> <seq_profile> <framework>
#   framework: sglang | vllm

set -u

MODEL="${1:?model id required}"
PRECISION="${2:?precision required (nvfp4|fp8)}"
SEQ="${3:?seq profile required (1k1k|1k4k|4k1k)}"
FRAMEWORK="${4:?framework required (sglang|vllm)}"

PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}"

case "$SEQ" in
  1k1k) ISL=1024; OSL=1024 ;;
  1k4k) ISL=1024; OSL=4096 ;;
  4k1k) ISL=4096; OSL=1024 ;;
  *) echo "Unknown seq profile: $SEQ"; exit 1 ;;
esac

case "$FRAMEWORK" in
  sglang|vllm) : ;;
  *) echo "Unknown framework: $FRAMEWORK"; exit 1 ;;
esac

CID=$(docker ps -q | head -1)
if [ -z "$CID" ]; then
  echo "ERROR: no running container." >&2
  exit 2
fi

_model_base="${MODEL##*/}"
_model_base="${_model_base%-NVFP4}"
_model_base="${_model_base%-nvfp4}"
_model_base="${_model_base%-FP8}"
_model_base="${_model_base%-fp8}"
MODEL_SHORT="${MODEL_SHORT:-${_model_base,,}}"

RESULTS_DIR=~/benchmark/results_595/${FRAMEWORK}/${MODEL_SHORT}/${SEQ}
JSON_DIR=${RESULTS_DIR}/json
LOG_DIR=${RESULTS_DIR}/logs
mkdir -p "$JSON_DIR" "$LOG_DIR"

echo "Sweep 595: framework=$FRAMEWORK model=$MODEL precision=$PRECISION seq=$SEQ model_short=$MODEL_SHORT"
echo "Server at $BASE_URL (container $CID)"
echo "  json -> $JSON_DIR"
echo "  logs -> $LOG_DIR"

PLATEAU_THRESHOLD="${PLATEAU_THRESHOLD:-0.10}"
PREV_TPS=""

for CONC in 1 2 4 8 16 32 64 128 256 512; do
  PROMPTS=$((CONC > 4 ? CONC * 10 : 40))
  TAG="${MODEL##*/}_${PRECISION}_${FRAMEWORK}_tp8_conc${CONC}_${SEQ}"
  LOG=${LOG_DIR}/${TAG}.log
  JSON=${JSON_DIR}/${TAG}.json
  echo "=== $(date) | Conc=$CONC | Prompts=$PROMPTS | $SEQ | $FRAMEWORK ==="

  if [ "$FRAMEWORK" = "sglang" ]; then
    docker exec -i "$CID" \
      python3 -m sglang.bench_serving \
      --backend sglang-oai-chat \
      --base-url "$BASE_URL" \
      --model "$MODEL" \
      --dataset-name random \
      --random-input-len $ISL \
      --random-output-len $OSL \
      --random-range-ratio 1.0 \
      --num-prompts $PROMPTS \
      --max-concurrency $CONC \
      --ready-check-timeout-sec 0 \
      --output-file /results/${FRAMEWORK}/${MODEL_SHORT}/${SEQ}/json/${TAG}.json \
      2>&1 | tee "$LOG"
  else
    docker exec -i "$CID" \
      vllm bench serve \
      --backend openai-chat \
      --base-url "$BASE_URL" \
      --model "$MODEL" \
      --dataset-name random \
      --random-input-len $ISL \
      --random-output-len $OSL \
      --num-prompts $PROMPTS \
      --max-concurrency $CONC \
      --ready-check-timeout-sec 0 \
      --endpoint /v1/chat/completions \
      --save-result \
      --result-filename /results/${FRAMEWORK}/${MODEL_SHORT}/${SEQ}/json/${TAG}.json \
      2>&1 | tee "$LOG"
  fi

  if grep -q "Failed requests:" "$LOG"; then
    FAILS=$(grep "Failed requests:" "$LOG" | awk '{print $NF}')
    if [ "$FAILS" != "0" ] 2>/dev/null; then
      echo "!!! $FAILS failures at conc=$CONC — stopping sweep"
      break
    fi
  fi

  if grep -qE "ValueError: Initial test run failed|Error: Bad Request|^Traceback|OCI runtime exec failed" "$LOG"; then
    echo "!!! bench tool error at conc=$CONC — stopping sweep"
    break
  fi

  if [ ! -s "$JSON" ]; then
    echo "!!! no JSON result at conc=$CONC — stopping sweep"
    break
  fi

  CUR_TPS=$(python3 -c "
import json, sys
try:
    d = json.load(open('$JSON'))
    v = d.get('output_throughput')
    print(v if v is not None else '')
except Exception as e:
    sys.stderr.write(f'parse failed: {e}\n')
    print('')
" 2>/dev/null)

  if [ -z "$CUR_TPS" ]; then
    echo "!!! could not parse output_throughput — stopping sweep"
    break
  fi

  if [ -n "$PREV_TPS" ]; then
    read -r GAIN_PCT STOP < <(python3 -c "
prev = float('$PREV_TPS')
cur  = float('$CUR_TPS')
thr  = float('$PLATEAU_THRESHOLD')
gain = (cur - prev) / prev if prev > 0 else 0.0
stop = 1 if gain < thr else 0
print(f'{gain*100:.2f} {stop}')
")
    echo "+++ conc=$CONC output_throughput=${CUR_TPS} tok/s (gain vs prev ${GAIN_PCT}%)"
    if [ "$STOP" = "1" ]; then
      echo "!!! plateau at conc=$CONC (gain ${GAIN_PCT}% < 10%) — stopping sweep"
      break
    fi
  else
    echo "+++ conc=$CONC output_throughput=${CUR_TPS} tok/s (baseline)"
  fi
  PREV_TPS="$CUR_TPS"
done
echo "=== sweep done at $(date) ==="
