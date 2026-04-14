#!/bin/bash
# check_kimi_download.sh — quick status on the Kimi K2.5 NVFP4 download
# Target size: ~591 GB

TARGET=~/.cache/huggingface/hub/models--nvidia--Kimi-K2.5-NVFP4
TOTAL_GB=591

echo "=== Kimi K2.5 NVFP4 download status @ $(date) ==="

# 1. Is the download process alive?
PID=$(pgrep -f "huggingface-cli download nvidia/Kimi-K2.5-NVFP4" | head -1)
if [ -n "$PID" ]; then
  echo "Process:      ALIVE  (pid $PID)"
  ps -o pid,etime,stat,cmd -p "$PID" | tail -1
else
  echo "Process:      NOT RUNNING  (download may be finished, stalled, or killed)"
fi

# 2. How much is on disk?
if [ -d "$TARGET" ]; then
  SIZE_BYTES=$(du -sb "$TARGET" 2>/dev/null | awk '{print $1}')
  SIZE_GB=$(awk "BEGIN {printf \"%.1f\", $SIZE_BYTES/1024/1024/1024}")
  PCT=$(awk "BEGIN {printf \"%.1f\", ($SIZE_GB/$TOTAL_GB)*100}")
  echo "On disk:      ${SIZE_GB} GB / ~${TOTAL_GB} GB  (${PCT}%)"
else
  echo "On disk:      0 GB  (target dir does not exist yet)"
fi

# 3. Incomplete shard count (files ending in .incomplete)
INCOMPLETE=$(find "$TARGET" -name "*.incomplete" 2>/dev/null | wc -l)
echo "Incomplete:   $INCOMPLETE shard(s) still downloading"

# 4. Free space on the cache partition
echo "Free space:   $(df -h "$(dirname "$TARGET")" | awk 'NR==2 {print $4 " free of " $2}')"
