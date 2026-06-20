#!/bin/bash
# cache_dropper.sh — Aggressively drops Linux page cache on Grace Blackwell
# to free unified memory for GPU workloads.
# Deploy to /home/<user>/ on every node. Run as a systemd service:
#   sudo systemd-run --unit=cache-dropper --working-directory=/home/<user> /home/<user>/cache_dropper.sh

THRESHOLD_KB=10485760  # 10 GB in KB
INTERVAL=1

echo "[cache-dropper] Dropping cache when free < ${THRESHOLD_KB} KB (interval: ${INTERVAL}s)"

while true; do
    meminfo=$(cat /proc/meminfo)
    free_kb=$(echo "$meminfo" | grep "^MemFree:" | awk '{print $2}')

    if [[ -n "$free_kb" && "$free_kb" -lt "$THRESHOLD_KB" ]]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
    fi

    sleep $INTERVAL
done
