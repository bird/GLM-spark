#!/bin/bash
# oom_fixer.sh — Protects critical vLLM/Ray processes from the OOM killer
# by setting oom_score_adj to -1000 (never kill).
# Deploy to /home/<user>/ on every node. Run as a systemd service:
#   sudo systemd-run --unit=oom-fixer --working-directory=/home/<user> /home/<user>/oom_fixer.sh

PROCESSES="vllm|gcs_server|raylet|ray::|ray start|EngineCore"
INTERVAL=1

echo "[oom-fixer] Monitoring processes: $PROCESSES (interval: ${INTERVAL}s)"

while true; do
    pids=$(ps aux | grep -E "$PROCESSES" | grep -v grep | awk '{print $2}')
    for pid in $pids; do
        current=$(cat /proc/$pid/oom_score_adj 2>/dev/null)
        if [[ "$current" != "-1000" ]]; then
            echo -1000 > /proc/$pid/oom_score_adj 2>/dev/null
        fi
    done
    sleep $INTERVAL
done
