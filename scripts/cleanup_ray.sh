#!/bin/bash
# cleanup_ray.sh — Safely stop Ray and clean stale sessions
# Run on each node before relaunching the cluster.
# Deploy to /home/<user>/ on every DGX Spark node.

echo "[cleanup] Stopping container..."
docker stop vllm_ds4 2>/dev/null
docker rm -f vllm_ds4 2>/dev/null

echo "[cleanup] Gracefully stopping Ray (SIGTERM)..."
killall -TERM raylet gcs_server 2>/dev/null
sleep 5

echo "[cleanup] Force killing remaining Ray processes..."
killall -9 raylet gcs_server 2>/dev/null
killall -9 -f "ray::" 2>/dev/null
sleep 2

echo "[cleanup] Removing Ray session data..."
rm -rf /tmp/ray

echo "[cleanup] Dropping caches..."
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null

echo "[cleanup] Done. Remaining raylet processes:"
ps aux | grep raylet | grep -v grep | wc -l
