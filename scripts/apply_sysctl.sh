#!/bin/bash
# apply_sysctl.sh — Kernel tuning for Grace Blackwell unified memory
# Makes settings persistent across reboots.
# Run once on every DGX Spark node: sudo bash apply_sysctl.sh

echo "[sysctl] Applying kernel tuning for vLLM on Grace Blackwell..."

sudo tee /etc/sysctl.d/99-vllm-spark.conf > /dev/null << 'EOF'
vm.swappiness=100
vm.vfs_cache_pressure=1000
vm.min_free_kbytes=2097152
vm.dirty_ratio=5
vm.dirty_background_ratio=1
EOF

sudo sysctl -p /etc/sysctl.d/99-vllm-spark.conf

echo "[sysctl] Current values:"
echo "  swappiness:          $(cat /proc/sys/vm/swappiness)"
echo "  vfs_cache_pressure:  $(cat /proc/sys/vm/vfs_cache_pressure)"
echo "  min_free_kbytes:     $(cat /proc/sys/vm/min_free_kbytes)"
echo "  dirty_ratio:         $(cat /proc/sys/vm/dirty_ratio)"
echo "  dirty_background:    $(cat /proc/sys/vm/dirty_background_ratio)"
echo "[sysctl] Settings persist across reboots via /etc/sysctl.d/99-vllm-spark.conf"
