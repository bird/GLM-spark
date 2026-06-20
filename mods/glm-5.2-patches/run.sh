#!/bin/bash
set -e

VLLM_BASE="/usr/local/lib/python3.12/dist-packages/vllm"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[glm-5.2-patches] Applying patches from $SCRIPT_DIR..."

# Patch 1: deepseek_v2.py
# - Disables DSA (Dynamic Shared Attention) — REAP checkpoint has indexer
#   weights that aren't handled
# - Skips loading indexer weights
# - Adds REAP expert remapping (256 -> 156 experts per layer)
cp "$SCRIPT_DIR/deepseek_v2.py" "$VLLM_BASE/model_executor/models/deepseek_v2.py"
echo "[glm-5.2-patches] Patched deepseek_v2.py"

# Patch 2: weight_utils.py
# - Adds posix_fadvise(DONTNEED) to prevent page cache accumulation
#   during weight loading (critical on Grace Blackwell unified memory)
# - Adds PP-aware file filtering so each node only loads its own shards
cp "$SCRIPT_DIR/weight_utils.py" "$VLLM_BASE/model_executor/model_loader/weight_utils.py"
echo "[glm-5.2-patches] Patched weight_utils.py"

# Patch 3: default_loader.py
# - Calls filter_files_by_pp_rank to skip irrelevant checkpoint shards
cp "$SCRIPT_DIR/default_loader.py" "$VLLM_BASE/model_executor/model_loader/default_loader.py"
echo "[glm-5.2-patches] Patched default_loader.py"

# Patch 4: triton_decode_attention.py
# - Forces num_stages=1 when BLOCK_DMODEL >= 512 (MLA with Lk=576)
# - Fixes: OutOfResources shared memory (required 102400, limit 101376 on SM_121)
cp "$SCRIPT_DIR/triton_decode_attention.py" "$VLLM_BASE/v1/attention/ops/triton_decode_attention.py"
echo "[glm-5.2-patches] Patched triton_decode_attention.py"

echo "[glm-5.2-patches] All patches applied successfully."
