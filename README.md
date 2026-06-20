# GLM-5.2 469B on 3× NVIDIA DGX Spark

Serve the **GLM-5.2-NVFP4-REAP-469B** model (753B → 469B REAP-pruned, NVFP4 quantized) across a cluster of three NVIDIA DGX Spark nodes using vLLM with pipeline parallelism.

- **Context:** 256K tokens
- **Throughput:** ~4.4 tok/s decode, ~2,500–3,800 tok/s prefill
- **Architecture:** PP=3, TP=1, Ray distributed backend
- **Quantization:** NVFP4 (4-bit) with fp8 KV cache

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Spark #1   │      │  Spark #2   │      │  Spark #3   │
│ (PP rank 0) │ ───▶ │ (PP rank 1) │ ───▶ │ (PP rank 2) │
│  26 layers  │      │  27 layers  │ NCCL │  26 layers  │
│  83.9 GiB   │      │  90.3 GiB   │      │  92.1 GiB   │
│  Head node  │      │   Worker    │      │   Worker    │
└─────────────┘      └─────────────┘      └─────────────┘
       │
       ▼
  :8000 HTTP API
```

## Hardware Requirements

| Component | Spec |
|-----------|------|
| Nodes | 3× NVIDIA DGX Spark (GB10 Grace Blackwell) |
| GPU | 121 GB unified LPDDR5x per node (363 GB total) |
| Bandwidth | 273 GB/s per node, 200G RoCE dual-port ConnectX-7 |
| Disk | ≥1 TB NVMe per node (head node: ≥4 TB recommended) |
| Network | 10 GbE minimum between nodes (RoCE optional) |
| OS | Ubuntu 22.04+ (aarch64) |

## Prerequisites

### 1. Docker Image

Build or pull the vLLM Docker image for DGX Spark:

```bash
# The image must include vLLM, Ray, FlashInfer, and CUTLASS
# Target version: vLLM 0.1.dev16581+gdda4668b5
docker pull <your-registry>/vllm-node-dsv4-cl:latest
```

### 2. Model Download

Download the model to **every node** at the same path:

```bash
# On each node:
mkdir -p ~/models
huggingface-cli download 0xSero/GLM-5.2-NVFP4-REAP-469B \
  --local-dir ~/models/GLM-5.2-NVFP4-REAP-469B
```

> **Disk space:** ~287 GB per node. The head node (4 TB) is recommended.

### 3. Passwordless SSH

Set up passwordless SSH from the head node to all worker nodes:

```bash
# On head node:
ssh-keygen -t ed25519
ssh-copy-id <user>@<worker-1-ip>
ssh-copy-id <user>@<worker-2-ip>
```

## Quick Start

### Step 1 — Clone & Configure

```bash
git clone https://github.com/bird/GLM-spark.git
cd GLM-spark

# Create .env from template
cp .env.example .env
# Edit .env with your node IPs and interface names
vim .env
```

### Step 2 — System Tuning (all nodes)

Run these once on **every** DGX Spark node:

```bash
# 1. Kernel tuning (persistent across reboots)
sudo bash scripts/apply_sysctl.sh

# 2. Create swap (64 GB+ recommended)
sudo fallocate -l 64G /swapfile2
sudo chmod 600 /swapfile2 && sudo mkswap /swapfile2 && sudo swapon /swapfile2
echo '/swapfile2 none swap sw 0 0' | sudo tee -a /etc/fstab

# 3. Stop GUI and unnecessary services
sudo systemctl stop gdm snapd 2>/dev/null

# 4. Deploy daemons (OOM protector + cache dropper)
cp scripts/oom_fixer.sh scripts/cache_dropper.sh ~/
chmod +x ~/oom_fixer.sh ~/cache_dropper.sh
sudo systemd-run --unit=oom-fixer --working-directory=$HOME ~/oom_fixer.sh
sudo systemd-run --unit=cache-dropper --working-directory=$HOME ~/cache_dropper.sh
```

### Step 3 — Launch

From the **head node** only:

```bash
export VLLM_SPARK_EXTRA_DOCKER_ARGS="--pid=host -v $HOME/models:/home/bird/models:ro"
nohup ./launch-cluster.sh recipes/glm-5.2-nvfp4-reap-469b.yaml -d > /tmp/recipe_launch.log 2>&1 &
```

### Step 4 — Wait (~15 min)

Weight loading + MoE initialization takes approximately 15 minutes. Monitor progress:

```bash
# Inside the container on the head node:
docker exec vllm_ds4 tail -f /tmp/vllm_serve.log
```

When you see `Application startup complete`, the API is ready.

### Step 5 — Verify

```bash
curl http://<head-ip>:8000/v1/models

curl http://<head-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.2-NVFP4-REAP-469B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Configuration

### Recipe (`recipes/glm-5.2-nvfp4-reap-469b.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `pipeline_parallel` | 3 | One rank per Spark node |
| `tensor_parallel` | 1 | GB10 has a single GPU |
| `gpu_memory_utilization` | 0.85 | Leaves ~18 GB for OS on 121 GB nodes |
| `max_model_len` | 262144 | 256K context |
| `kv_cache_dtype` | fp8 | Halves KV cache memory |
| `block_size` | 128 | Optimized for long-context |
| `max_num_seqs` | 1 | Single-request optimization |
| `max_num_batched_tokens` | 2048 | Chunked prefill granularity |

### Key Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `VLLM_DISABLE_DSA` | 1 | Disables Dynamic Shared Attention (REAP incompatible) |
| `MALLOC_ARENA_MAX` | 1 | Minimizes malloc arena overhead on unified memory |
| `RAY_health_check_failure_threshold` | 600 | Prevents false node-death during MoE init |
| `RAY_health_check_period_ms` | 10000 | Health check interval |
| `RAY_CGRAPH_get_timeout` | 3600 | Ray compiled graph timeout (MoE init is slow) |
| `NCCL_IB_DISABLE` | 1 | Socket transport (IB/RoCE needs switch config) |

## Patches (`mods/glm-5.2-patches/`)

Four vLLM source patches are applied automatically at container startup:

### 1. `deepseek_v2.py` — Model Compatibility
- Disables DSA (Dynamic Shared Attention)
- Skips loading of indexer weights not present in REAP checkpoint
- Adds REAP expert remapping (original 256 → pruned 156 experts/layer)

### 2. `weight_utils.py` — Memory Management
- Adds `posix_fadvise(DONTNEED)` during safetensors loading to prevent page cache accumulation
- **Critical on Grace Blackwell**: unified memory means page cache competes with GPU for RAM
- Adds PP-aware file filtering (each node only loads its own checkpoint shards)

### 3. `default_loader.py` — PP-Aware Loading
- Calls `filter_files_by_pp_rank()` to skip irrelevant shards during weight loading

### 4. `triton_decode_attention.py` — Kernel Fix
- Forces `num_stages=1` when `BLOCK_DMODEL >= 512` (MLA with Lk=576)
- Fixes: `OutOfResources: shared memory, Required: 102400, Hardware limit: 101376` on SM_121

## Operational Guide

### Clean Shutdown & Restart

Always use the cleanup script before relaunching — `docker stop` alone leaves orphaned Ray processes (due to `--pid=host`):

```bash
# On every node:
sudo bash ~/cleanup_ray.sh
```

Or from the head node:

```bash
# Clean all nodes at once:
ssh <worker-1> "sudo bash ~/cleanup_ray.sh"
ssh <worker-2> "sudo bash ~/cleanup_ray.sh"
sudo bash ~/cleanup_ray.sh
```

### Checking Logs

```bash
# vLLM server log (inside container on head node)
docker exec vllm_ds4 tail -f /tmp/vllm_serve.log

# Ray GCS server log
docker exec vllm_ds4 cat /tmp/ray/session_latest/logs/gcs_server.out | tail -20

# Per-worker Ray logs
ssh <worker-ip> "docker exec vllm_ds4 cat /tmp/ray/session_latest/logs/worker-*.err" | tail -20
```

### Node Recovery

If a worker node becomes unreachable (OOM crash):

1. **Physically power-cycle the node** (ASUS Ascent Sparks do not auto-recover)
2. Re-apply kernel tuning (persisted via sysctl.d, but verify):
   ```bash
   sudo sysctl -p /etc/sysctl.d/99-vllm-spark.conf
   ```
3. Restart daemons:
   ```bash
   sudo systemd-run --unit=oom-fixer --working-directory=$HOME ~/oom_fixer.sh
   sudo systemd-run --unit=cache-dropper --working-directory=$HOME ~/cache_dropper.sh
   ```
4. Clean Ray on all nodes, then relaunch

## Troubleshooting

### Worker node marked dead during startup

**Cause:** Ray GCS health check fails because the raylet is unresponsive during MoE init (memory pressure on Grace Blackwell).

**Fix:** Ensure `RAY_health_check_failure_threshold=600` is set in `launch-cluster.sh` → `get_env_flags()`. With a 10-second period, this gives 100 minutes of grace.

### `OutOfResources: shared memory` during decode

**Cause:** Triton MLA decode kernel requires more shared memory than SM_121 provides (101,376 bytes).

**Fix:** The `triton_decode_attention.py` patch handles this. Verify it was applied by checking the startup log for `[glm-5.2-patches] Patched triton_decode_attention.py`.

### OOM crash during weight loading

**Cause:** Page cache from safetensors loading consumes all 121 GB of unified memory.

**Fix:**
1. Verify `cache_dropper.sh` is running (`systemctl is-active cache-dropper`)
2. Verify `posix_fadvise` patch is applied (check for `[glm-5.2-patches] Patched weight_utils.py`)
3. Ensure swap is configured (64 GB+ recommended)
4. Run `apply_sysctl.sh` for kernel tuning

### `ActorHandleNotFoundError` during init

**Cause:** Ray compiled graph timeout (default 600s, but MoE init can take 20+ minutes).

**Fix:** Set `CONTAINER_RAY_CGRAPH_get_timeout=3600` in `.env`.

### SSH connection refused on worker node

**Cause:** Node OOM'd. The system is alive at the network level but SSH daemon was killed.

**Fix:** Power-cycle the node physically. ASUS Ascent Sparks do not auto-power on after a crash.

## Performance

| Metric | Value |
|--------|-------|
| Decode throughput | ~4.4 tok/s (single request) |
| Prefill throughput | ~2,500–3,800 tok/s |
| TTFT (short prompt) | ~0.4 s |
| TTFT (25K prompt) | ~60 s |
| Weight loading time | ~10 min per node |
| MoE init time | 2–15 min per node (varies by rank) |
| KV cache per node | 7–15 GB (depends on PP rank) |
| Max concurrency @ 256K | 1.6–1.9× |

### Decode Bottleneck Analysis

Theoretical memory-bound decode limit: ~28.4 tok/s (273 GB/s ÷ 9.6 GB active params per token).

Actual: 4.4 tok/s = **15.5% of theoretical**.

Overhead sources:
- `--enforce-eager` (no CUDA graphs): kernel launch overhead per step
- Python dispatch + scheduler overhead
- PP synchronization (3 NCCL hops per token)
- Socket transport (NCCL over TCP, no IB/RoCE)

Removing `--enforce-eager` would improve throughput significantly but causes heartbeat timeouts during CUDA graph compilation on Grace Blackwell.

## Speculative Decoding

Ngram speculative decoding was explored but **does not work** with PP=3 in this vLLM version due to multiple code bugs in the ngram + PP interaction path. The experimental patches are in `experimental/ngram-patches/` for reference. See [`experimental/ngram-patches/`](experimental/ngram-patches/) for details.

## Model Details

| Property | Value |
|----------|-------|
| Original parameters | 753B |
| REAP-pruned parameters | 469B |
| Experts per layer (original) | 256 |
| Experts per layer (pruned) | 156 |
| Active experts per token | 8 |
| Hidden size | 6,144 |
| Layers | 79 (78 dense/MoE + 1 MTP) |
| Vocab size | 154,880 |
| Architecture | `GlmMoeDsaForCausalLM` |
| Quantization | NVFP4 (4-bit) |
| Disk size | ~287 GB |

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) — inference engine
- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) — hardware platform
- [0xSero](https://huggingface.co/0xSero) — REAP-pruned NVFP4 checkpoint
