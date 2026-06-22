# AEON v0.23.0 + Speculative Decode Investigation

**Date:** June 22, 2026
**Verdict:** Not worth it — 8% slower, none of the AEON-specific features are usable with our setup.

## What was tried

### 1. AEON v0.23.0+aeon.sm121a.dflash image

The [AEON-7/vllm-ultimate-dgx-spark](https://ghcr.io/aeon-7/aeon-vllm-ultimate:latest) image is a purpose-built DGX Spark vLLM with:
- vLLM v0.23.0 + SM121a patches
- DFlash speculative decode
- NVFP4 KV cache support
- CUDA 13.0

Ray 2.55.1 was injected (pip install) to enable multi-node PP=3. The existing GLM-5.2 patches (deepseek_v2.py, weight_utils.py, default_loader.py, triton_decode_attention.py) were grafted via wholesale file copy — this worked without import errors despite the API differences between dev16581 and v0.23.0.

### 2. Speculative decode (ngram_gpu)

Tried `--speculative-config '{"method":"ngram_gpu","num_speculative_tokens":3}'` with `--enforce-eager`.

## Why it doesn't work

### Spec decode is fundamentally broken with PP=3

| Failure | Phase | Detail |
|---|---|---|
| `'GPUModelRunner' object has no attribute 'drafter'` | CUDA graph capture | Drafter only initializes on PP rank 0; workers crash |
| Engine hangs indefinitely | Inference (with --enforce-eager) | Even bypassing CUDA graphs, ngram_gpu stalls during generation |

vLLM's speculative decode infrastructure assumes PP=1. The drafter/proposer is initialized only on the first pipeline stage, but CUDA graph capture and inference require it on all stages. This is the same bug on both dev16581 and v0.23.0.

### DFlash/MTP requires draft model weights

The REAP-pruned checkpoint has **zero MTP/nextn weights** (`num_nextn_predict_layers: 1` is in config.json but no weights exist in the safetensors). DFlash and all MTP-based methods (glm4_moe_mtp, deepseek_mtp, etc.) cannot function without draft weights.

### NVFP4 KV cache is incompatible with MLA

`vllm/config/vllm.py:2134` raises `ValueError: nvfp4 KV cache is not supported with MLA (Multi-head Latent Attention) backends`. Our model uses MLA (TRITON_MLA backend), so this is a hard block.

## Performance comparison

| Build | Version | tok/s | Notes |
|---|---|---|---|
| dsv4 (dev16581) | 0.1.dev16581+gdda4668b5 | **4.6** | Purpose-built Spark image, CUDA graphs |
| AEON | 0.23.0+aeon.sm121a.dflash | **4.24** | ~8% slower (inductor compilation overhead) |

AEON is slower due to v0.23's inductor compilation adding overhead that the dev build doesn't have.

## What WAS valuable from this exercise

### Cache-dropper execute-bit bug (spark-3)

The root cause of spark-3's chronic crashes all session: `/home/bird/cache_dropper_v3.sh` was deployed with mode `-rw-r--r--` (no execute bit). systemd reported `status=203/EXEC` and the service never ran. This left spark-3 with ~10GB free RAM during weight loading (vs ~18GB on nodes where the service worked), causing PP rank load failures and SSH death under memory pressure.

Fix: `sudo chmod +x /home/bird/cache_dropper_v3.sh && sudo systemctl restart cache-dropper`

Check on all nodes: `systemctl is-active cache-dropper`

## Recommendation

Stay on the **dsv4 dev16581 build**. It's faster and all AEON features are unusable with PP=3 + MLA + REAP-pruned checkpoint.

The only path to meaningful speedup is **RDMA via a managed switch** (MikroTik CRS804), which would eliminate the ~60ms/token PP Socket communication overhead (2 hops × ~30ms each). RDMA would cut this to ~5µs/hop, potentially 2-3x decode speedup.
