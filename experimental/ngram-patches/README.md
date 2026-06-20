# Experimental: Ngram Speculative Decoding + PP Patches

These patches were developed to enable ngram speculative decoding with pipeline parallelism (PP=3) on vLLM. While the engine successfully initializes with these patches, **generation fails with `IndexError`** due to additional unresolved bugs in the ngram + PP interaction path.

**Status:** Does not work. Kept for reference and future investigation.

## What was attempted

1. **MTP (Multi-Token Prediction)** — Draft model architecture (`deepseek_mtp`) doesn't implement `SupportsPP` interface. Hard blocker.

2. **Ngram speculative decoding** — Six cascading code bugs were patched:
   - `gpu_model_runner.py`: `hasattr` guards for `self.drafter` access on non-last PP ranks (3 patches)
   - `scheduler.py`: Skip requests with non-positive scheduled tokens (speculative cleanup sends -3 tokens)
   - `gpu_model_runner.py`: Early return when `num_scheduled_tokens <= 0`
   - After all patches, engine starts but generation crashes with `IndexError: list index out of range`

3. **Memory overhead** — Ngram init adds ~3-5 GB overhead, causing OOM on worker nodes during MoE init. Required 128 GB swap + `MALLOC_ARENA_MAX=1` to survive.

## Files

- `gpu_model_runner.py` — Modified `vllm/v1/worker/gpu_model_runner.py` with `hasattr(self, "drafter")` guards
- `scheduler.py` — Modified `vllm/v1/core/sched/scheduler.py` with `num_tokens_scheduled <= 0` skip

## To experiment

1. Copy these files into `mods/glm-5.2-patches/`
2. Add copy commands to `mods/glm-5.2-patches/run.sh`
3. Add `--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}'` to the recipe
4. Set `gpu_memory_utilization: 0.80` and ensure ≥128 GB swap on all nodes
