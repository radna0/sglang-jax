# DFLASH on TPU for GPT‑OSS (sglang‑jax)

This is the **spec‑v1** DFLASH implementation ported into `sglang-jax`. The immediate goal is:

1) **Correctness**: target-only and DFLASH both run without crashes on TPU  
2) **Train a GPT‑OSS draft** (outside this repo) so accept_len/accept_rate rises  
3) **Measure real decode speedup** (long decode, large batch) and iterate toward **5–6×**

## Current guardrails (intentional)

- DFLASH is **greedy-only** (verification is greedy; draft is greedy).
- DFLASH requires `--page-size 1`.
- DFLASH requires `--disable-overlap-schedule`.
- DFLASH currently requires `--disable-radix-cache` (prefix reuse is not supported yet for draft KV materialization).

## Smoke run (untrained draft)

Even with a random / poorly-trained draft model, this smoke run is useful to validate:

- the server starts
- DFLASH can do draft → verify → commit/rollback without crashing
- acceptance stats are non-NaN (likely low)

### Baseline (target-only)

```bash
PYTHONPATH=python \
python -m sgl_jax \
  --model-path <GPT_OSS_TARGET_REPO_OR_PATH> \
  --device tpu \
  --tp-size 1
```

### DFLASH (target + draft)

```bash
PYTHONPATH=python \
python -m sgl_jax \
  --model-path <GPT_OSS_TARGET_REPO_OR_PATH> \
  --device tpu \
  --tp-size 1 \
  --speculative-algorithm DFLASH \
  --disable-overlap-schedule \
  --disable-radix-cache \
  --speculative-draft-model-path <DFLASH_DRAFT_REPO_OR_PATH> \
  --speculative-num-draft-tokens 8
```

Notes:

- `--speculative-num-draft-tokens` is the **DFLASH block size** (verify window length).
- For an untrained draft model, it is normal for accept_rate to be very low and throughput to be worse than baseline.

## Unit tests (CPU/JAX)

```bash
cd harmony/cuda-norm/external/sglang-jax
PYTHONPATH=python python -m unittest -q \
  sgl_jax.test.speculative.test_dflash_utils \
  sgl_jax.test.speculative.test_dflash_verify_input
```

## Training (draft model)

`sglang-jax` only covers **inference/runtime**. Draft training for GPT‑OSS is handled in this repo via **EasyDeL**:

- EasyDeL trainer: `harmony/cuda-norm/external/EasyDeL/easydel/trainers/dflash_trainer.py`
- Teacher-cache builder scripts: `harmony/cuda-norm/scripts/tpu_dflash_build_teacher_cache.py`
- Cache acceptance checker: `harmony/cuda-norm/scripts/tpu_dflash_cache_accept_check.py`

Once you have an EasyDeL run directory with a trained draft, the remaining step is to export/load it as an HF-style checkpoint for `--speculative-draft-model-path` (conversion/shim work is tracked separately).

## Export + benchmark (recommended)

These repo scripts wire the full TPU loop:

1) Export an EasyDeL run dir (tensorstore) into an HF-style `DFlashDraftModel` directory:

```bash
RUN_DIR=/dev/shm/dflash-checkpoints/<run_name>/run-10000 \
DST=harmony/cuda-norm/artifacts/dflash_draft_ckpts/<ckpt_name> \
bash harmony/cuda-norm/scripts/tpu_sglang_jax_export_dflash_ckpt.sh
```

2) Run a baseline-vs-DFLASH throughput comparison (same prompts / same `max_new_tokens`):

```bash
MODEL_PATH=/path/to/gpt-oss-20b \
DRAFT_PATH=harmony/cuda-norm/artifacts/dflash_draft_ckpts/<ckpt_name> \
PROMPT_LEN=1024 MAX_NEW_TOKENS=2048 NUM_PROMPTS=32 \
bash harmony/cuda-norm/scripts/tpu_sglang_jax_bench_dflash_vs_base.sh
```

Notes:

- The benchmark uses `--grammar-backend none` to avoid optional constrained-decoding deps.
- It reads `HF_TOKEN` from `harmony/cuda-norm/.env` automatically if present (and not already set).
