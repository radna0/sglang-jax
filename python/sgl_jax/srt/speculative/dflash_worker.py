from __future__ import annotations

import logging

from sgl_jax.srt.managers.tp_worker import ModelWorker, ModelWorkerBatch
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class DFlashWorker(ModelWorker):
    """DFLASH worker placeholder for SGLang‑JAX (TPU).

    The upstream DFLASH implementation (PR #16818) targets the PyTorch GPU runtime
    (`sglang.srt.*`). SGLang TPU support lives in the separate JAX runtime
    (`sglang-jax`) and currently only ships EAGLE/EAGLE3 speculative decoding.

    We wire the scheduler + algorithm enum first so we can implement DFLASH
    incrementally without breaking the server args layer.
    """

    def __init__(self, server_args, target_worker: ModelWorker):
        self.server_args = server_args
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # NOTE: We intentionally do NOT set up a draft model runner yet. DFLASH in
        # SGLang‑JAX will mirror the PyTorch PR (#16818) semantics:
        # - capture target hidden features (K layers) during TARGET_VERIFY
        # - run a draft encoder-only model over (context + block) to propose tokens
        # - verify with TARGET_VERIFY logits and accept_len logic
        # - materialize/commit/rollback draft KV cache

    def forward_batch_speculative_generation(self, model_worker_batch: ModelWorkerBatch):
        raise NotImplementedError(
            "DFLASH speculative decoding is not implemented yet in SGLang‑JAX. "
            "Scaffolding is present (server args + state pytrees), but the worker "
            "state machine and draft model are pending."
        )
