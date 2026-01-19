from __future__ import annotations

import logging
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.scheduler import GenerationBatchResult
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.mem_cache.common import alloc_token_slots
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sgl_jax.srt.speculative.dflash_utils import (
    resolve_dflash_mask_token_id,
    resolve_dflash_target_layer_ids,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


def _build_cache_loc_flat(
    *,
    req_to_token: np.ndarray,
    req_pool_indices: np.ndarray,
    seq_lens: np.ndarray,
    pad_to: int | None = None,
) -> np.ndarray:
    if len(seq_lens) == 0:
        if pad_to is None:
            return np.array([], dtype=np.int32)
        return np.zeros((int(pad_to),), dtype=np.int32)
    out: list[np.ndarray] = []
    for req_pool_idx, seq_len in zip(req_pool_indices.tolist(), seq_lens.tolist(), strict=True):
        ln = int(seq_len)
        if ln <= 0:
            continue
        out.append(req_to_token[int(req_pool_idx), :ln].astype(np.int32, copy=False))
    flat = np.concatenate(out, axis=0) if out else np.array([], dtype=np.int32)
    if pad_to is None:
        return flat
    pad_to_int = int(pad_to)
    if flat.shape[0] > pad_to_int:
        raise ValueError(f"cache_loc overflows pad_to: need={flat.shape[0]} pad_to={pad_to_int}")
    padded = np.zeros((pad_to_int,), dtype=np.int32)
    if flat.shape[0]:
        padded[: flat.shape[0]] = flat
    return padded


class DFlashWorker(ModelWorker):
    """DFLASH speculative decoding worker (spec-v1) for SGLang-JAX (TPU).

    This is a close semantic port of PR #16818 into the JAX runtime:
    - Keep one "current" token per request in `DFlashDraftInput.verified_id` (not in target KV).
    - Each verify step commits (current + accepted drafts) into target KV and produces a new
      current token ("bonus") for the next iteration.
    - Draft KV is materialized from captured target hidden features (K layers) so the draft
      model never recomputes the prefix.
    """

    def __init__(self, server_args, target_worker: ModelWorker):
        self.server_args = server_args
        self.target_worker = target_worker
        self.page_size = int(server_args.page_size)
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        if not self.speculative_algorithm.is_dflash():
            raise ValueError("DFlashWorker requires speculative_algorithm=DFLASH.")

        # DFLASH v1: greedy-only and token-level KV.
        if self.page_size != 1:
            raise ValueError("DFLASH currently requires --page-size 1.")

        self.block_size = int(server_args.speculative_num_draft_tokens)
        if self.block_size <= 1:
            raise ValueError("DFLASH requires --speculative-num-draft-tokens >= 2.")
        # Scheduler expects this attribute for speculative decoding bookkeeping.
        self.speculative_num_draft_tokens = self.block_size

        # Draft worker/model runner (separate KV cache pool).
        super().__init__(
            server_args,
            target_worker.mesh,
            req_to_token_pool=None,
            is_draft_worker=True,
        )
        self.draft_model_runner = self.model_runner

        # Share embed + lm_head weights from the target.
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        self.draft_model_runner.model.set_embed_and_head(embed, head)

        # Enable JIT for the draft model runner too (ModelWorker skips initialize_jit for draft workers).
        self.draft_model_runner.initialize_jit()

        # Configure target hidden capture (K selected target layers).
        target_num_layers = int(self.target_worker.model_runner.model_config.num_hidden_layers)
        draft_hf_cfg = self.draft_model_runner.model_config.hf_config
        draft_num_layers = int(getattr(draft_hf_cfg, "num_hidden_layers", 0) or 0)
        if draft_num_layers <= 0:
            draft_num_layers = int(self.draft_model_runner.model_config.num_hidden_layers)

        target_layer_ids = resolve_dflash_target_layer_ids(
            draft_hf_config=draft_hf_cfg,
            target_num_layers=target_num_layers,
            default_num_context_features=draft_num_layers,
        )

        if not hasattr(self.target_worker.model_runner.model, "set_dflash_layers_to_capture"):
            raise RuntimeError(
                "DFLASH requires the target model to support `set_dflash_layers_to_capture(layer_ids)`."
            )
        self.target_worker.model_runner.model.set_dflash_layers_to_capture(target_layer_ids)
        self.num_context_features = len(target_layer_ids)
        self.target_hidden_size = int(self.target_worker.model_runner.model_config.hidden_size)

        self.mask_token_id = resolve_dflash_mask_token_id(draft_hf_config=draft_hf_cfg)

        if os.environ.get("SGLANG_DFLASH_VERBOSE", "0").lower() in ("1", "true", "yes", "y", "on"):
            try:
                cfg = getattr(draft_hf_cfg, "dflash_config", None)
                cfg_dict = dict(cfg) if cfg is not None and not isinstance(cfg, dict) else (cfg or {})
            except Exception:
                cfg_dict = {}
            logger.info(
                "[DFLASH] init: block_size=%s target_hidden_size=%s K=%s mask_token_id=%s",
                int(self.block_size),
                int(self.target_hidden_size),
                int(self.num_context_features),
                int(self.mask_token_id),
            )
            logger.info(
                "[DFLASH] init: target_layer_ids=%s draft_model_path=%s",
                list(target_layer_ids),
                getattr(server_args, "speculative_draft_model_path", None),
            )
            if cfg_dict:
                logger.info("[DFLASH] init: draft.dflash_config=%s", cfg_dict)

        # Draft-side KV bookkeeping uses a separate req_to_token table keyed by the SAME
        # req_pool_idx values as the target scheduler, but mapping into the draft KV pool.
        target_req_to_token = self.target_worker.model_runner.req_to_token_pool.req_to_token
        self._draft_req_to_token = np.zeros_like(target_req_to_token, dtype=np.int32)

        draft_hidden_size = int(self.draft_model_runner.model_config.hidden_size)
        if draft_hidden_size != self.target_hidden_size:
            raise ValueError(
                "DFLASH requires draft and target to share hidden_size. "
                f"got draft_hidden_size={draft_hidden_size} target_hidden_size={self.target_hidden_size}"
            )
        if int(getattr(self.draft_model_runner.model, "num_context_features", self.num_context_features)) != int(
            self.num_context_features
        ):
            raise ValueError(
                "DFLASH draft model num_context_features must match target hidden capture. "
                f"got draft_num_context_features={getattr(self.draft_model_runner.model, 'num_context_features', None)} "
                f"target_num_context_features={self.num_context_features}"
            )

    def run_spec_decode_precompile(self):
        # DFLASH is extremely sensitive to JAX compilation overhead because it
        # introduces additional forward modes (draft propose + target verify).
        # Without precompile, each decode batch shape (bs changes as requests
        # finish) can trigger a fresh compilation, destroying throughput.
        self._precompile_spec_extend()
        self._precompile_spec_decode()

    def _precompile_spec_extend(self) -> None:
        start = time.perf_counter()
        bs_max = int(self.precompile_bs_paddings[-1])
        cache_loc_pad = int(self.precompile_cache_loc_paddings[-1])
        max_req_len = int(self.max_req_len)
        logger.info(
            "[DFLASH_SPEC_EXTEND] Precompile bs=%s token_paddings=%s cache_loc_pad=%s",
            bs_max,
            self.precompile_token_paddings,
            cache_loc_pad,
        )
        for num_tokens in self.precompile_token_paddings:
            num_tokens = int(num_tokens)
            if bs_max > num_tokens:
                continue
            # Never precompile shapes that exceed the engine's configured maximum
            # request length. Doing so overflows the per-request req_to_token
            # tables (and can trigger very large TPU compilations).
            if num_tokens > max_req_len:
                continue
            batch = self.generate_model_worker_batch(
                bs_max,
                num_tokens,
                ForwardMode.EXTEND,
                cache_loc_pad,
                do_penalties=False,
                speculative_algotithm=self.speculative_algorithm,
            )
            # Precompile should not request logprobs. The DFLASH worker does not
            # support logprob outputs yet, and the default ModelWorkerBatch
            # path can set return_output_logprob_only=True.
            batch.return_logprob = False
            batch.return_output_logprob_only = False
            draft_alloc = self.draft_model_runner.token_to_kv_pool_allocator
            draft_state = draft_alloc.backup_state()
            try:
                self.forward_batch_speculative_generation(batch)
            finally:
                draft_alloc.restore_state(draft_state)
                # Clear draft req->token mapping for the precompile req ids.
                self._draft_req_to_token[:bs_max, :] = 0
        logger.info("[DFLASH_SPEC_EXTEND] Done in %.1fs", time.perf_counter() - start)

    def _precompile_spec_decode(self) -> None:
        start = time.perf_counter()
        logger.info(
            "[DFLASH_SPEC_DECODE] Precompile bs_paddings=%s block_size=%s",
            self.precompile_bs_paddings,
            int(self.block_size),
        )

        feat = int(self.num_context_features * self.target_hidden_size)
        dummy_target_hidden = jnp.empty(
            (0, feat),
            dtype=jnp.bfloat16 if str(self.server_args.dtype) == "bfloat16" else jnp.float32,
        )

        for bs in self.precompile_bs_paddings:
            bs = int(bs)
            aligned_cache_loc_size = (
                (bs * int(self.max_req_len) + self.page_size - 1) // self.page_size
            ) * self.page_size
            batch = self.generate_model_worker_batch(
                bs,
                bs,
                ForwardMode.DECODE,
                aligned_cache_loc_size,
                do_penalties=False,
                speculative_algotithm=self.speculative_algorithm,
            )
            # Precompile should not request logprobs. The DFLASH worker does not
            # support logprob outputs yet, and the default ModelWorkerBatch
            # path can set return_output_logprob_only=True.
            batch.return_logprob = False
            batch.return_output_logprob_only = False
            batch.spec_info = DFlashDraftInput(
                verified_id=np.ones((bs,), dtype=np.int32),
                target_hidden=dummy_target_hidden,
                ctx_lens_cpu=[0] * bs,
                draft_seq_lens_cpu=[0] * bs,
            )
            draft_alloc = self.draft_model_runner.token_to_kv_pool_allocator
            target_alloc = self.target_worker.model_runner.token_to_kv_pool_allocator
            draft_state = None
            target_state = None
            try:
                draft_state = draft_alloc.backup_state()
            except NotImplementedError:
                draft_state = None
            try:
                target_state = target_alloc.backup_state()
            except NotImplementedError:
                target_state = None
            try:
                self.forward_batch_speculative_generation(batch)
            finally:
                if draft_state is not None:
                    draft_alloc.restore_state(draft_state)
                if target_state is not None:
                    target_alloc.restore_state(target_state)
                self._draft_req_to_token[:bs, :] = 0

        logger.info("[DFLASH_SPEC_DECODE] Done in %.1fs", time.perf_counter() - start)

    def _draft_kv_assign(
        self,
        *,
        req_pool_indices: np.ndarray,
        start_offsets: np.ndarray,
        token_ids_flat: np.ndarray,
        block_size: int,
    ) -> None:
        # token_ids_flat is the *KV indices* (not token ids), flattened [bs * block]
        # Start offsets: per request, where to start writing in req_to_token.
        bs = int(len(req_pool_indices))
        for i in range(bs):
            req_pool_idx = int(req_pool_indices[i])
            start = int(start_offsets[i])
            end = start + int(block_size)
            s = i * int(block_size)
            e = s + int(block_size)
            self._draft_req_to_token[req_pool_idx, start:end] = token_ids_flat[s:e]

    def _draft_kv_clear(
        self,
        *,
        req_pool_indices: np.ndarray,
        start_offsets: np.ndarray,
        block_size: int,
    ) -> None:
        bs = int(len(req_pool_indices))
        for i in range(bs):
            req_pool_idx = int(req_pool_indices[i])
            start = int(start_offsets[i])
            end = start + int(block_size)
            self._draft_req_to_token[req_pool_idx, start:end] = 0

    def _append_target_hidden_to_draft_kv(
        self,
        *,
        req_pool_indices: np.ndarray,
        draft_input: DFlashDraftInput,
    ) -> None:
        bs = int(len(req_pool_indices))
        if bs == 0:
            return

        if draft_input.target_hidden is None:
            return

        if draft_input.ctx_lens_cpu is None or draft_input.draft_seq_lens_cpu is None:
            raise RuntimeError("DFLASH draft state missing ctx_lens_cpu/draft_seq_lens_cpu.")
        if len(draft_input.ctx_lens_cpu) != bs or len(draft_input.draft_seq_lens_cpu) != bs:
            raise RuntimeError(
                "DFLASH draft state length mismatch: "
                f"ctx_lens={len(draft_input.ctx_lens_cpu)} draft_seq_lens={len(draft_input.draft_seq_lens_cpu)} bs={bs}"
            )

        total_ctx = int(sum(int(x) for x in draft_input.ctx_lens_cpu))
        if total_ctx <= 0:
            return

        if int(getattr(draft_input.target_hidden, "shape", (0,))[0]) != total_ctx:
            raise RuntimeError(
                "DFLASH target_hidden packing mismatch: "
                f"target_hidden.shape[0]={int(draft_input.target_hidden.shape[0])} "
                f"sum(ctx_lens_cpu)={total_ctx}"
            )

        # Allocate draft KV slots for the tokens we are materializing, and write req->token mapping.
        ctx_cache_loc_chunks: list[np.ndarray] = []
        ctx_positions_chunks: list[np.ndarray] = []
        new_draft_seq_lens_cpu: list[int] = []

        for i, (cache_len, ctx_len) in enumerate(
            zip(draft_input.draft_seq_lens_cpu, draft_input.ctx_lens_cpu, strict=True)
        ):
            cache_len_i = int(cache_len)
            ctx_len_i = int(ctx_len)
            new_draft_seq_lens_cpu.append(cache_len_i + ctx_len_i)
            if ctx_len_i <= 0:
                continue

            loc = self.draft_model_runner.token_to_kv_pool_allocator.alloc(ctx_len_i)
            if loc is None:
                raise RuntimeError(f"DFLASH draft OOM allocating ctx_len={ctx_len_i}.")
            loc = np.asarray(loc, dtype=np.int32)

            req_pool_idx = int(req_pool_indices[i])
            self._draft_req_to_token[req_pool_idx, cache_len_i : cache_len_i + ctx_len_i] = loc

            ctx_cache_loc_chunks.append(loc)
            ctx_positions_chunks.append(np.arange(cache_len_i, cache_len_i + ctx_len_i, dtype=np.int32))

        ctx_cache_loc = (
            np.concatenate(ctx_cache_loc_chunks, axis=0)
            if ctx_cache_loc_chunks
            else np.array([], dtype=np.int32)
        )
        ctx_positions = (
            np.concatenate(ctx_positions_chunks, axis=0)
            if ctx_positions_chunks
            else np.array([], dtype=np.int32)
        )

        if ctx_cache_loc.size == 0:
            return

        # Project target hidden features down to the draft hidden size, then materialize K/V per draft layer.
        ctx_hidden = self.draft_model_runner.model.project_target_hidden(draft_input.target_hidden)

        ctx_positions_dev = jax.device_put(jnp.asarray(ctx_positions, dtype=jnp.int32))
        ctx_cache_loc_dev = jax.device_put(jnp.asarray(ctx_cache_loc, dtype=jnp.int32))

        for layer in self.draft_model_runner.model.layers:
            k, v = layer.self_attn.kv_proj_only(hidden_states=ctx_hidden)
            k = layer.self_attn.apply_k_rope(positions=ctx_positions_dev, k=k)
            self.draft_model_runner.token_to_kv_pool.set_kv_buffer(
                int(layer.layer_id),
                ctx_cache_loc_dev,
                k,
                v,
                is_decode=True,
            )

        draft_input.target_hidden = draft_input.target_hidden[:0]
        draft_input.ctx_lens_cpu = [0] * bs
        draft_input.draft_seq_lens_cpu = new_draft_seq_lens_cpu

    def _append_target_hidden_block_to_draft_kv(
        self,
        *,
        req_pool_indices: np.ndarray,
        kv_seq_lens: np.ndarray,
        commit_lens: np.ndarray,
        target_hidden_block: jax.Array,
    ) -> None:
        """Commit a fixed-size (bs×block_size) verify block into the draft KV cache.

        This is the TPU decode fast-path. It avoids dynamic shapes from
        `sum(commit_lens)` by always materializing K/V for the full verify block
        and masking out uncommitted positions (KV index 0).
        """
        bs = int(len(req_pool_indices))
        if bs == 0:
            return

        block = int(self.block_size)
        if int(getattr(target_hidden_block, "shape", (0,))[0]) != bs * block:
            raise RuntimeError(
                "DFLASH target_hidden_block shape mismatch: "
                f"target_hidden_block.shape[0]={int(target_hidden_block.shape[0])} expected={bs * block}"
            )

        # Allocate draft KV slots for the whole block, then free uncommitted slots.
        alloc = self.draft_model_runner.token_to_kv_pool_allocator
        block_cache_loc = alloc.alloc(bs * block)
        if block_cache_loc is None:
            raise RuntimeError(f"DFLASH draft OOM allocating commit block: bs={bs} block={block}.")
        block_cache_loc = np.asarray(block_cache_loc, dtype=np.int32).reshape(bs, block)

        positions = (
            kv_seq_lens.reshape(bs, 1) + np.arange(block, dtype=np.int32).reshape(1, block)
        ).astype(np.int32)

        masked_cache_loc = block_cache_loc.copy()
        for i in range(bs):
            keep = int(commit_lens[i])
            if keep < 0 or keep > block:
                raise RuntimeError(f"DFLASH invalid commit_len={keep} (block={block})")
            if keep < block:
                alloc.free(block_cache_loc[i, keep:])
                masked_cache_loc[i, keep:] = 0

        # Update req->token mapping for committed tokens only.
        for i in range(bs):
            keep = int(commit_lens[i])
            if keep <= 0:
                continue
            req_pool_idx = int(req_pool_indices[i])
            start = int(kv_seq_lens[i])
            self._draft_req_to_token[req_pool_idx, start : start + keep] = block_cache_loc[i, :keep]

        # Project target hidden features down to the draft hidden size, then materialize K/V per draft layer.
        ctx_hidden = self.draft_model_runner.model.project_target_hidden(target_hidden_block)
        ctx_positions_dev = jax.device_put(jnp.asarray(positions.reshape(-1), dtype=jnp.int32))
        ctx_cache_loc_dev = jax.device_put(jnp.asarray(masked_cache_loc.reshape(-1), dtype=jnp.int32))

        for layer in self.draft_model_runner.model.layers:
            k, v = layer.self_attn.kv_proj_only(hidden_states=ctx_hidden)
            k = layer.self_attn.apply_k_rope(positions=ctx_positions_dev, k=k)
            self.draft_model_runner.token_to_kv_pool.set_kv_buffer(
                int(layer.layer_id),
                ctx_cache_loc_dev,
                k,
                v,
                is_decode=True,
            )

    def _draft_block_propose(
        self,
        *,
        req_pool_indices: np.ndarray,
        kv_seq_lens: np.ndarray,
        verified_id: np.ndarray,
    ) -> np.ndarray:
        bs = int(len(req_pool_indices))
        block = int(self.block_size)

        block_ids = np.full((bs, block), self.mask_token_id, dtype=np.int32)
        block_ids[:, 0] = verified_id.astype(np.int32, copy=False)

        positions = (
            kv_seq_lens.reshape(bs, 1) + np.arange(block, dtype=np.int32).reshape(1, block)
        ).astype(np.int32)

        # Allocate draft KV slots for the whole block temporarily, run the draft model, then rollback.
        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        alloc_state = allocator.backup_state()
        try:
            block_cache_loc = allocator.alloc(bs * block)
            if block_cache_loc is None:
                raise RuntimeError(f"DFLASH draft OOM allocating block bs={bs} block={block}.")
            block_cache_loc = np.asarray(block_cache_loc, dtype=np.int32)
            self._draft_kv_assign(
                req_pool_indices=req_pool_indices,
                start_offsets=kv_seq_lens,
                token_ids_flat=block_cache_loc,
                block_size=block,
            )

            cache_loc_pad = (
                (bs * int(self.max_req_len) + self.page_size - 1) // self.page_size
            ) * self.page_size
            cache_loc = _build_cache_loc_flat(
                req_to_token=self._draft_req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=kv_seq_lens + block,
                pad_to=cache_loc_pad,
            )

            spec_info = DFlashVerifyInput(
                draft_token_num=block,
                custom_mask=None,
            )

            batch = ModelWorkerBatch(
                bid=0,
                forward_mode=ForwardMode.TARGET_VERIFY,
                input_ids=block_ids.reshape(-1).astype(np.int32),
                real_input_ids_len=bs * block,
                seq_lens=kv_seq_lens.astype(np.int32),
                out_cache_loc=block_cache_loc,
                req_pool_indices=req_pool_indices.astype(np.int32),
                sampling_info=None,
                positions=positions.reshape(-1).astype(np.int32),
                extend_start_loc=(np.arange(bs, dtype=np.int32) * int(block)).astype(np.int32),
                cache_loc=cache_loc.astype(np.int32),
                return_logprob=False,
                return_output_logprob_only=False,
                top_logprobs_nums=None,
                token_ids_logprobs=None,
                extend_prefix_lens=kv_seq_lens.astype(np.int32),
                extend_seq_lens=np.full((bs,), int(block), dtype=np.int32),
                extend_logprob_start_lens=None,
                extend_input_logprob_token_ids=None,
                lora_ids=None,
                real_bs=bs,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                launch_done=None,
                spec_info=spec_info,
                spec_algorithm=self.speculative_algorithm,
                tree_cache=None,
            )

            forward_metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(batch)
            self.draft_model_runner.attn_backend.forward_metadata = forward_metadata

            forward_batch = ForwardBatch.init_new(batch, self.draft_model_runner)
            logits_output, _ = self.draft_model_runner.forward(
                forward_batch,
                logits_metadata=LogitsMetadata(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    capture_hidden_mode=CaptureHiddenMode.NULL,
                ),
            )

            logits = jnp.asarray(logits_output.next_token_logits).reshape(bs, block, -1)
            next_ids = jnp.argmax(logits[:, :-1, :], axis=-1).astype(jnp.int32)  # [bs, block-1]
            return np.asarray(jax.device_get(next_ids), dtype=np.int32)
        finally:
            allocator.restore_state(alloc_state)
            self._draft_kv_clear(
                req_pool_indices=req_pool_indices,
                start_offsets=kv_seq_lens,
                block_size=block,
            )

    def forward_batch_speculative_generation(self, model_worker_batch: ModelWorkerBatch):
        profile = os.environ.get("SGLANG_DFLASH_PROFILE", "0").lower() in ("1", "true", "yes", "y")
        strict_bonus_decode = os.environ.get("SGLANG_DFLASH_STRICT_BONUS_DECODE", "0").lower() in (
            "1",
            "true",
            "yes",
            "y",
        )
        if model_worker_batch.return_logprob or model_worker_batch.return_output_logprob_only:
            raise NotImplementedError("DFLASH does not support logprobs yet in sglang-jax.")

        if model_worker_batch.forward_mode.is_extend():
            # Target prefill: produce initial current token, and materialize prompt tokens into draft KV.
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            # DFLASH v1 is greedy-only. Avoid going through the generic sampler
            # here (it expects per-request temperature arrays with shapes that
            # may differ under speculative bookkeeping) and compute argmax
            # directly from logits instead.
            logits_output, _, cache_miss_count = self.target_worker.forward_batch_generation(
                model_worker_batch, skip_sample=True, sampling_metadata=None
            )
            next_token_ids_device = jnp.argmax(logits_output.next_token_logits, axis=-1).astype(jnp.int32)
            next_token_ids = np.asarray(jax.device_get(next_token_ids_device), dtype=np.int32)[
                : model_worker_batch.real_bs
            ]
            if logits_output.hidden_states is None:
                raise RuntimeError("DFLASH prefill requires target hidden capture (CaptureHiddenMode.FULL).")
            if int(logits_output.hidden_states.shape[-1]) != int(
                self.num_context_features * self.target_hidden_size
            ):
                raise RuntimeError(
                    "DFLASH target hidden feature dim mismatch on prefill: "
                    f"hidden_states.shape[-1]={int(logits_output.hidden_states.shape[-1])} "
                    f"expected={int(self.num_context_features * self.target_hidden_size)} "
                    f"(K={self.num_context_features}, hidden={self.target_hidden_size})"
                )

            # Materialize prompt tokens (not including the new current token) into the draft KV cache.
            ctx_lens_cpu = (
                model_worker_batch.extend_seq_lens.tolist()
                if model_worker_batch.extend_seq_lens is not None
                else [0] * model_worker_batch.real_bs
            )
            draft_seq_lens_cpu = (
                model_worker_batch.extend_prefix_lens.tolist()
                if model_worker_batch.extend_prefix_lens is not None
                else [0] * model_worker_batch.real_bs
            )
            # During JAX precompile we may use dummy batches where extend_seq_lens
            # does not reflect the true packed hidden_states layout. In that
            # case, fall back to a deterministic "packed-into-first-request"
            # convention so KV materialization can still compile without
            # requiring real prompt semantics.
            try:
                hs_rows = int(logits_output.hidden_states.shape[0])
                total_ctx = int(sum(int(x) for x in ctx_lens_cpu[: model_worker_batch.real_bs]))
                if hs_rows != total_ctx:
                    ctx_lens_cpu = [hs_rows] + [0] * max(int(model_worker_batch.real_bs) - 1, 0)
                    draft_seq_lens_cpu = [int(draft_seq_lens_cpu[0]) if draft_seq_lens_cpu else 0] + [0] * max(
                        int(model_worker_batch.real_bs) - 1, 0
                    )
            except Exception:
                pass
            draft_input = DFlashDraftInput(
                verified_id=next_token_ids,
                target_hidden=logits_output.hidden_states,
                ctx_lens_cpu=[int(x) for x in ctx_lens_cpu[: model_worker_batch.real_bs]],
                draft_seq_lens_cpu=[int(x) for x in draft_seq_lens_cpu[: model_worker_batch.real_bs]],
            )
            self._append_target_hidden_to_draft_kv(
                req_pool_indices=model_worker_batch.req_pool_indices[: model_worker_batch.real_bs],
                draft_input=draft_input,
            )

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                next_draft_input=draft_input,
                cache_miss_count=int(cache_miss_count),
                allocate_lens=None,
                accept_lens=None,
                bid=model_worker_batch.bid,
                extend_input_len_per_req=None,
                extend_logprob_start_len_per_req=None,
            )

        # Decode: draft -> target verify -> commit current+accepted into KV -> produce bonus as new current.
        bs = int(model_worker_batch.real_bs)
        draft_input = model_worker_batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError("DFLASH decode expects ModelWorkerBatch.spec_info=DFlashDraftInput.")

        t0 = time.perf_counter() if profile else 0.0
        req_pool_indices = model_worker_batch.req_pool_indices[:bs].astype(np.int32, copy=False)
        seq_lens = model_worker_batch.seq_lens[:bs].astype(np.int32, copy=False)
        kv_seq_lens = (seq_lens - 1).astype(np.int32, copy=False)
        kv_seq_lens = np.maximum(kv_seq_lens, 0)

        if draft_input.draft_seq_lens_cpu is None:
            raise RuntimeError("DFLASH decode requires draft_seq_lens_cpu in DFlashDraftInput.")
        if len(draft_input.draft_seq_lens_cpu) != bs:
            raise RuntimeError(
                "DFLASH decode draft state mismatch: "
                f"len(draft_seq_lens_cpu)={len(draft_input.draft_seq_lens_cpu)} bs={bs}"
            )
        for i in range(bs):
            if int(draft_input.draft_seq_lens_cpu[i]) != int(kv_seq_lens[i]):
                raise RuntimeError(
                    "DFLASH KV length mismatch: "
                    f"req_pool_idx={int(req_pool_indices[i])} "
                    f"draft_seq_len={int(draft_input.draft_seq_lens_cpu[i])} "
                    f"kv_seq_len={int(kv_seq_lens[i])} "
                    "(this usually means commit/rollback bookkeeping is broken)."
                )

        # Ensure draft KV contains all committed target tokens before drafting.
        self._append_target_hidden_to_draft_kv(
            req_pool_indices=req_pool_indices,
            draft_input=draft_input,
        )
        t_kv_commit = time.perf_counter() if profile else 0.0

        draft_next = self._draft_block_propose(
            req_pool_indices=req_pool_indices,
            kv_seq_lens=kv_seq_lens,
            verified_id=np.asarray(draft_input.verified_id, dtype=np.int32),
        )
        t_draft = time.perf_counter() if profile else 0.0
        candidates = np.concatenate(
            [np.asarray(draft_input.verified_id, dtype=np.int32).reshape(bs, 1), draft_next],
            axis=1,
        ).astype(np.int32)  # [bs, block]

        # ---- target verify (prefill-style) over the full block
        block = int(self.block_size)
        tree_cache = getattr(model_worker_batch, "tree_cache", None)
        if tree_cache is not None:
            allocator = tree_cache.token_to_kv_pool_allocator
            req_to_token = tree_cache.req_to_token_pool.req_to_token
        else:
            allocator = self.target_worker.model_runner.token_to_kv_pool_allocator
            req_to_token = self.target_worker.model_runner.req_to_token_pool.req_to_token

        out_cache_loc = allocator.alloc(bs * block)
        if out_cache_loc is None:
            raise RuntimeError(f"DFLASH target OOM allocating verify block: bs={bs} block={block}")
        out_cache_loc = np.asarray(out_cache_loc, dtype=np.int32)

        for i, req_pool_idx in enumerate(req_pool_indices.tolist()):
            start = int(kv_seq_lens[i])
            s = i * block
            e = s + block
            req_to_token[int(req_pool_idx), start : start + block] = out_cache_loc[s:e]

        cache_loc_pad = (
            (bs * int(self.target_worker.max_req_len) + self.page_size - 1) // self.page_size
        ) * self.page_size
        cache_loc = _build_cache_loc_flat(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=kv_seq_lens + block,
            pad_to=cache_loc_pad,
        )
        positions = (
            kv_seq_lens.reshape(bs, 1) + np.arange(block, dtype=np.int32).reshape(1, block)
        ).astype(np.int32)

        verify_input = DFlashVerifyInput(
            draft_token=jnp.asarray(candidates.reshape(-1), dtype=jnp.int32),
            positions=jnp.asarray(positions.reshape(-1), dtype=jnp.int32),
            custom_mask=None,
            draft_token_num=block,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

        verify_batch = ModelWorkerBatch(
            bid=model_worker_batch.bid,
            forward_mode=ForwardMode.TARGET_VERIFY,
            input_ids=candidates.reshape(-1).astype(np.int32),
            real_input_ids_len=bs * block,
            seq_lens=kv_seq_lens.astype(np.int32),
            out_cache_loc=out_cache_loc,
            req_pool_indices=req_pool_indices,
            sampling_info=model_worker_batch.sampling_info,
            positions=positions.reshape(-1).astype(np.int32),
            extend_start_loc=(np.arange(bs, dtype=np.int32) * int(block)).astype(np.int32),
            cache_loc=cache_loc.astype(np.int32),
            return_logprob=False,
            return_output_logprob_only=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_prefix_lens=kv_seq_lens.astype(np.int32),
            extend_seq_lens=np.full((bs,), int(block), dtype=np.int32),
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            lora_ids=model_worker_batch.lora_ids,
            real_bs=bs,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            launch_done=None,
            spec_info=verify_input,
            spec_algorithm=self.speculative_algorithm,
            tree_cache=model_worker_batch.tree_cache,
        )

        forward_metadata = self.target_worker.model_runner.attn_backend.get_eagle_forward_metadata(
            verify_batch
        )
        logits_output, _, cache_miss_count = self.target_worker.forward_batch_generation(
            verify_batch,
            skip_sample=True,
            sampling_metadata=None,
            forward_metadata=forward_metadata,
        )
        t_verify = time.perf_counter() if profile else 0.0
        if logits_output.hidden_states is None:
            raise RuntimeError("DFLASH verify requires logits_output.hidden_states (CaptureHiddenMode.FULL).")
        if int(logits_output.hidden_states.shape[-1]) != int(
            self.num_context_features * self.target_hidden_size
        ):
            raise RuntimeError(
                "DFLASH target hidden feature dim mismatch on verify: "
                f"hidden_states.shape[-1]={int(logits_output.hidden_states.shape[-1])} "
                f"expected={int(self.num_context_features * self.target_hidden_size)} "
                f"(K={self.num_context_features}, hidden={self.target_hidden_size})"
            )

        accept_len_jax, commit_lens_jax, bonus_jax = verify_input.verify_greedy(
            batch_size=bs,
            logits_output=logits_output,
        )
        accept_len = np.asarray(jax.device_get(accept_len_jax), dtype=np.int32).reshape(bs)
        commit_lens = np.asarray(jax.device_get(commit_lens_jax), dtype=np.int32).reshape(bs)
        bonus = np.asarray(jax.device_get(bonus_jax), dtype=np.int32).reshape(bs)

        # Free uncommitted verify KV slots: keep first commit_len tokens per request.
        for i in range(bs):
            s = i * block
            out = out_cache_loc[s : s + block]
            keep = int(commit_lens[i])
            if keep < block:
                allocator.free(out[keep:])
                start = int(kv_seq_lens[i]) + keep
                req_to_token[int(req_pool_indices[i]), start : int(kv_seq_lens[i]) + block] = 0

        # Optional quality-first mode: compute the next token ("bonus") using the
        # standard DECODE kernel after committing current+accepted tokens. This
        # reduces numerical drift between TARGET_VERIFY (q_len=block) and the
        # baseline greedy decode path.
        if strict_bonus_decode:
            last_committed = np.zeros((bs,), dtype=np.int32)
            last_out_cache_loc = np.zeros((bs,), dtype=np.int32)
            new_seq_lens = (kv_seq_lens + commit_lens).astype(np.int32)
            for i in range(bs):
                idx = int(max(commit_lens[i], 1) - 1)
                last_committed[i] = int(candidates[i, idx])
                last_out_cache_loc[i] = int(out_cache_loc[i * block + idx])

            decode_cache_loc = _build_cache_loc_flat(
                req_to_token=req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=new_seq_lens,
                pad_to=cache_loc_pad,
            )

            decode_batch = ModelWorkerBatch(
                bid=model_worker_batch.bid,
                forward_mode=ForwardMode.DECODE,
                input_ids=last_committed.astype(np.int32),
                real_input_ids_len=bs,
                seq_lens=new_seq_lens,
                out_cache_loc=last_out_cache_loc.astype(np.int32),
                req_pool_indices=req_pool_indices,
                sampling_info=model_worker_batch.sampling_info,
                positions=(new_seq_lens - 1).astype(np.int32),
                extend_start_loc=np.arange(bs, dtype=np.int32),
                cache_loc=decode_cache_loc.astype(np.int32),
                return_logprob=False,
                return_output_logprob_only=False,
                top_logprobs_nums=None,
                token_ids_logprobs=None,
                extend_prefix_lens=None,
                extend_seq_lens=None,
                extend_logprob_start_lens=None,
                extend_input_logprob_token_ids=None,
                lora_ids=model_worker_batch.lora_ids,
                real_bs=bs,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                launch_done=None,
                spec_info=None,
                spec_algorithm=self.speculative_algorithm,
                tree_cache=model_worker_batch.tree_cache,
            )
            decode_metadata = self.target_worker.model_runner.attn_backend.get_forward_metadata(decode_batch)
            decode_logits, _, _ = self.target_worker.forward_batch_generation(
                decode_batch,
                skip_sample=True,
                sampling_metadata=None,
                forward_metadata=decode_metadata,
            )
            bonus = np.asarray(
                jax.device_get(jnp.argmax(decode_logits.next_token_logits, axis=-1)),
                dtype=np.int32,
            ).reshape(bs)

        # Update draft state: commit (current + accepted) into draft KV, and set new current to bonus.
        #
        # On TPU, using a variable-sized commit buffer (`sum(commit_lens)`) causes
        # recompiles and kills throughput. Materialize the full verify block with
        # a fixed shape (bs×block) and mask out uncommitted positions instead.
        self._append_target_hidden_block_to_draft_kv(
            req_pool_indices=req_pool_indices,
            kv_seq_lens=kv_seq_lens,
            commit_lens=commit_lens,
            target_hidden_block=logits_output.hidden_states,
        )
        # Keep draft-side KV length bookkeeping in sync with the target.
        if draft_input.draft_seq_lens_cpu is not None:
            draft_input.draft_seq_lens_cpu = [
                int(kv_seq_lens[i]) + int(commit_lens[i]) for i in range(bs)
            ]
        draft_input.ctx_lens_cpu = [0] * bs
        draft_input.target_hidden = draft_input.target_hidden[:0] if draft_input.target_hidden is not None else None
        draft_input.verified_id = bonus.astype(np.int32)

        # Output tokens to append this step: accepted drafts + bonus, padded to `block` stride.
        flat_next = np.zeros((bs, block), dtype=np.int32)
        accept_lens_out = np.zeros((bs,), dtype=np.int32)
        for i in range(bs):
            a = int(accept_len[i])
            toks = np.concatenate([candidates[i, 1 : 1 + a], bonus[i : i + 1]], axis=0)
            flat_next[i, : toks.shape[0]] = toks
            accept_lens_out[i] = int(toks.shape[0])

        debug_accept = os.environ.get("SGLANG_DFLASH_DEBUG_ACCEPT", "0").lower() in (
            "1",
            "true",
            "yes",
            "y",
        )
        if debug_accept or profile:
            # `accept_len` excludes the bonus token; `accept_lens_out` includes it.
            mean_a = float(np.mean(accept_len)) if bs else 0.0
            mean_out = float(np.mean(accept_lens_out)) if bs else 0.0
            p50 = float(np.quantile(accept_len, 0.5)) if bs else 0.0
            p90 = float(np.quantile(accept_len, 0.9)) if bs else 0.0
            logger.info(
                "[DFLASH accept] bs=%d block=%d accept_len(mean=%.3f p50=%.1f p90=%.1f) out_tokens(mean=%.3f)",
                bs,
                int(self.block_size),
                mean_a,
                p50,
                p90,
                mean_out,
            )

        if profile:
            # Note: this includes any implicit JAX compilation triggered in these regions.
            logger.info(
                "[DFLASH profile] bs=%d block=%d kv_commit=%.3fs draft=%.3fs verify=%.3fs total=%.3fs",
                bs,
                int(self.block_size),
                max(0.0, t_kv_commit - t0),
                max(0.0, t_draft - t_kv_commit),
                max(0.0, t_verify - t_draft),
                max(0.0, time.perf_counter() - t0),
            )

        return GenerationBatchResult(
            logits_output=None,
            # Keep a flat, padded [bs * block] list here; the scheduler resolves
            # per-request accepted tokens via `accept_lens` (see
            # SchedulerOutputProcessorMixin._resolve_spec_decode_token_ids).
            next_token_ids=flat_next.reshape(-1),
            next_draft_input=draft_input,
            accept_lens=accept_lens_out.astype(np.int32),
            cache_miss_count=int(cache_miss_count),
            allocate_lens=None,
            bid=model_worker_batch.bid,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )
