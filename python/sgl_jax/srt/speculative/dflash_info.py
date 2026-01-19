from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

if TYPE_CHECKING:
    from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode


@register_pytree_node_class
@dataclass
class DFlashDraftInput:
    """Per-batch DFLASH draft state (spec-v1, non-overlap) for SGLang-JAX.

    This mirrors the upstream SGLang (PyTorch) DFlash worker’s state:
    - `verified_id`: current token per request (the anchor for the next draft block)
    - `target_hidden`: flattened context features for tokens to materialize into draft KV
        shape: [sum(ctx_lens), K * hidden]
    - `ctx_lens_cpu`: per-request context-feature lengths (CPU list)
    - `draft_seq_lens_cpu`: how many tokens already materialized in draft KV (CPU list)
    """

    verified_id: Any | None = None  # typically np.ndarray[int32] shape [bs]
    target_hidden: jax.Array | None = None  # [sum(ctx_lens), K*hidden] (device)
    ctx_lens_cpu: list[int] | None = None
    draft_seq_lens_cpu: list[int] | None = None

    def tree_flatten(self):
        children = (self.verified_id, self.target_hidden)
        aux = {
            "ctx_lens_cpu": self.ctx_lens_cpu,
            "draft_seq_lens_cpu": self.draft_seq_lens_cpu,
        }
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.verified_id = children[0]
        obj.target_hidden = children[1]
        obj.ctx_lens_cpu = aux_data.get("ctx_lens_cpu", None)
        obj.draft_seq_lens_cpu = aux_data.get("draft_seq_lens_cpu", None)
        return obj

    def filter_batch(self, new_indices: jax.Array, has_been_filtered: bool = True):
        keep = np.asarray(jax.device_get(new_indices), dtype=np.int32).tolist()
        if self.verified_id is not None:
            vid = np.asarray(self.verified_id)
            self.verified_id = vid[keep]

        if self.ctx_lens_cpu is not None:
            old_ctx = list(self.ctx_lens_cpu)
            self.ctx_lens_cpu = [int(old_ctx[i]) for i in keep]

        if self.draft_seq_lens_cpu is not None:
            old_dsl = list(self.draft_seq_lens_cpu)
            self.draft_seq_lens_cpu = [int(old_dsl[i]) for i in keep]

        if self.target_hidden is None:
            return
        if not self.ctx_lens_cpu:
            self.target_hidden = self.target_hidden[:0]
            return

        # Re-pack flattened [sum(ctx_lens), ...] by kept requests.
        old_ctx_lens = old_ctx if "old_ctx" in locals() else list(self.ctx_lens_cpu)
        offsets = [0]
        for ln in old_ctx_lens:
            offsets.append(offsets[-1] + int(ln))
        segs = []
        for idx in keep:
            ln = int(old_ctx_lens[idx])
            if ln <= 0:
                continue
            segs.append(self.target_hidden[offsets[idx] : offsets[idx + 1]])
        self.target_hidden = jnp.concatenate(segs, axis=0) if segs else self.target_hidden[:0]

    def merge_batch(self, spec_info: "DFlashDraftInput"):
        if spec_info is None:
            return
        if self.verified_id is None:
            self.verified_id = spec_info.verified_id
        elif spec_info.verified_id is not None:
            self.verified_id = np.concatenate(
                [np.asarray(self.verified_id), np.asarray(spec_info.verified_id)], axis=0
            )

        if self.ctx_lens_cpu is None:
            self.ctx_lens_cpu = list(spec_info.ctx_lens_cpu or [])
        else:
            self.ctx_lens_cpu.extend(list(spec_info.ctx_lens_cpu or []))

        if self.draft_seq_lens_cpu is None:
            self.draft_seq_lens_cpu = list(spec_info.draft_seq_lens_cpu or [])
        else:
            self.draft_seq_lens_cpu.extend(list(spec_info.draft_seq_lens_cpu or []))

        if self.target_hidden is None:
            self.target_hidden = spec_info.target_hidden
        elif spec_info.target_hidden is not None:
            self.target_hidden = jnp.concatenate([self.target_hidden, spec_info.target_hidden], axis=0)


@register_pytree_node_class
@dataclass
class DFlashVerifyInput:
    """Inputs for a target verify forward in DFLASH (spec-v1).

    This object is stored on `ModelWorkerBatch.spec_info` and is consumed by the
    attention backend (custom_mask) and by the speculative worker (accept logic).
    """

    draft_token: jax.Array | None = None  # [bs * block]
    positions: jax.Array | None = None  # [bs * block]
    custom_mask: jax.Array | None = None  # flattened allow-mask (int32 preferred)
    draft_token_num: int = 0
    capture_hidden_mode: "CaptureHiddenMode" | Any = None

    def tree_flatten(self):
        # Keep `draft_token_num` static in aux_data. Converting tracers to Python
        # ints inside tree_unflatten breaks JIT (ConcretizationTypeError).
        children = (self.draft_token, self.positions, self.custom_mask)
        aux_data = {
            "capture_hidden_mode": self.capture_hidden_mode,
            "draft_token_num": int(self.draft_token_num),
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.capture_hidden_mode = aux_data["capture_hidden_mode"]
        obj.draft_token = children[0]
        obj.positions = children[1]
        obj.custom_mask = children[2]
        obj.draft_token_num = int(aux_data.get("draft_token_num", 0) or 0)
        return obj

    def filter_batch(self, new_indices: jax.Array, has_been_filtered: bool = True):
        # For spec-v1 we keep this simple; once we implement full DFLASH verify,
        # we will need to filter flattened blocks consistently.
        pass

    def merge_batch(self, spec_info: "DFlashVerifyInput"):
        raise NotImplementedError("DFLASH verify batching is not implemented yet.")

    def prepare_for_verify(self, model_worker_batch: Any):
        # Local import to avoid circular imports during module initialization.
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

        if model_worker_batch.forward_mode.is_idle():
            return

        model_worker_batch.input_ids = self.draft_token
        model_worker_batch.positions = self.positions
        model_worker_batch.forward_mode = ForwardMode.TARGET_VERIFY
        model_worker_batch.spec_info = self
        model_worker_batch.capture_hidden_mode = self.capture_hidden_mode

    def verify_greedy(
        self,
        *,
        batch_size: int,
        logits_output: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Greedy DFLASH verification (pure tensor logic, no req mutation).

        Returns:
          accept_len: int32[bs] accepted draft tokens (excluding anchor token)
          commit_lens: int32[bs] number of tokens to append to draft-KV (anchor + accepted)
          bonus: int32[bs] next token to append (becomes the new verified_id)
        """
        from sgl_jax.srt.speculative.dflash_utils import compute_dflash_accept_len_and_bonus

        bs = int(batch_size)
        block = int(self.draft_token_num)
        if bs <= 0 or block <= 0:
            raise ValueError(f"Invalid verify shapes: bs={bs}, block={block}.")
        if self.draft_token is None:
            raise ValueError("DFLASH verify requires draft_token.")

        candidates = jnp.asarray(self.draft_token, dtype=jnp.int32).reshape(bs, block)
        logits = getattr(logits_output, "next_token_logits", None)
        if logits is None:
            raise ValueError("DFLASH verify requires logits_output.next_token_logits.")
        target_predict = jnp.argmax(logits, axis=-1).astype(jnp.int32).reshape(bs, block)

        accept_len, bonus = compute_dflash_accept_len_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        commit_lens = (accept_len + 1).astype(jnp.int32)
        return accept_len, commit_lens, bonus

    def extract_commit_target_hidden(
        self,
        *,
        batch_size: int,
        logits_output: Any,
        commit_lens: jax.Array,
    ) -> jax.Array:
        """Extract flattened target hidden context features for committed tokens.

        We materialize draft KV for the anchor token and accepted draft tokens
        (not the bonus token). These correspond to the first `commit_lens[i]`
        positions of the verify block per request.
        """
        bs = int(batch_size)
        block = int(self.draft_token_num)
        hidden = getattr(logits_output, "hidden_states", None)
        if hidden is None:
            raise ValueError("DFLASH verify requires logits_output.hidden_states (capture_hidden_mode=FULL).")

        hidden = jnp.asarray(hidden).reshape(bs, block, -1)
        commit_lens = np.asarray(jax.device_get(commit_lens), dtype=np.int32).reshape(bs)

        segs = []
        for i in range(bs):
            ln = int(commit_lens[i])
            if ln <= 0:
                continue
            segs.append(hidden[i, :ln, :])
        if segs:
            return jnp.concatenate(segs, axis=0)
        return hidden.reshape(-1, hidden.shape[-1])[:0]
