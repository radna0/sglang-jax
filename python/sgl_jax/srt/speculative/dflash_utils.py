from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"
DEFAULT_DFLASH_MASK_TOKEN_ID = 200000


def build_target_layer_ids(num_target_layers: int, num_context_features: int) -> list[int]:
    """Select evenly-spaced target layer indices for DFLASH context features.

    Mirrors the upstream helper in PR #16818.
    """
    if int(num_target_layers) <= 0:
        raise ValueError(f"num_target_layers must be positive, got {num_target_layers}.")
    if int(num_context_features) <= 0:
        raise ValueError(f"num_context_features must be positive, got {num_context_features}.")

    num_target_layers = int(num_target_layers)
    num_context_features = int(num_context_features)

    if num_context_features == 1:
        return [num_target_layers // 2]

    start = 1
    end = num_target_layers - 3
    if end < start:
        raise ValueError(
            "DFLASH layer selection requires num_target_layers >= 4. "
            f"Got num_target_layers={num_target_layers}."
        )

    span = end - start
    return [
        int(round(start + (i * span) / (num_context_features - 1)))
        for i in range(num_context_features)
    ]


def get_dflash_config(config: Any) -> dict:
    cfg = getattr(config, "dflash_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    try:
        return dict(cfg)
    except Exception:
        return {}


def resolve_dflash_target_layer_ids(
    *,
    draft_hf_config: Any,
    target_num_layers: int,
    default_num_context_features: int,
) -> list[int]:
    cfg = get_dflash_config(draft_hf_config)
    layer_ids = cfg.get("target_layer_ids", None)
    if layer_ids is None:
        return build_target_layer_ids(target_num_layers, default_num_context_features)

    if not isinstance(layer_ids, (list, tuple)):
        raise ValueError(
            "DFLASH dflash_config.target_layer_ids must be a list of ints, "
            f"got type={type(layer_ids).__name__}."
        )

    resolved = [int(x) for x in layer_ids]
    if len(resolved) <= 0:
        raise ValueError("DFLASH dflash_config.target_layer_ids must be non-empty.")
    for idx, val in enumerate(resolved):
        if val < 0 or val >= int(target_num_layers):
            raise ValueError(
                "DFLASH target_layer_ids contains an out-of-range layer id. "
                f"target_layer_ids[{idx}]={val}, target_num_layers={int(target_num_layers)}."
            )
    return resolved


def resolve_dflash_mask_token(*, draft_hf_config: Any) -> str:
    cfg = get_dflash_config(draft_hf_config)
    mask_token = cfg.get("mask_token", None)
    if mask_token is None:
        return DEFAULT_DFLASH_MASK_TOKEN
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError(
            "DFLASH dflash_config.mask_token must be a non-empty string, "
            f"got {mask_token!r}."
        )
    return mask_token


def resolve_dflash_mask_token_id(*, draft_hf_config: Any) -> int:
    """Resolve the integer mask token id for DFLASH.

    SGLang-JAX (TPU) does not have access to the tokenizer inside the model worker,
    so we must avoid relying on string->id lookups at runtime.

    Precedence:
      1) `draft_hf_config.dflash_config.mask_token_id`
      2) fallback to GPT-OSS reserved token id 200000 when within vocab size
    """
    cfg = get_dflash_config(draft_hf_config)
    mask_token_id = cfg.get("mask_token_id", None)
    if mask_token_id is not None:
        try:
            mask_token_id = int(mask_token_id)
        except Exception as e:
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be an int, "
                f"got {mask_token_id!r}."
            ) from e
        vocab_size = int(getattr(draft_hf_config, "vocab_size", 0) or 0)
        if vocab_size > 0 and not (0 <= mask_token_id < vocab_size):
            raise ValueError(
                "DFLASH mask_token_id out of vocab range. "
                f"mask_token_id={mask_token_id}, vocab_size={vocab_size}."
            )
        return mask_token_id

    vocab_size = int(getattr(draft_hf_config, "vocab_size", 0) or 0)
    if vocab_size > DEFAULT_DFLASH_MASK_TOKEN_ID:
        return int(DEFAULT_DFLASH_MASK_TOKEN_ID)

    raise ValueError(
        "DFLASH requires a mask token id but none was provided. "
        "Set `dflash_config.mask_token_id` in the draft model config."
    )


@jax.jit
def compute_dflash_accept_len_and_bonus(
    *,
    candidates: jax.Array,  # [bs, block]
    target_predict: jax.Array,  # [bs, block]
) -> tuple[jax.Array, jax.Array]:
    """Compute accept length + bonus token for greedy DFlash verify (spec-v1).

    Mirrors the semantics from the SGLang (PyTorch) DFlash PR:
      - candidates[:, 0] is the current verified token (anchor).
      - Accept draft tokens while candidates[:, 1:] == target_predict[:, :-1] consecutively.
      - `accept_len` is the count of accepted draft tokens (excluding the anchor).
      - `bonus` is the target prediction immediately after the accepted prefix:
          bonus = target_predict[:, accept_len]
        where accept_len is clamped into [0, block-1].
    """
    if candidates.ndim != 2 or target_predict.ndim != 2:
        raise ValueError("candidates/target_predict must be rank-2 arrays [bs, block].")
    if candidates.shape != target_predict.shape:
        raise ValueError(
            f"candidates and target_predict must have the same shape, got {candidates.shape} vs {target_predict.shape}."
        )

    bs, block = candidates.shape
    if block <= 1:
        accept_len = jnp.zeros((bs,), dtype=jnp.int32)
        bonus = target_predict[:, 0].astype(jnp.int32)
        return accept_len, bonus

    # Greedy verify rule (matches upstream SGLang DFLASH):
    # accept while candidates[:, 1:] == target_predict[:, :-1]
    eq = candidates[:, 1:] == target_predict[:, :-1]  # [bs, block-1]

    # Prefix length of True run (first False stops acceptance).
    # Convert to int32 so we can use cumprod trick.
    eq_i32 = eq.astype(jnp.int32)
    prefix_mask = jnp.cumprod(eq_i32, axis=1)  # [bs, block-1] in {0,1}
    accept_len = jnp.sum(prefix_mask, axis=1).astype(jnp.int32)  # [bs]

    # Bonus token is the target prediction at position accept_len (offset within [0..block-1]).
    idx = jnp.clip(accept_len, 0, block - 1).astype(jnp.int32)
    bonus = jnp.take_along_axis(target_predict, idx[:, None], axis=1)[:, 0].astype(jnp.int32)
    return accept_len, bonus
