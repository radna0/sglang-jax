import unittest

import jax.numpy as jnp

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
    static_validate_inputs_fused,
)


class RaggedPagedAttentionKVHeadDimTest(unittest.TestCase):
    def test_kv_cache_head_dim_must_be_128_aligned(self):
        # Minimal shapes: 1 token, 8 Q heads, 1 KV head, head_dim=64.
        q = jnp.zeros((1, 8, 64), dtype=jnp.bfloat16)
        k = jnp.zeros((1, 1, 64), dtype=jnp.bfloat16)
        v = jnp.zeros((1, 1, 64), dtype=jnp.bfloat16)

        kv_lens = jnp.zeros((1,), dtype=jnp.int32)
        page_indices = jnp.zeros((1,), dtype=jnp.int32)
        cu_q_lens = jnp.zeros((2,), dtype=jnp.int32)
        cu_kv_lens = jnp.zeros((2,), dtype=jnp.int32)
        distribution = jnp.zeros((3,), dtype=jnp.int32)

        # Bad: head_dim is unpadded (64).
        kv_cache_fused_bad = jnp.zeros((1, 1, 2, 64), dtype=jnp.bfloat16)
        with self.assertRaises(ValueError):
            static_validate_inputs_fused(
                q,
                k,
                v,
                kv_cache_fused_bad,
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
            )

        # Good: head_dim is padded/aligned to 128.
        kv_cache_fused_ok = jnp.zeros((1, 1, 2, 128), dtype=jnp.bfloat16)
        static_validate_inputs_fused(
            q,
            k,
            v,
            kv_cache_fused_ok,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
        )

