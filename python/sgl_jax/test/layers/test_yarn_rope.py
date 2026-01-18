import unittest

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.embeddings import YaRNRotaryEmbedding, get_rope


def _yarn_reference_inv_freq(
    *,
    base: float,
    dim: int,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    original_max_position: int,
    truncate: bool,
) -> np.ndarray:
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (dim * np.log(max_position_embeddings / (num_rotations * 2 * np.pi))) / (
            2 * np.log(base)
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low = np.floor(low)
            high = np.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(low, high, dim):
        if low == high:
            high += 0.001
        linear_func = (np.arange(dim, dtype=np.float32) - low) / (high - low)
        return np.clip(linear_func, 0.0, 1.0)

    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float32) / float(dim))
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    low, high = find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_position, truncate
    )
    inv_freq_extrapolation_factor = 1.0 - linear_ramp_factor(low, high, dim // 2)
    inv_freq = (
        inv_freq_interpolation * (1.0 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq.astype(np.float32)


class TestYaRNRope(unittest.TestCase):
    def test_get_rope_returns_yarn(self):
        rope = get_rope(
            head_size=8,
            rotary_dim=8,
            max_position=131072,
            base=150000,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 4096,
                "truncate": False,
            },
            dtype=jnp.bfloat16,
        )
        self.assertIsInstance(rope, YaRNRotaryEmbedding)
        self.assertGreater(rope.attention_factor, 1.0)

    def test_yarn_inv_freq_matches_reference(self):
        rope = get_rope(
            head_size=8,
            rotary_dim=8,
            max_position=131072,
            base=150000,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 4096,
                "truncate": False,
            },
            dtype=jnp.bfloat16,
        )
        ref = _yarn_reference_inv_freq(
            base=150000.0,
            dim=8,
            factor=32.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_max_position=4096,
            truncate=False,
        )
        got = np.asarray(rope._inv_freq_np, dtype=np.float32)
        np.testing.assert_allclose(got, ref, rtol=0, atol=0)

    def test_attention_factor_scales_cos_sin(self):
        rope = get_rope(
            head_size=8,
            rotary_dim=8,
            max_position=131072,
            base=150000,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 4096,
                "truncate": False,
            },
            dtype=jnp.float32,
        )
        positions = jnp.array([1], dtype=jnp.int32)
        query = jnp.ones((1, 8), dtype=jnp.float32)
        key = jnp.ones((1, 8), dtype=jnp.float32)
        query_yarn, _ = rope(positions, query, key)
        self.assertGreater(float(jnp.linalg.norm(query_yarn)), float(jnp.linalg.norm(query)))


if __name__ == "__main__":
    unittest.main()
