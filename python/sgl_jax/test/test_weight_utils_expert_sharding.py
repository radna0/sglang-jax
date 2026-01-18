import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.utils.weight_utils import WeightLoader


class _DummyModelConfig:
    ep_size = 1


class WeightUtilsExpertShardingTest(unittest.TestCase):
    def test_shard_weight_accepts_expert_axis_on_non_expert_mesh(self):
        devices = jax.devices()
        # Keep the test lightweight; it only checks that we can construct and
        # apply an (expert, tensor) sharding spec even when the "main" mesh is
        # ("data", "tensor").
        mesh = Mesh(np.array(devices[:1]), axis_names=("data",))

        loader = WeightLoader.__new__(WeightLoader)
        loader.mesh = mesh
        loader.model_config = _DummyModelConfig()

        w3 = jnp.zeros((2, 3, 8), dtype=jnp.float32)
        w3_sharded = loader._shard_weight(w3, ("expert", None, "tensor"))
        self.assertEqual(w3_sharded.shape, w3.shape)

        w2 = jnp.zeros((2, 8), dtype=jnp.float32)
        w2_sharded = loader._shard_weight(w2, ("expert", "tensor"))
        self.assertEqual(w2_sharded.shape, w2.shape)
