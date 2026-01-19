import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.speculative.dflash_info import DFlashVerifyInput


class TestDFlashVerifyInput(unittest.TestCase):
    def test_verify_greedy_accept_and_bonus(self):
        # candidates[:,0] is current token, candidates[:,1:] are draft tokens.
        candidates = jnp.asarray(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            dtype=jnp.int32,
        )
        bs, block = candidates.shape
        vocab = 16

        # Construct target_predict by shaping logits so argmax hits these ids.
        # We want:
        # - row0: accept_len=2 because candidates[0,1:3] == target_predict[0,0:2]
        #         bonus = target_predict[0,2]
        # - row1: accept_len=0 because candidates[1,1] != target_predict[1,0]
        target_predict = jnp.asarray(
            [
                [2, 3, 9, 10],
                [0, 0, 0, 0],
            ],
            dtype=jnp.int32,
        )

        logits = jnp.zeros((bs * block, vocab), dtype=jnp.float32)
        flat_tp = target_predict.reshape(-1)
        logits = logits.at[jnp.arange(bs * block), flat_tp].set(1.0)

        verify_input = DFlashVerifyInput(
            draft_token=candidates.reshape(-1),
            positions=None,
            custom_mask=None,
            draft_token_num=block,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

        logits_output = SimpleNamespace(next_token_logits=logits, hidden_states=None)
        accept_len, commit_lens, bonus = verify_input.verify_greedy(
            batch_size=bs,
            logits_output=logits_output,
        )

        accept_len = jax.device_get(accept_len).tolist()
        commit_lens = jax.device_get(commit_lens).tolist()
        bonus = jax.device_get(bonus).tolist()

        self.assertEqual(accept_len, [2, 0])
        self.assertEqual(commit_lens, [3, 1])
        self.assertEqual(bonus, [9, 0])

    def test_extract_commit_target_hidden(self):
        bs = 2
        block = 4
        feat = 6

        # hidden_states are [bs*block, feat] and should be sliced per-request,
        # taking the first commit_lens[i] tokens from each block.
        hidden_states = jnp.arange(bs * block * feat, dtype=jnp.float32).reshape(bs * block, feat)

        logits_output = SimpleNamespace(next_token_logits=None, hidden_states=hidden_states)
        verify_input = DFlashVerifyInput(
            draft_token=jnp.zeros((bs * block,), dtype=jnp.int32),
            positions=None,
            custom_mask=None,
            draft_token_num=block,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

        commit_lens = jnp.asarray([1, 3], dtype=jnp.int32)
        out = verify_input.extract_commit_target_hidden(
            batch_size=bs,
            logits_output=logits_output,
            commit_lens=commit_lens,
        )

        out_np = jax.device_get(out)
        # Expected: take hidden[0,0] and hidden[1,0:3]
        expected = jnp.concatenate(
            [
                hidden_states.reshape(bs, block, feat)[0, :1, :],
                hidden_states.reshape(bs, block, feat)[1, :3, :],
            ],
            axis=0,
        )
        self.assertTrue(jnp.all(out_np == jax.device_get(expected)))


if __name__ == "__main__":
    unittest.main()

