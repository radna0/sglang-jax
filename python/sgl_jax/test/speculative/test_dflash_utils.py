import unittest

import jax.numpy as jnp

from sgl_jax.srt.speculative.dflash_utils import compute_dflash_accept_len_and_bonus


class TestDFlashAcceptRule(unittest.TestCase):
    def test_accept_len_and_bonus_matches_reference_rule(self):
        # candidates[:, 0] is the current/anchor token.
        candidates = jnp.array(
            [
                [10, 11, 12, 13],
                [20, 21, 22, 23],
                [30, 31, 32, 33],
            ],
            dtype=jnp.int32,
        )

        # target_predict[t] is the argmax token predicted *after* candidates[t] at that position.
        # Accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
        target_predict = jnp.array(
            [
                # accept_len=2 (11==11, 12==12), stop at 13!=99, bonus = target_predict[2] = 99
                [11, 12, 99, 0],
                # accept_len=0 (21!=77), bonus = target_predict[0] = 77
                [77, 0, 0, 0],
                # accept_len=3 (31==31,32==32,33==33), bonus = target_predict[3] = 123
                [31, 32, 33, 123],
            ],
            dtype=jnp.int32,
        )

        accept_len, bonus = compute_dflash_accept_len_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )

        self.assertEqual(accept_len.tolist(), [2, 0, 3])
        self.assertEqual(bonus.tolist(), [99, 77, 123])


if __name__ == "__main__":
    unittest.main()

