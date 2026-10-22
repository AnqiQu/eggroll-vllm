"""Tests for log_heldout_to_wandb.heldout_rows (pure python, no W&B).
Run:  PYTHONPATH=.:tests python -m unittest test_log_heldout_to_wandb"""
import os
import tempfile
import unittest

from log_heldout_to_wandb import heldout_rows

CSV = """file,split,accuracy_pct,correct,total,no_answer,avg_tokens,model
run_a_base,len_1,82.11,1083,1319,40,287.2,base
run_a_base,len_2,33.82,163,482,6,414.9,base
run_a_step_50,len_1,84.2,1,1,0,300.0,m
run_a_step_100,len_2,44.6,1,1,0,,m
run_a_shuffle_step_50,len_1,10.0,1,1,0,1.0,m
"""


class TestHeldoutRows(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as f:
            f.write(CSV)

    def tearDown(self):
        os.remove(self.path)

    def test_steps_base_and_longer_prefix_excluded(self):
        rows = heldout_rows(self.path, "run_a")
        self.assertEqual(sorted(rows), [0, 50, 100])   # run_a_shuffle_* is another run
        self.assertEqual(rows[0], {"heldout/len_1": 82.11, "heldout/tokens_len_1": 287.2,
                                   "heldout/len_2": 33.82, "heldout/tokens_len_2": 414.9})
        self.assertEqual(rows[100], {"heldout/len_2": 44.6})   # empty avg_tokens -> no tokens key
        self.assertEqual(sorted(heldout_rows(self.path, "run_a_shuffle")), [50])


if __name__ == "__main__":
    unittest.main()
