"""Tests for summarize_results.py (pure python).
Run:  python -m unittest tests.test_summarize_results"""
import json
import os
import tempfile
import unittest

import summarize_results as S


def sample(i, ok):
    return {"index": i, "question": "q", "ground_truth": 1.0, "predicted": 1.0 if ok else None,
            "is_correct": ok, "generated_text": "long text"}


# one model; len_1 = all 4 problems in shuffled order, 1 generation;
# len_2 = a subset {2, 5, 9} with 2 generations each
DUMP = {"runs/x/merged/step_50": {
    "GSM-LongHorizon/test_len_1.jsonl": {
        "correct": 2, "total": 4, "no_answer": 1, "accuracy": 0.5, "answer_rate": 0.75,
        "avg_num_tokens_in_responses": 321.5, "num_samples": 4,
        "samples": [sample(3, True), sample(0, False), sample(2, True), sample(1, False)],
        "ok_all": [[1], [0], [1], [0]]},
    "GSM-LongHorizon/test_len_2.jsonl": {
        "correct": 3, "total": 6, "no_answer": 0, "accuracy": 0.5, "answer_rate": 1.0,
        "samples": [sample(9, True), sample(9, False), sample(2, False), sample(2, False),
                    sample(5, True), sample(5, True)],
        "ok_all": [[1, 0], [0, 0], [1, 1]]},
}}


class TestSummarize(unittest.TestCase):
    def test_is_dump(self):
        self.assertTrue(S.is_dump(DUMP))
        self.assertFalse(S.is_dump({"steps": [1, 2]}))
        self.assertFalse(S.is_dump({}))
        self.assertFalse(S.is_dump({"m": {"d": {"accuracy": 0.5}}}))  # a summary is not a dump

    def test_full_split_in_index_order(self):
        s = S.summarize(DUMP)["runs/x/merged/step_50"]["GSM-LongHorizon/test_len_1.jsonl"]
        self.assertEqual(s["is_correct"], ["0011"])  # problems 0,1,2,3
        self.assertNotIn("indices", s)
        self.assertEqual((s["num_problems"], s["num_generations"]), (4, 1))
        self.assertEqual((s["accuracy"], s["correct"], s["total"], s["no_answer"]), (0.5, 2, 4, 1))
        self.assertEqual(s["avg_num_tokens_in_responses"], 321.5)
        self.assertNotIn("samples", s)
        self.assertNotIn("ok_all", s)

    def test_subset_with_generations(self):
        s = S.summarize(DUMP)["runs/x/merged/step_50"]["GSM-LongHorizon/test_len_2.jsonl"]
        self.assertEqual(s["indices"], "2,5,9")
        self.assertEqual(s["is_correct"], ["011", "010"])  # gen 0 / gen 1 over problems 2,5,9
        self.assertEqual(s["num_generations"], 2)
        self.assertNotIn("avg_num_tokens_in_responses", s)  # absent in the dump -> absent here

    def test_inconsistent_correct_raises(self):
        bad = json.loads(json.dumps(DUMP))
        bad["runs/x/merged/step_50"]["GSM-LongHorizon/test_len_1.jsonl"]["correct"] = 3
        with self.assertRaises(ValueError):
            S.summarize(bad)

    def test_accuracy_readers_see_same_numbers(self):
        # the nesting is unchanged, so the eval scripts' "for model, dsets: v['accuracy']" loop works
        summ = S.summarize(DUMP)
        for model, dsets in DUMP.items():
            for ds, v in dsets.items():
                self.assertEqual(summ[model][ds]["accuracy"], v["accuracy"])

    def test_csv_rows(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "len2_step_50.json"), "w") as f:
                json.dump(S.summarize(DUMP), f)
            rows = S.csv_rows(d)
        self.assertEqual([r[:3] for r in rows], [["len2_step_50", "len_1", "50.00"], ["len2_step_50", "len_2", "50.00"]])
        self.assertEqual(rows[0][6], "321.5")
        self.assertEqual(rows[1][6], "")


if __name__ == "__main__":
    unittest.main()
