#!/usr/bin/env python
"""
analyze_heldout.py -- paired comparison of h1_gsm_eval.py result files.

Accuracy alone hides what training did: two checkpoints can have the same
accuracy while one of them changed the outcome on a quarter of the questions.
This script pairs every checkpoint with a baseline file on the same questions
and reports, per test split:

  acc%     accuracy
  fixed    questions the baseline got wrong and this checkpoint got right
  broke    questions the baseline got right and this checkpoint got wrong
  net      fixed - broke (the only part of "fixed" that is real progress)
  soft%    outputs matching h1's soft <reasoning>..</reasoning><answer>..</answer>
  strict%  outputs matching h1's strict format (exact newlines)
  <ans>%   outputs containing an <answer> tag at all
  noans    outputs the evaluator could not extract an answer from
  chars    mean output length in characters
  same%    outputs byte-identical to the baseline's

Usage:
  python analyze_heldout.py results/len2_gated_base.json results/len2_gated_step_*.json
  python analyze_heldout.py --splits len_2 results/len2_gated_base.json results/len2_gated_step_299.json
  python analyze_heldout.py --splits len_1,len_2 results/stage1_len1_baseline_qwen3-1.7b.json results/stage1_pop1024_step_*.json
"""
import argparse
import json
import os
import re

import numpy as np

SOFT = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", re.DOTALL)
STRICT = re.compile(r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$", re.DOTALL)


def load(path):
    """-> {split_name: {"samples": [...], "no_answer": int}}"""
    out = {}
    with open(path) as f:
        data = json.load(f)
    for _model, dsets in data.items():
        for ds, v in dsets.items():
            split = os.path.basename(ds).replace("test_", "").replace("train_", "").replace(".jsonl", "")
            out[split] = v
    return out


def step_key(path):
    m = re.search(r"step_(\d+)", path)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", help="eval json of the reference model (e.g. base)")
    ap.add_argument("checkpoints", nargs="+", help="eval jsons to compare against it")
    ap.add_argument("--splits", default=None, help="comma-separated splits to show (e.g. len_2 or len_1,len_2)")
    args = ap.parse_args()
    if args.splits:
        args.splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    base = load(args.baseline)
    ckpts = sorted(args.checkpoints, key=step_key)
    for split, bv in base.items():
        if args.splits and split not in args.splits:
            continue
        bs = bv["samples"]
        bi = {s["index"]: s for s in bs}
        btxt = [s["generated_text"] for s in bs]
        print(f"\n[{split}] baseline {os.path.basename(args.baseline)}: acc={np.mean([s['is_correct'] for s in bs])*100:.1f}%  "
              f"soft={np.mean([bool(SOFT.match(t)) for t in btxt])*100:.1f}%  strict={np.mean([bool(STRICT.match(t)) for t in btxt])*100:.1f}%  "
              f"noans={bv.get('no_answer', '?')}  chars={np.mean([len(t) for t in btxt]):.0f}  n={len(bs)}")
        print(f"  {'checkpoint':34s} {'acc%':>6s} {'fixed':>6s} {'broke':>6s} {'net':>5s} {'soft%':>6s} {'strict%':>8s} {'<ans>%':>7s} {'noans':>6s} {'chars':>6s} {'same%':>6s}")
        for path in ckpts:
            cv = load(path).get(split)
            if cv is None:
                continue
            cs = cv["samples"]
            fixed = sum(1 for s in cs if s["index"] in bi and not bi[s["index"]]["is_correct"] and s["is_correct"])
            broke = sum(1 for s in cs if s["index"] in bi and bi[s["index"]]["is_correct"] and not s["is_correct"])
            txt = [s["generated_text"] for s in cs]
            same = np.mean([s["index"] in bi and s["generated_text"] == bi[s["index"]]["generated_text"] for s in cs])
            print(f"  {os.path.basename(path)[:34]:34s} {np.mean([s['is_correct'] for s in cs])*100:6.1f} {fixed:6d} {broke:6d} {fixed-broke:+5d} "
                  f"{np.mean([bool(SOFT.match(t)) for t in txt])*100:6.1f} {np.mean([bool(STRICT.match(t)) for t in txt])*100:8.1f} "
                  f"{np.mean(['<answer>' in t for t in txt])*100:7.1f} {cv.get('no_answer', 0):6d} {np.mean([len(t) for t in txt]):6.0f} {same*100:6.1f}")
    print("\nnet = fixed - broke. Large fixed AND broke with net ~0 means training moved correctness on many\n"
          "questions with no consistent direction (churn), not that correctness is untouchable.")


if __name__ == "__main__":
    main()
