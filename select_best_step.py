#!/usr/bin/env python
"""Pick a stage's best checkpoint by h1's rule (highest summed accuracy on
len_1..len_3) from results/<prefix>_<arm>_step_*.json and write its merged
path to a BEST file, which the next curriculum stage reads as its base model.
Same rule as the inline block in submit_h1_qwen25_3b_stage_correctonly.sh.

  python select_best_step.py --prefix grpo_q25_3b_len1 --arm plain \
      --merged runs/grpo_h1_qwen25_3b_plain/stage1_len1/merged --out runs/.../stage1_len1/BEST
"""
import argparse
import glob
import json
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--merged", required=True, help="dir holding step_N model dirs")
    ap.add_argument("--out", required=True, help="BEST file to write")
    ap.add_argument("--splits", nargs="+", default=["len_1", "len_2", "len_3"])
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    scores = {}
    for path in glob.glob(os.path.join(args.results_dir, f"{args.prefix}_{args.arm}_step_*.json")):
        step = int(os.path.basename(path).rsplit("_", 1)[1][:-5])
        acc = {}
        for _model, dsets in json.load(open(path)).items():
            for d, v in dsets.items():
                acc[os.path.basename(d).replace("test_", "").replace(".jsonl", "")] = v["accuracy"] * 100
        if all(k in acc for k in args.splits):
            scores[step] = (sum(acc[k] for k in args.splits), acc)
    if not scores:
        sys.exit("no evaluated steps found")
    best = max(scores, key=lambda s: (scores[s][0], -s))
    path = os.path.join(args.merged, f"step_{best}")
    assert os.path.isfile(os.path.join(path, "config.json")), path
    with open(args.out, "w") as f:
        f.write(path + "\n")
    print(f"BEST step_{best}: combined {'+'.join(args.splits)} = {scores[best][0]:.1f}  {scores[best][1]}")
    print(f"wrote {args.out} -> {path}")


if __name__ == "__main__":
    main()
