#!/usr/bin/env python
"""
summarize_results.py -- compact, committable summaries of h1_gsm_eval.py dumps.

h1_gsm_eval.py --out_file writes every question and every generated answer
(4-7 MB per checkpoint). Those full dumps are not committed (results/.gitignore):
they stay on Isambard ($SCRATCH/eggroll-vllm/results) and in local copies, and
the ones from before 2026-09-29 are also in git history. For each dump
results/<name>.json this writes results/summary/<name>.json with the SAME nesting
({model: {dataset: {...}}}), so code that only reads "accuracy" (the eval
scripts' comparison tables, select_best_step.py) works on either file:

  accuracy, correct, total, no_answer, answer_rate, avg_num_tokens_in_responses
  num_problems, num_generations
  is_correct   one '0'/'1' string per generation; character i = the i-th problem
               in dataset-index order
  indices      only when the dump covered a subset: comma-separated dataset indices

and rebuilds results/summary/accuracy.csv (one row per summary file x dataset)
from every summary on disk. Paired analyses that need only per-problem
correctness can run on summaries; anything that reads the generated text
(analyze_heldout.py's format/length columns, rescore_heldout.py) needs the dump.

  python summarize_results.py                  # every dump in results/
  python summarize_results.py results/foo.json # just these
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

SCALARS = ("accuracy", "correct", "total", "no_answer", "answer_rate", "avg_num_tokens_in_responses")


def is_dump(obj) -> bool:
    """h1_gsm_eval.py output: {model: {dataset: {..., "samples": [...]}}}."""
    return (isinstance(obj, dict) and bool(obj)
            and all(isinstance(ds, dict) and ds and all(isinstance(v, dict) and "samples" in v for v in ds.values())
                    for ds in obj.values()))


def summarize_split(v: dict) -> dict:
    per_index: dict[int, list[bool]] = {}
    for s in v["samples"]:  # one entry per generation, generations of a problem are consecutive
        per_index.setdefault(int(s["index"]), []).append(bool(s["is_correct"]))
    idx = sorted(per_index)
    gens = {len(g) for g in per_index.values()}
    if len(gens) > 1:
        raise ValueError(f"problems have different numbers of generations: {sorted(gens)}")
    n_gen = gens.pop() if gens else 0
    out = {k: v[k] for k in SCALARS if k in v}
    out["num_problems"] = len(idx)
    out["num_generations"] = n_gen
    out["is_correct"] = ["".join("1" if per_index[i][g] else "0" for i in idx) for g in range(n_gen)]
    if idx != list(range(len(idx))):
        out["indices"] = ",".join(map(str, idx))
    ones = sum(s.count("1") for s in out["is_correct"])
    if "correct" in v and ones != v["correct"]:
        raise ValueError(f"per-problem correctness ({ones}) disagrees with 'correct' ({v['correct']})")
    return out


def summarize(obj: dict) -> dict:
    return {model: {ds: summarize_split(v) for ds, v in dsets.items()} for model, dsets in obj.items()}


def csv_rows(summary_dir: str) -> list[list]:
    rows = []
    for path in sorted(glob.glob(os.path.join(summary_dir, "*.json"))):
        name = os.path.basename(path)[: -len(".json")]
        with open(path) as f:
            obj = json.load(f)
        for model, dsets in obj.items():
            for ds, v in sorted(dsets.items()):
                split = os.path.basename(ds).replace("test_", "").replace(".jsonl", "")
                rows.append([name, split, f"{100 * v.get('accuracy', float('nan')):.2f}", v.get("correct", ""),
                             v.get("total", ""), v.get("no_answer", ""),
                             f"{v['avg_num_tokens_in_responses']:.1f}" if "avg_num_tokens_in_responses" in v else "",
                             model])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="dump files (default: every dump in results/)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default=None, help="default: <results-dir>/summary")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "summary")
    os.makedirs(out_dir, exist_ok=True)

    paths = args.paths or sorted(glob.glob(os.path.join(args.results_dir, "*.json")))
    n = 0
    for path in paths:
        with open(path) as f:
            obj = json.load(f)
        if not is_dump(obj):
            if args.paths:
                print(f"skip {path}: not an h1_gsm_eval.py dump", file=sys.stderr)
            continue
        out = os.path.join(out_dir, os.path.basename(path))
        with open(out, "w") as f:
            json.dump(summarize(obj), f, indent=2)
            f.write("\n")
        n += 1
    rows = csv_rows(out_dir)
    with open(os.path.join(out_dir, "accuracy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "accuracy_pct", "correct", "total", "no_answer", "avg_tokens", "model"])
        w.writerows(rows)
    print(f"summarized {n} dump(s) -> {out_dir}/ ; accuracy.csv has {len(rows)} rows")


if __name__ == "__main__":
    main()
