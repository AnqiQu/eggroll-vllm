#!/usr/bin/env python
"""
log_heldout_to_wandb.py -- put held-out accuracy (results/summary/accuracy.csv) in W&B.

The trainers log their own per-step metrics (ES: reward/*, diag/*; GRPO: TRL's), but
the held-out evals run after training, outside those runs. This logs them as one W&B
run per result prefix, named "<prefix>-heldout" (job_type "heldout", group <prefix>),
with x-axis train_step (base = 0) and heldout/len_k = accuracy % on test_len_k
(+ heldout/tokens_len_k = mean response tokens). The run id is derived from the
prefix, so rerunning resumes the same run and logs only steps newer than the last one.

  python summarize_results.py results/<prefix>_*.json   # refresh accuracy.csv first
  python log_heldout_to_wandb.py q25_3b_onlylen5_fp32master_sigma0.001 grpo600_fix_pub_len5_plain
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re


def heldout_rows(csv_path: str, prefix: str) -> dict[int, dict[str, float]]:
    """{train_step: {"heldout/len_k": acc %, "heldout/tokens_len_k": tokens}} for one prefix (base -> 0)."""
    out: dict[int, dict[str, float]] = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if not r["file"].startswith(prefix + "_"):
                continue
            tail = r["file"][len(prefix) + 1:]
            if tail == "base":
                step = 0
            elif m := re.fullmatch(r"step_(\d+)", tail):
                step = int(m.group(1))
            else:
                continue   # a longer prefix that merely starts with this one
            d = out.setdefault(step, {})
            d[f"heldout/{r['split']}"] = float(r["accuracy_pct"])
            if r.get("avg_tokens"):
                d[f"heldout/tokens_{r['split']}"] = float(r["avg_tokens"])
    return out


def run_id(prefix: str) -> str:
    return "heldout-" + hashlib.md5(prefix.encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--csv", default="results/summary/accuracy.csv")
    ap.add_argument("--project", default="eggroll-h1")
    args = ap.parse_args()
    import wandb

    for prefix in args.prefixes:
        rows = heldout_rows(args.csv, prefix)
        if not rows:
            print(f"{prefix}: no rows in {args.csv}, skipped")
            continue
        run = wandb.init(project=args.project, id=run_id(prefix), name=f"{prefix}-heldout", resume="allow",
                         job_type="heldout", group=prefix, config={"result_prefix": prefix})
        run.define_metric("train_step")
        run.define_metric("heldout/*", step_metric="train_step")
        last = run.summary.get("train_step", -1)
        new = [s for s in sorted(rows) if s > last]
        for s in new:
            run.log({"train_step": s, **rows[s]})
        run.finish()
        print(f"{prefix}: logged {len(new)} of {len(rows)} steps -> {args.project}/{run_id(prefix)}")


if __name__ == "__main__":
    main()
