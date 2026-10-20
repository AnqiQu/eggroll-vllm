#!/usr/bin/env python
"""
parse_grpo_trainer_state.py -- per-step training curves of h1 GRPO runs (submit_grpo_h1_stage.sh).

TRL writes the whole per-step log_history into every checkpoint's trainer_state.json, so
the newest checkpoint of a stage holds the full curve. This writes one compact JSON in the
results/train_curves_*.json format (see results/README.md), so the curves survive pruning
the checkpoints:

  {"_readme": ..., "runs": {run_key: {"meta": {...}, "steps": [{"step": N, ...}, ...]}}}

  python parse_grpo_trainer_state.py --out results/train_curves_grpo.json \\
      runs/grpo_h1_qwen25_3b_600steps/stage1_len1 runs/grpo_h1_readme_seedtest_s5/stage1_len1 ...

A run key is <output root>/<stage dir>; meta records the checkpoint read and the BEST step.
"""
import argparse
import glob
import json
import os
import sys


def compact(v):
    return float(f"{v:.5g}") if isinstance(v, float) else v


def parse_stage(stage_dir):
    ckpts = glob.glob(os.path.join(stage_dir, "checkpoints", "checkpoint-*"))
    if not ckpts:
        raise SystemExit(f"{stage_dir}: no checkpoints/checkpoint-*")
    last = max(ckpts, key=lambda d: int(d.rsplit("-", 1)[1]))
    with open(os.path.join(last, "trainer_state.json")) as f:
        state = json.load(f)
    # train rows carry "loss"; the final summary row (train_runtime, ...) does not
    rows = [{k: compact(v) for k, v in x.items()} for x in state["log_history"] if "loss" in x]
    meta = {"checkpoint": last, "global_step": state["global_step"], "max_steps": state["max_steps"]}
    best = os.path.join(stage_dir, "BEST")
    if os.path.exists(best):
        meta["best"] = open(best).read().strip()
    return meta, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("stage_dirs", nargs="+", help="runs/<output root>/stage<S>_len<S>")
    args = ap.parse_args()
    out = {"_readme": "Per-step GRPO training metrics (TRL log_history) from the newest checkpoint's "
           "trainer_state.json, by parse_grpo_trainer_state.py. One row per optimizer step "
           "(1 question x 16 generations); rewards/<func>/mean are h1's reward functions.", "runs": {}}
    for d in args.stage_dirs:
        d = d.rstrip("/")
        key = os.path.join(os.path.basename(os.path.dirname(d)), os.path.basename(d))
        meta, rows = parse_stage(d)
        out["runs"][key] = {"meta": meta, "steps": rows}
        print(f"{key}: {len(rows)} steps (to {meta['global_step']} of {meta['max_steps']})", file=sys.stderr)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
