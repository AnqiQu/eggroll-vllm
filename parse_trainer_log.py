#!/usr/bin/env python
"""
parse_trainer_log.py -- per-step training curves from es_lora_multinode.py stdout logs.

Reads the trainer's status lines (reward/*, DIAG:, PAIR DIAGNOSTIC:, ES UPDATE DIAG:,
TIMES:) for every "======= ES Step N / M =======" block and writes one compact JSON
in the results/train_curves_*.json format (see results/README.md):

  {"_readme": ..., "runs": {run_key: {"meta": {...}, "steps": [{"step": N, ...}, ...]}}}

Several logs can feed one run (a crash + resume, or a job that hit its walltime):
later logs win for steps they share, and within a log the LAST occurrence of a step
wins (a retry inside the same job re-runs steps after the resume checkpoint).

  python parse_trainer_log.py --out results/train_curves_q25_3b.json \\
      --run stage1_len1 logs/a.out logs/b.out  --run stage2_len2 logs/c.out
"""
import argparse
import json
import re
import sys

STEP_RE = re.compile(r"^======= ES Step (\d+) / (\d+) =======")
DONE_RE = re.compile(r"^======= ES Step (\d+) finished =======")
KV_RE = re.compile(r"([A-Za-z_][\w/|+\- ]*?)[=:]\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)(?=[,\s]|$)")

# status line prefix -> metric namespace (names are normalised to the W&B keys used elsewhere)
PREFIXES = {
    "reward/": "",
    "DIAG: ": "diag/",
    "PAIR DIAGNOSTIC: ": "pair/",
    "ES UPDATE DIAG: ": "upd/",
    "TIMES: ": "time/",
}
RENAME = {"pair/disagree_rate": "pair/disagree", "pair/mean|F+ - F-|": "pair/absdiff_mean",
          "upd/realised/intended": "upd/realised_ratio", "time/total": "time/total_s",
          "time/vLLM+Score": "time/generate_s", "time/ES update": "time/update_s"}


def parse_log(path):
    steps, cur, meta = {}, None, {}
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = STEP_RE.match(line)
            if m:
                cur = {"step": int(m.group(1))}
                meta.setdefault("num_iterations", int(m.group(2)))
                continue
            if cur is None:
                if "base model" in line and ":" in line and "base_model" not in meta:
                    meta["base_model"] = line.split(":", 1)[1].strip()
                continue
            if DONE_RE.match(line):
                steps[cur["step"]] = cur   # last occurrence wins
                cur = None
                continue
            for prefix, ns in PREFIXES.items():
                if line.startswith(prefix):
                    body = line[len(prefix):]
                    for k, v in KV_RE.findall(body):
                        key = ns + k.strip().replace(" ", "_") if prefix != "reward/" else "reward/" + k.strip()
                        key = RENAME.get(ns + k.strip(), key)
                        cur[key] = float(v)
                    break
    return steps, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", nargs="+", action="append", metavar=("KEY", "LOG"), required=True,
                    help="run key followed by one or more logs (chronological)")
    ap.add_argument("--readme", default="")
    args = ap.parse_args()
    out = {"_readme": args.readme or "Per-step training metrics parsed from trainer stdout by parse_trainer_log.py. "
           "reward/frac_correct = population-mean correctness on the step's prompts; diag/* as in es_diagnostics.py; "
           "time/* seconds per step.", "runs": {}}
    for spec in args.run:
        key, logs = spec[0], spec[1:]
        merged, meta = {}, {"logs": logs}
        for lg in logs:
            s, m = parse_log(lg)
            merged.update(s)
            for k, v in m.items():
                meta.setdefault(k, v)
        rows = [merged[k] for k in sorted(merged)]
        out["runs"][key] = {"meta": meta, "steps": rows}
        keys = sorted({k for r in rows for k in r} - {"step"})
        print(f"{key}: {len(rows)} steps from {len(logs)} log(s); metrics: {len(keys)}", file=sys.stderr)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
