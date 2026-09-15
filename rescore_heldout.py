#!/usr/bin/env python
"""
rescore_heldout.py -- re-score h1_gsm_eval.py result files offline.

The result JSONs store every generated text, so accuracy can be recomputed
without a GPU. This script applies the SAME extraction and comparison rules as
the (patched) vendored evaluator and reports old vs new accuracy per split.

Why it exists: upstream h1's gsm_eval.py extracts the answer with
    _extract_xml_answer(text) or _legacy_extract_answer(text)
so an extracted 0 (falsy) is discarded and the sample is usually scored as
"no answer". h1_gsm_eval.py in this repo now checks for None instead. Run this
on files produced before that fix to see what changed (typically a handful of
questions whose ground truth is 0).

Usage:
  python rescore_heldout.py results/len2_gated_*.json
  python rescore_heldout.py --write results/len2_gated_step_299.json   # rewrite accuracy fields in place
"""
import argparse
import json
import os
import re
from typing import Optional

TOL = 1e-2  # h1_gsm_eval.py: abs(pred - gt) < 1e-2


def extract_xml_answer(text: str) -> Optional[float]:
    m = re.search(r"<answer>\s*([+-]?\d+(?:\.\d+)?)\s*</answer>", text, flags=re.DOTALL)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def legacy_extract_answer(text: str, is_instruct: bool = True) -> Optional[float]:
    text = text.replace(",", "")
    if m := re.search(r"[Tt]he answer is:?\s*([+-]?\d+(?:\.\d+)?)", text):
        try:
            return float(m.group(1))
        except ValueError:
            pass
    if is_instruct and "####" in text:
        tail = text.split("####")[-1].strip()
        if m := re.search(r"([+-]?\d+(?:\.\d+)?)", tail):
            try:
                return float(m.group(1))
            except ValueError:
                pass
    parts = re.split(r"answer", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", parts[-1])
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
    lines = text.strip().splitlines()
    if lines:
        numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", lines[-1])
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
    return None


def extract_answer(text: str, fixed: bool = True) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        pass
    xml = extract_xml_answer(text)
    if fixed:
        return xml if xml is not None else legacy_extract_answer(text)
    return xml or legacy_extract_answer(text)  # upstream behaviour (drops 0)


def rescore(samples, fixed=True):
    correct = no_answer = 0
    new_samples = []
    for s in samples:
        pred = extract_answer(s["generated_text"], fixed=fixed)
        gt = float(s["ground_truth"])
        ok = pred is not None and abs(pred - gt) < TOL
        if pred is None:
            no_answer += 1
        elif ok:
            correct += 1
        new_samples.append({**s, "predicted": pred, "is_correct": ok})
    return correct, no_answer, new_samples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--write", action="store_true", help="rewrite the files with re-scored fields")
    args = ap.parse_args()
    print(f"{'file':40s} {'split':8s} {'old acc%':>9s} {'new acc%':>9s} {'delta':>6s} {'old noans':>9s} {'new noans':>9s}")
    for path in args.files:
        with open(path) as f:
            data = json.load(f)
        changed = False
        for model, dsets in data.items():
            for ds, v in dsets.items():
                split = os.path.basename(ds).replace("test_", "").replace(".jsonl", "")
                total = v["total"]
                old_acc = v["accuracy"]
                correct, noans, new_samples = rescore(v["samples"], fixed=True)
                new_acc = correct / total if total else 0.0
                print(f"{os.path.basename(path)[:40]:40s} {split:8s} {old_acc*100:9.2f} {new_acc*100:9.2f} {(new_acc-old_acc)*100:+6.2f} {v['no_answer']:9d} {noans:9d}")
                if args.write:
                    v.update({"correct": correct, "no_answer": noans, "accuracy": new_acc,
                              "answer_rate": (total - noans) / total if total else 0.0,
                              "samples": new_samples, "rescored_with_zero_fix": True})
                    changed = True
        if args.write and changed:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  rewrote {path}")


if __name__ == "__main__":
    main()
