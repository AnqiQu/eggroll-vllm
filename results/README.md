# results/

Run outputs small enough to version, so results stay comparable across machines
without the cluster.

## Committed

| Path | What | Written by |
| --- | --- | --- |
| `summary/<name>.json` | per evaluated model and test split: accuracy, counts, mean output tokens, and `is_correct` (one `0`/`1` per problem, in dataset-index order) | `summarize_results.py` |
| `summary/accuracy.csv` | one row per summary x split | `summarize_results.py` |
| `train_curves_len*.json` | per-step training metrics parsed from trainer logs | log parser |
| `probe_*.json` | perturbation-sensitivity probe | `es_reference.py probe` |
| `pop_*.json` | population evaluations (per-member correctness, no text) | `population_eval.py` |
| `es_training_charts.html` | self-contained chart page | |

A summary has the same `{model: {dataset: {...}}}` nesting as the full dump, so
anything that only reads `accuracy` works on either.

## Not committed

* **Full eval dumps** `results/<prefix>_<arm>_<label>.json` from
  `h1_gsm_eval.py --out_file` (every eval submit script writes these): every
  question and generated answer, 4-7 MB each. Gitignored here; they live on
  Isambard in `$SCRATCH/eggroll-vllm/results/` and in local copies. Dumps
  committed before 2026-09-29 are still in git history
  (`git log --all -- results/<name>.json`, then `git show <commit>:results/<name>.json`).
  `analyze_heldout.py` and `rescore_heldout.py` read the generated text, so they
  need the dumps, not the summaries.
* Checkpoints and merged weights: `runs/` (gitignored at the top level).

## After an eval on the cluster

```bash
scp '<isambard>:<$SCRATCH>/eggroll-vllm/results/<prefix>_*.json' results/
python summarize_results.py          # summary/<name>.json + summary/accuracy.csv
git add results/summary && git commit -m "results: <what this run was>"
```
