# results/

Version-controlled, **lightweight** run outputs — evaluation metrics JSON and
summary tables. These are committed to git so runs stay reproducible and
comparable across machines/teammates.

The eval scripts write here automatically:

| Script | Writes |
| --- | --- |
| `run_h1_smoke.sh` | `results/smoke_eval.json` |
| `run_h1_curriculum.sh` (printed eval commands) | `results/stage<N>_len<H>_step_<step>.json` |

## What does NOT go here

Large artifacts — training checkpoints and merged model weights
(`model_weights.safetensors`, ~GBs) — live under `runs/` (gitignored) and would
exceed GitHub's 100 MB per-file limit. A `.gitignore` in this folder blocks
common binary types as a safety net; keep only small JSON/text summaries here.

## Committing results

```bash
git add results/            # metrics only; runs/ stays ignored
git commit -m "results: <what this run was>"
git push origin main
```
