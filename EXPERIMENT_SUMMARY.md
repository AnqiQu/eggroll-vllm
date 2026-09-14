# EGGROLL × h1 GSM-LongHorizon — Experiment Summary & Open Problem

_Last updated: 2026-09-14_

## Goal
Replicate h1's **GSM-LongHorizon** long-horizon math-reasoning experiment, but
**swap the optimizer from GRPO → EGGROLL** (low-rank Evolution Strategies), and
see whether ES can improve multi-hop reasoning accuracy.

## Setup
- **Model:** `Qwen/Qwen3-1.7B` — this is already the post-trained/instruct model
  (not `-Base`), run in **non-thinking mode** (`enable_thinking=False`).
- **Optimizer — EGGROLL:** low-rank (LoRA) **antithetic** Evolution Strategies.
  Explores by applying rank-r weight perturbations to a population of members
  (mirror pairs ±Δ); each rollout gets a **scalar fitness**; the update is a
  fitness-weighted combination of the perturbations. No backprop. σ = exploration
  radius, lr = step size.
- **Reward (from h1, DrGRPO):** `correctness` (2.0 if answer matches via
  `grade_answer`, else 0) **+** `format` (XML-tag rewards: xmlcount + soft_format
  + strict_format + int/float ≈ up to ~2.0), summed with equal weight. Required
  output format is `<reasoning>…</reasoning><answer>…</answer>`.
- **Curriculum:** 5 stages by horizon (number of chained reasoning hops). Stage N
  warm-starts from stage N-1's best checkpoint. Stage 1 = `train_len_1` (integer
  answers, 768-tok budget); Stage 2 = `train_len_2` (float answers, 1024-tok
  default); etc.
- **Hardware:** Isambard-AI Phase 2 — Slurm, 4×GH200 per node, Ray-based
  multi-GPU (one vLLM engine per GPU, TP=1 for a 1.7B model).

## Experiments run (chronological)

| # | Run | Config | Result |
|---|-----|--------|--------|
| 1 | Stage-1 (horizon-1), **full h1 reward** | pop-128 and pop-1024 | **Reward went up, accuracy flat.** First sign of trouble. |
| 2 | Implementation code review | — | **No bugs found.** EGGROLL + h1 reward port is faithful. |
| 3 | Diagnostics added | pair-disagreement metric + per-component reward logging | Confirmed gradient signal is healthy (see below). |
| 4 | **Correctness-only** fitness, stage-1 (job 6405644) | pop-128, format dropped from objective | **Correctness flat ~0.81** — but horizon-1 is near-ceiling, so this was the wrong testbed. |
| 5 | **Gated** reward, **horizon-2** (job 6455002) | pop-256, 2048-tok gen, from base, soft format gate | **Objective rose 3%→29%, but 100% from format; correctness flat ~37% (= base).** See below. |
| 6 | Held-out eval | base + 6 checkpoints × test_len_1/2/3 | **Set up, pending** (last confirmed state). |

## The core problem
**Under EGGROLL, the reward goes up but correctness does not.** Every gain comes
from the model learning to *format* its output, not to *reason* more correctly.

The horizon-2 gated run is the clearest evidence. The "gated" reward is binary:
**1 iff (answer correct AND well-formatted), else 0** — specifically designed to
remove partial-credit attractors and force ES to produce correct+formatted
answers. Over 300 steps:
- `frac_gated` (objective): **3.2% → 29%**
- `frac_format_ok`: **9.6% → 77%** ← the entire climb
- `frac_correct`: **~37% → ~37%, dead flat** (= base Qwen3-1.7B's ~38% on test_len_2)

**Smoking gun:** `frac_gated ≈ frac_correct × frac_format_ok`, with the ratio →
**1.00** by mid-run. Correctness and format are statistically **independent** —
the gate created *zero* synergy. ES just fixed the formatting on the ~37% of
answers it was already getting right; it never made more answers correct.

This is the same failure mode as the horizon-1 full-reward run (format-hacking),
now confirmed on a harder horizon where there was genuine correctness headroom
(37% → 60-75% was available and untouched).

## Hypotheses tested and ruled out
- **Implementation bug** → ruled out by full code review (exp #2).
- **Dead gradient / σ too small** → **refuted.** `pair_disagree_rate` (fraction
  of antithetic ± pairs with different fitness = the ES signal) is healthy and
  *rising* (6% → 29% on horizon-2; stable ~0.17 on horizon-1). ES has plenty of
  signal — it just spends it all on the format axis.
- **Objective formulation is the bottleneck** → **refuted.** Two different
  objectives (correctness-only, gated) both leave correctness flat. Reshaping the
  reward is not the fix.

## Fixes attempted → outcomes
1. **Correctness-only fitness** (drop format from objective) → correctness flat
   (tested only on near-ceiling horizon-1).
2. **Binary gated reward** (correct AND formatted) — a collaborator's suggestion
   → correctness still flat on horizon-2.
3. **Longer CoT budget** (1024 → 2048 tokens) — also collaborator's suggestion →
   **inert.** Mean output stayed ~320 tokens, never approaching even 1024. The
   binding constraint is the model's *propensity* to reason at length, not the
   token budget.
4. **Diagnostics** (pair disagreement + component logging) → didn't fix anything
   but isolated the problem to the reasoning axis.

## Current interpretation & open fork
Low-rank antithetic ES readily finds **format** directions (shallow
output-distribution changes) but not **reasoning/correctness** directions (deeper
capability changes) on multi-hop math. Format is trivially learnable by ES;
correctness is not.

The real fork (not more reward reshaping):
- **Capability-side ES levers:** higher LoRA rank `r`, larger σ, larger
  population, more steps — give correctness-improving directions a chance to be
  found.
- **Actual longer reasoning:** enable thinking mode, or a reward that shapes for
  genuine multi-step CoT — the model needs to reason more, not just have budget to.

**Immediate next step:** the held-out eval (base vs. all 6 checkpoints on
test_len_1/2/3) to definitively confirm no checkpoint beats base on correctness.
Training curves predict "no," but this is the publishable confirmation.

## Config knobs added this project
- `EGGROLL_FITNESS_MODE` = `correctness` | `total` | `gated`
- `EGGROLL_FORMAT_CHECK` = `strict` | `soft` (the format gate for gated mode)
- W&B metrics: `reward/frac_correct`, `reward/frac_format_ok`, `reward/frac_gated`,
  `reward/total_if_summed`, `pair_disagree_rate`, `pair_absdiff_mean`
