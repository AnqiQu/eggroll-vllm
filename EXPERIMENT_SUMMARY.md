# EGGROLL × h1 GSM-LongHorizon — Experiment Summary & Open Problem

_Last updated: 2026-09-16_

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
| 6 | Held-out eval | base + 6 checkpoints × test_len_1/2/3 | **Done. No checkpoint beats base on correctness** (test_len_2: base 38.6%, step_299 39.2%, all within ±1 pt after the evaluator zero-fix). Format was learned on held-out too (soft-format 0.2% → 82%). See "Held-out eval" below. |
| 7 | Perturbation-sensitivity probe (`es_reference.py probe`, job 6543946) | base vs gated step_299, σ ∈ {1e-3, 3e-3, 1e-2, 3e-2} | No entropy collapse; 26% of pairs disagree on correctness at σ=1e-3; σ=1e-3 already lowers accuracy on average. See "Perturbation-sensitivity probe". |
| 8 | **Correctness-only, horizon-2, fp32 master** (jobs 6555686/6555687, identical replicates; eval job 6600956) | pop-256, σ=1e-3, lr=2e-4, 600 steps from base, `--fp32-master` | **First run that beats base on held-out correctness: test_len_2 38.6% → 51.5% at step 599 (paired net +62/482), test_len_3 14.6% → 30.3%, test_len_1 flat.** Training `frac_correct` 0.40 → 0.51. See "Correctness-only horizon-2 with fp32 master". |
| 8b | Same, σ=3e-4 / lr=6e-5 (job 6555703) | as 8 with smaller σ | Training `frac_correct` flat (0.42 → 0.44). Checkpoints lost (all three arms wrote to one dir; stale clone, see below). |
| 9 | **bf16 control** for #8 (job 6600957, submitted 2026-09-16) | as 8 with `FP32_MASTER=0` | **Running.** Decides whether #8's gain is the fp32 master or the correctness-only objective. |
| 10 | Gated run continued 300 → 900 steps (job 6600958, submitted 2026-09-16) | resume from gated step_299, bf16 | **Running.** The literal "run longer" test. |

## The core problem
_Status 2026-09-16: this section describes the bf16 full-reward and gated runs (#1, #5). Run #8 below (correctness-only objective + fp32 master) breaks the pattern; whether the objective or the precision fix is responsible is what run #9 decides._

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

## Held-out eval of the horizon-2 gated run (2026-09-15)
`python analyze_heldout.py --splits len_2 results/len2_gated_base.json results/len2_gated_step_*.json`
(paired on the same 482 test_len_2 questions; numbers after `rescore_heldout.py`):

| checkpoint | acc | fixed | broke | net | soft-format | chars |
|---|---|---|---|---|---|---|
| base | 38.6% | – | – | – | 0.2% | 1088 |
| step_100 | 34.9% | 49 | 69 | −20 | 39% | 953 |
| step_200 | 39.2% | 66 | 65 | +1 | 63% | 989 |
| step_299 | 39.2% | 63 | 63 | 0 | 82% | 972 |

Three things the accuracy column alone hides:
- **Correctness is not frozen — it churns.** By step_299 the model flips the
  outcome on 126 of 482 questions (63 wrong→right, 63 right→wrong) with net
  zero. Training *does* move correctness a lot; it has no consistent direction
  along that axis. So "ES cannot reach correctness directions" is the wrong
  picture; "the correctness component of the update has no signal" is closer.
- **Format transferred to held-out** (0.2% → 82% soft-format) and outputs got
  ~10% shorter. In the sampled right→wrong cases the step_299 model writes a
  shorter chain and drops a hop of the chained problem.
- **Evaluator fix:** the vendored `h1_gsm_eval.py` discarded an extracted answer
  of `0` (Python falsy) and scored it as "no answer" (upstream h1 has the same
  bug). Fixed; effect ≤ 1 pt everywhere. `rescore_heldout.py` re-scores old
  files offline.

The stage-1 pop-1024 checkpoints (full h1 reward, trained on horizon-1) show a
small positive net on test_len_2 at some steps (+13…+16 at steps 50/200/250,
i.e. ~41–42% vs 38.6%) but not monotonically; too small to build on.

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
_Superseded on 2026-09-16 by "Correctness-only horizon-2 with fp32 master" below; kept for the record._

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

## Update 2026-09-14 — professor feedback (see `PROFESSOR_FEEDBACK_NOTES.md`)
Two comments from the supervisor meeting, worked through in detail in the
notes file: (1) *not enough iterations — format is optimised first, correctness
may follow once format plateaus*; (2) *re-implement the algorithm in your own
code to separate algorithm / data / code issues; be mindful of entropy
collapse*. Consequences for the table above:

- **The "objective formulation refuted" conclusion is weaker than stated.** The
  correctness-only run (#4) was on near-ceiling horizon-1, and the gated run
  (#5) stopped while `frac_format_ok` was still climbing (77%). Neither tests
  "correctness after format saturates". Two runs added to test it directly:
  `submit_h1_stage2_len2_correctonly.sh` (the missing matrix cell) and
  `submit_h1_stage2_len2_gated_resume.sh` (steps 300-899 of run #5).
- **Candidate code issue found: bf16 rounding of the ES step.** The fp32 update
  is added in place to bf16 engine weights; with this config the typical
  per-entry step (~1e-5) is below half a bf16 ulp for most weights, so ~90% of
  entries do not change and the realised step has cosine ~0.5 with the
  intended one. Strong coherent signals (format) mostly survive; weak
  consistent ones (correctness) are zeroed every step. Measured from now on as
  `diag/update/*`; removable with `FP32_MASTER=1` (fp32 master weights).
- **Entropy collapse is now monitored** (`diag/pair_identical_rate`,
  `diag/entropy/margin` with `ENTROPY_TOPK=20`). The old `pair_disagree_rate`
  rose partly mechanically under the binary reward and does not measure output
  diversity.
- **Independent re-implementation** `es_reference.py` (plain PyTorch, fp32
  master, no vLLM/Ray/PEFT) with a `probe` mode that measures, per sigma, how
  often a perturbation changes text / format / correctness on base vs the
  gated `step_299` — `submit_probe_sensitivity.sh`, 1 GPU.

## Perturbation-sensitivity probe (2026-09-15, `results/probe_len2_*.json`)
`es_reference.py probe` on base vs the gated `step_299`, 32 train_len_2 prompts,
8 antithetic pairs per sigma (full table in `PROFESSOR_FEEDBACK_NOTES.md` §3.4):
- **No entropy collapse:** greedy margin 18.4 → 17.9 nats, entropy 0.063 →
  0.070, close-race tokens 1.5% → 1.5%, zero identical pairs.
- **Correctness is very perturbable at σ=1e-3:** 26% of antithetic pair-prompt
  comparisons disagree on correctness. Signal quantity is not the bottleneck.
- **σ=1e-3 is already destructive on average** (perturbed accuracy 40.6% →
  37.5%; σ=3e-3 → 15%; σ≥1e-2 → 0%). Larger σ is ruled out; a smaller σ arm
  (3e-4, lr scaled, fp32 master) is the one to try.
- **Format is partly learnable by damage:** at σ=3e-3, 39% of broken base
  outputs match the soft format (base 0%), consistent with echoing the prompt's
  template. Format at step_299 is fragile (21% of perturbations break it), so
  the gated objective never stops spending the update on format.
- Priority is now the correctness-only horizon-2 run (three arms: fp32 master,
  bf16, σ=3e-4), then the gated resume.

## Correctness-only horizon-2 with fp32 master (2026-09-16): first held-out gain
Run #8: `submit_h1_stage2_len2_correctonly.sh`, fitness = correctness only (no
format term anywhere in the objective), horizon-2 from base, pop 256, σ=1e-3,
lr=2e-4, r=1, 600 steps (= two passes over the 2395 training prompts),
`--fp32-master` on. Per-step curves: `results/train_curves_len2.json`; held-out
files: `results/len2_conly_fp32master_sigma0.001_*.json` (evaluator with the
zero-fix, base re-evaluated in the same job).

**What went wrong with the batch.** The three arms were submitted from a cluster
clone that predated commit 5c74e99, so (a) the "bf16 control" (job 6555687) ran
with the fp32 master ON — its per-step `frac_correct` correlates 0.998 with job
6555686 (same seed), i.e. the two are one run — and (b) all three arms wrote to
one checkpoint dir. Job 6555687 wrote last everywhere (checked via
`training_state.json`), so the surviving checkpoints are the fp32 σ=1e-3 arm;
the σ=3e-4 checkpoints are gone. The dir was renamed to
`runs/h1_curriculum_len2_correctonly_fp32master_sigma0.001`; the bf16 control
was resubmitted with the fixed script (job 6600957, own dir), gated resume as
job 6600958.

**Training (population mean over 256 perturbed members × 8 prompts):**

| steps | 0–99 | 100–199 | 200–299 | 300–399 | 400–499 | 500–599 |
|---|---|---|---|---|---|---|
| fp32 master, σ=1e-3 (#8) | 0.399 | 0.427 | 0.460 | 0.481 | 0.491 | 0.512 |
| fp32 master, σ=3e-4 (#8b) | 0.422 | 0.426 | 0.433 | 0.439 | 0.443 | 0.444 |
| gated, bf16 (#5, same data order) | 0.380 | 0.373 | 0.375 | – | – | – |

Per-step values are dominated by which 8 prompts were drawn (all runs share the
seed and data order; the σ=1e-3 and gated curves correlate 0.93 step by step),
so the meaningful comparison is on the same prompts: #8 minus #5 is +0.02 over
steps 0–99 and +0.08 over steps 200–299, before any prompt has been seen twice.
Second pass minus first pass on the same prompts: +0.066 (#8), +0.015 (#8b).

**Diagnostics (#8, first 100 vs last 100 steps):** correctness pairs-with-signal
0.92 → 0.88 (no shortage of signal); identical antithetic pairs 0.01% → 0.05%;
top-k entropy 0.064 → 0.053 nats, greedy margin 18.2 → 19.0 (mild sharpening,
not collapse); truncation at 1024 tokens 0.5% → 1.3%. With the fp32 master,
`diag/update/applied_frac` sits at ~0.17 and `realised/intended` at ~3: each
step only the entries whose fp32 master crossed a bf16 rounding boundary flip,
and they flip by a whole bf16 spacing, which is a few times the intended step.
That is the expected signature of the master working (nothing is lost across
steps); it is not comparable with the bf16 arm's ~0.10 / 0.58. The σ=3e-4 arm
still had pair disagreement (pairs-with-signal 0.80, 19% pair disagreement,
2% identical pairs) but half its population produced non-distinct outputs and
it barely moved.

**Held-out (`analyze_heldout.py`, paired with base on the same questions):**

| checkpoint | len_1 acc | len_2 acc | len_2 fixed / broke / net | len_3 acc | len_3 net | len_2 chars |
|---|---|---|---|---|---|---|
| base | 76.8 | 38.6 | – | 14.6 | – | 1088 |
| step_100 | 76.9 | 41.5 | 45 / 31 / +14 | 19.4 | +14 | 1160 |
| step_200 | 78.0 | 45.6 | 62 / 28 / +34 | 24.1 | +28 | 1255 |
| step_300 | 77.6 | 46.7 | 76 / 37 / +39 | 24.8 | +30 | 1343 |
| step_400 | 78.3 | 49.0 | 88 / 38 / +50 | 28.2 | +40 | 1373 |
| step_450 | 76.2 | 52.9 | 98 / 29 / +69 | 25.9 | +33 | 1472 |
| step_500 | 76.8 | 50.8 | 91 / 32 / +59 | 31.6 | +50 | 1491 |
| step_599 | 76.3 | 51.5 | 95 / 33 / +62 | 30.3 | +46 | 1581 |

- Net gain on the trained horizon grows almost monotonically and "broke" stays
  near 30 while "fixed" triples: this is directed improvement, not churn (the
  gated run had 63 / 63).
- It transfers to the harder horizon-3 (never trained on) and does not hurt
  horizon-1.
- No format was learned (soft-format 0%; nothing in the objective asked for
  it) and outputs got ~45% longer, the opposite of the gated run's shorter,
  hop-dropping outputs. Watch the training truncation rate if this continues
  (1024-token budget in training, 2048 at eval).
- best len_2 checkpoint by accuracy: step_450 (52.9%); by len_2+len_3: step_500.

**Open confound.** Relative to the gated run, #8 changes two things at once:
the objective (correctness only) and the update precision (fp32 master). The
earlier correctness-only run (#4) was bf16 but on near-ceiling horizon-1, so
it does not settle it. Run #9 (bf16, otherwise identical, same seed — its
first steps reproduce #8's population fitness to ±0.01) is the A/B; its
training `frac_correct` at matching steps against #8 (and then the same
held-out eval via `ARM=bf16_sigma0.001 sbatch submit_eval_len2_correctonly.sh`)
answers it directly.

## Config knobs added this project
- `EGGROLL_FITNESS_MODE` = `correctness` | `total` | `gated`
- `EGGROLL_FORMAT_CHECK` = `strict` | `soft` (the format gate for gated mode)
- W&B metrics: `reward/frac_correct`, `reward/frac_format_ok`, `reward/frac_gated`,
  `reward/total_if_summed`, `pair_disagree_rate`, `pair_absdiff_mean`
- `ENTROPY_TOPK` (= `--entropy-topk`), `FP32_MASTER` (= `--fp32-master`),
  `RESUME_FROM` (= `--resume-from`) in `run_h1_curriculum.sh`
- W&B `diag/*` metrics: per-axis `*/cos_with_fitness`, `*/pairs_with_signal`,
  `pair_identical_rate`, `pair_first_div_frac`, `distinct_outputs_frac`,
  `entropy/*`, `update/*` (see `PROFESSOR_FEEDBACK_NOTES.md` §2.2, §3.3)
