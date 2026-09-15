# Professor feedback: what it means, and how to test it

_Companion to `EXPERIMENT_SUMMARY.md`. Written 2026-09-14._

Two comments came out of the meeting:

1. *"Maybe you are not running it for enough iterations. If format is getting
   optimised and the format plateaus, then maybe correctness will be optimised
   next."*
2. *"Try it on your own code. There is the algorithm, the data and the code —
   use your own code to see if there are issues."* — and *"be mindful of
   entropy collapse."*

This note explains both in the language of the actual ES loop, says what the
existing runs do and do not tell us about them, and lays out the experiments
(and code) added to settle them. A concrete candidate code issue turned up
while writing it (Section 3.3), which is exactly the kind of thing comment 2 is
about.

---

## 0. The ES update in one line (needed for everything below)

Each step, for antithetic pair `k`, member `2k` gets weights `W + eps_k` and
member `2k+1` gets `W - eps_k` (`eps_k = B_k A_k`, rank `r`, per-entry std
`sigma`). After per-prompt mean-centring and (with `--normalize-with-std`)
dividing by the population std, the update is

```
W  <-  W + lr / (P * sigma) * sum_k  d_k * eps_k,      d_k = F(+eps_k) - F(-eps_k)
```

So the update **direction** is a weighted sum of the random perturbations, and
the weights `d_k` are the *fitness differences within a pair*. The update
literally contains no information about anything that does not change `d_k`.
With `--normalize-with-std` the update **magnitude** is fixed per step
(`||d|| ~ const`), regardless of how much or little the fitness varied.

Decoding is greedy (`temperature=0.0`). The weight perturbation is therefore
the **only** source of diversity between population members.

---

## 1. "Not enough iterations": the sequential-optimisation hypothesis

### 1.1 What the professor is saying

The objective has (at least) two axes, format and correctness, and they are
not equally *easy* for ES to move:

* **Format** is a shallow, low-dimensional change to the output distribution
  (emit `<reasoning>\n` first, `</answer>` last). Many random rank-1
  perturbations nudge it, so many pairs have `d_k != 0` because of format.
* **Correctness** on multi-hop GSM is a deep change. Few random perturbations
  flip an answer from wrong to right, so few pairs have a correctness-driven
  `d_k`.

While both axes are live, the vector `d` is dominated by format differences,
and the fixed-size step is spent almost entirely on format. The correctness
information is still *in* `d` (as a small component), but each step realises
only a small fraction of it, and an ES estimate of a weak direction in a
~10^9-dimensional space needs many steps to accumulate. Once format saturates
— `frac_format_ok -> 1` under the gated reward, so pairs can no longer differ
on format — **all** remaining fitness variance is correctness, and the same
fixed-size step goes entirely to correctness. That is the "correctness will be
optimised next" regime.

Under the gated reward this is even sharper: `gated = correct AND format_ok`.
A pair only carries a correctness difference if *both* members are formatted.
At `frac_format_ok = 77%` roughly 40% of pair-prompt comparisons are
format-censored; at 100% none are.

### 1.2 What the existing runs say — less than the summary claims

* In the horizon-2 gated run, `frac_format_ok` went 9.6% -> 77% and was
  **still climbing at step 300**. The format axis had not saturated. So the
  hypothesis "correctness gets optimised after format plateaus" was never in a
  position to be tested by that run. The run was stopped in the format phase.
* The correctness-only run (#4) removes the format axis entirely — the
  cleanest possible test of the hypothesis — but it was run on **horizon-1,
  where the base model is already at ~81%**. A flat 0.81 there says nothing.
* The summary's "objective formulation is the bottleneck -> refuted" therefore
  rests on one run that could not move (ceiling) and one that was stopped
  early. The **correctness-only, horizon-2** cell of the matrix is empty.

Calibration for "how many steps is enough": h1's GRPO takes horizon-2 accuracy
from ~35% to ~56% in 200 steps ([h1 paper](https://arxiv.org/abs/2510.07312)),
but each GRPO step uses 16 sampled completions per prompt with token-level
credit assignment. ES at pop 256 gets one scalar per rollout and estimates the
gradient from 128 random rank-1 directions; the repo's own paper-default math
config uses pop 1024 x 16 prompts (16k rollouts/step, 8x the h1 runs here).
Several hundred ES steps in the format phase is not a lot.

### 1.3 The two runs that settle it

| Script | What it tests | Cost |
| --- | --- | --- |
| `submit_h1_stage2_len2_gated_resume.sh` | The literal claim: continue the gated run from `checkpoint_step_299` to step 899. Does correctness move once `frac_format_ok` saturates? | 4 GPU, ~12-20 h (checkpoints every 50 steps; re-submit with `RESUME_FROM` if cut off) |
| `submit_h1_stage2_len2_correctonly.sh` | The missing matrix cell: correctness-only fitness, horizon-2, 600 steps from base. No format axis exists, so "format first" cannot apply. Also the fp32-master A/B (Section 3.3). | 4 GPU, ~12 h per arm |

The metric that makes this quantitative is new (`es_diagnostics.py`, logged
every step as `diag/*`):

* **`diag/frac_correct/cos_with_fitness`** — cosine between the pair-difference
  vector of *correctness* and the pair-difference vector of the *fitness
  actually optimised*. It is the fraction of the update direction that is
  about correctness. The professor's hypothesis predicts it rises toward 1 as
  `diag/frac_format_ok/pairs_with_signal` falls toward 0.
* **`diag/frac_correct/pairs_with_signal`** — fraction of antithetic pairs
  that disagree on correctness on at least one prompt. This is the raw amount
  of correctness signal ES has per step. If it is ~2% at pop 256, ES is
  estimating the correctness gradient from ~3 pairs per step.

Decision table after the resume run:

| `cos_with_fitness` (correctness) | `frac_correct` | Reading |
| --- | --- | --- |
| rises toward 1 | rises | Professor was right; it was the format phase. Run longer / drop format. |
| rises toward 1 | flat | The update *is* pointing at correctness and still nothing moves: capability-side limit (sigma / rank / population / bf16 rounding — Section 3.3) or collapse (Section 2). |
| stays ~0 | flat | Format never saturates or correctness never enters `d`: check `pairs_with_signal`; probe sigma (Section 3.2). |

---

## 2. "Be mindful of entropy collapse"

### 2.1 What it means in RL, and what it means here

In RL fine-tuning of reasoning models, *entropy collapse* is the observed
pattern that the policy's token entropy drops sharply early in training,
exploration dies, and accuracy saturates soon after
([Cui et al. 2025, "The Entropy Mechanism of RL for Reasoning LMs"](https://arxiv.org/abs/2505.22617)).
The fix there is to stop the update from over-sharpening high-covariance
tokens (Clip-Cov / KL-Cov, DAPO's clip-higher).

ES with greedy decoding has a direct analogue, and it is worse:

* The only way two population members differ is if `+eps` and `-eps` flip at
  least one argmax somewhere. A token can only flip if the perturbation shifts
  its logit by more than the **margin** between the top-1 and top-2 logits.
* Format training makes exactly those template tokens (`<reasoning>`, the
  newlines, `</answer>`) near-certain — margins grow — and the ES step (which
  rewards whatever made the +/- members differ) generally makes outputs more
  deterministic along the whole path.
* As margins grow past what `sigma` can overcome, pairs produce **identical
  text**, `d_k = 0`, and the update is zero no matter how many iterations you
  run. The population has collapsed onto one output per prompt.

So the two comments are in tension on purpose: "run longer, *provided*
exploration is still alive when the format phase ends." If the model has
collapsed by then, more iterations cannot help, and the fix is on the
exploration side.

### 2.2 Why the existing `pair_disagree_rate` does not settle this

`pair_disagree_rate` rose 6% -> 29% in the gated run and was read as "signal is
healthy". But the gated reward is binary, so the fraction of pairs that
straddle the 0/1 boundary rises mechanically as `frac_gated` goes 3% -> 29%.
It measures fitness disagreement, not output diversity, and says nothing about
*which axis* the disagreement is on. Both gaps are now covered:

| Metric (per step, W&B `diag/...`) | Reads |
| --- | --- |
| `pair_identical_rate` | fraction of +/- pairs whose greedy text is byte-identical. **The collapse metric.** Rising toward 1 = gradient dying. |
| `pair_first_div_frac` | where along the output (fraction of length) a pair first differs. Drifting late = perturbations only touch the tail. |
| `distinct_outputs_frac` | distinct outputs per prompt / population size. |
| `entropy/margin`, `entropy/entropy_topk`, `entropy/frac_margin_lt_0p5` | greedy top1-top2 log-prob margin and top-k entropy along the generated path (needs `ENTROPY_TOPK=20`, i.e. vLLM `logprobs=20`). Margin up / entropy down over training = collapse. |
| `frac_correct/pair_disagree_rate`, `frac_format_ok/pair_disagree_rate` | the old metric split by axis. |

Baseline these on the base model with `submit_probe_sensitivity.sh` (Section
3.2) and compare with the gated run's `merged/step_299`. If step_299 has a
much larger margin and a much lower `p_text_changed` at the same sigma than
base, the gated run collapsed.

### 2.3 Levers if it has collapsed (in order of least change to the method)

1. **Larger `sigma`** — the probe (Section 3.2) tells you the sigma at which
   correctness becomes perturbable at all. If `p_correct_changed` is ~0 at
   1e-3 but not at 3e-3, `sigma=1e-3` cannot learn correctness on this model
   regardless of iterations.
2. **Temperature > 0 with common random numbers.** `SamplingParams` already
   uses the same `seed` for every request, so the +/- members draw the same
   random stream and differ only where the perturbation moved a sampled token
   — the ES equivalent of CRN variance reduction. `TEMPERATURE=0.6` with
   `samples_per_prompt=1` is a one-line change (`run_h1_curriculum.sh`).
3. **Keep format out of the objective.** Correctness-only fitness with h1's
   permissive legacy answer extractor never rewards determinism for its own
   sake.
4. A diversity/entropy term in the fitness (the ES analogue of an entropy
   bonus). Not implemented; only if 1-3 fail.

---

## 3. "Try it on your own code"

### 3.1 What he means

A result has three ingredients: the **algorithm** (EGGROLL), the **data**
(GSM-LongHorizon + h1's rewards) and the **code** (the authors' vLLM/Ray
pipeline, which you adopted wholesale). A code review, even a careful one,
only ever says "I did not find a bug." The stronger test is to write the
algorithm yourself, as simply as possible, and run it on the same data:

* same behaviour (format learned, correctness flat) -> it is a property of
  algorithm + data, and the borrowed code is exonerated;
* different behaviour -> one of the two implementations is wrong, and the
  *difference* between them is the lead.

The value comes from the second implementation making **different incidental
choices** — precision, how the perturbation is applied, decoding stack — so
that anything that depends on such a choice shows up.

### 3.2 What was added: `es_reference.py`

A ~400-line plain PyTorch/transformers re-implementation. No vLLM, Ray or
PEFT; the only import from the repo is `h1_rewards` (the objective — kept
identical on purpose). Deliberate differences from the pipeline:

* fp32 master weights (Section 3.3);
* the perturbation is added to the dense weight (`W + B@A`) rather than going
  through vLLM's separate LoRA path;
* plain batched HF greedy decoding.

Two modes:

**`probe`** — no training. For the base model and any merged checkpoint,
generates greedily on N prompts, then for several `sigma` values applies
`n_perturb` antithetic pairs and measures how often the perturbation changes
the text, the format, and the correctness (split into *fixed* wrong->right and
*broke* right->wrong), plus the greedy margin/entropy along the base path.
This is the single most informative cheap experiment available: it says,
before any training, **how much correctness signal a pair carries at
`sigma=1e-3`**, and whether the gated checkpoint has collapsed.

```
sbatch submit_probe_sensitivity.sh       # base vs gated step_299, 1 GPU, ~2 h
```

**`train`** — the ES loop at small scale on one GPU, with the same per-axis and
collapse diagnostics.

```
# must go up within ~10-30 steps if the loop works at all (the repo's own 'zeros' idea)
python es_reference.py train --task sanity --model Qwen/Qwen3-1.7B --pop 16 --n-prompts 2 --iters 30 --max-new-tokens 64

# the real objective, small: compare its curves with the pipeline's at equal pop/sigma/lr
python es_reference.py train --task gsm --model Qwen/Qwen3-1.7B --data GSM-LongHorizon/train_len_2.jsonl \
    --reward-mode float --fitness-mode correctness --pop 32 --n-prompts 4 --iters 100 --max-new-tokens 1024 \
    --save-every 25 --out runs/es_reference_len2
```

Checkpoints from `--save-every` are ordinary HF directories, so
`h1_gsm_eval.py` evaluates them the same way as the pipeline's merged ones.

### 3.3 A concrete candidate code issue: bf16 rounding of the ES step

Reading `apply_lora_es_update` for the reimplementation turned this up. The
fp32 ES step is cast to the model dtype and added **in place to the bf16
engine weights** (vLLM loads Qwen3 with `dtype="auto"` = bf16):

```python
grad_shard = gradient.to(dtype=target_param.dtype)   # bf16
target_param.data[start:end].add_(valid_grad)        # bf16 + bf16 -> rounded to bf16
```

bf16 has 8 bits of mantissa. For a weight of magnitude ~0.02 the spacing
between representable values is ~1.2e-4, so any per-entry update smaller than
~6e-5 rounds to **zero** — every step, with no accumulation across steps.

With the h1 config (pop 256, `sigma=1e-3`, `lr=2e-4`, std-normalised
fitness) the typical per-entry update is ~1.2e-5 and the coherent maximum
(every pair agreeing) is ~1.4e-4. Emulating round-to-nearest bf16 on Gaussian
weights (`N(0, 0.02)`, the scale of Qwen dense projections):

| per-entry update std | entries that change | realised / intended norm | cos(realised, intended) |
| --- | --- | --- | --- |
| 1e-5 (typical step) | 12% | 0.61 | 0.46 |
| 3e-5 | 32% | 0.94 | 0.74 |
| 1e-4 (coherent step) | 68% | 1.04 | 0.95 |

The effect is **asymmetric**: a strong, coherent per-step signal (many pairs
agreeing, e.g. format early on) mostly survives; a weak but consistent signal
(few pairs disagreeing on correctness) is quantised away every step and never
accumulates. That is a mechanism that would produce "format learned,
correctness flat" out of a perfectly correct algorithm, and it would be
invisible to a code review that checks the maths.

Caveats: the paper's runs use the same code path and do learn on their tasks,
so this is a plausible degradation, not a proven bug; and the reimplementation
above avoids it by construction (fp32 master), so a pipeline-vs-reference
disagreement would point straight at it.

Two ways to test it on the real pipeline, both added:

* **Measure it.** Every step now logs `diag/update/applied_frac` (fraction of
  bf16 entries that actually changed), `diag/update/realised_ratio`,
  `diag/update/cos_realised_intended` and `diag/update/intended_rms`. If
  `applied_frac` is ~0.1 and `cos` ~0.5, most of every step is being thrown
  away.
* **Remove it.** `FP32_MASTER=1` (`--fp32-master`) keeps an fp32 master copy
  of the ES-updated weights on the CPU; the step accumulates there and the
  bf16-rounded master is written back to the engine. Per-step `applied_frac`
  will still look low (bf16 storage is still quantised) but nothing is lost
  across steps. `submit_h1_stage2_len2_correctonly.sh` defaults to
  `FP32_MASTER=1`; run it once with `FP32_MASTER=0` for the A/B (an empty
  value also means off; the script uses `${FP32_MASTER-1}` without the colon
  so an explicitly empty value is not silently re-defaulted to 1).

Memory: an fp32 master of the 1.7B target weights is ~5.6 GB of host RAM;
resuming from a checkpoint re-initialises the master from the saved bf16
weights (sub-ulp residue is lost at that point only).

---

## 3.4 Probe results (2026-09-15, `results/probe_len2_*.json`)

32 `train_len_2` prompts, 8 antithetic pairs per sigma, rank 1, greedy, 1024
tokens. "Perturbed accuracy" = accuracy of the perturbed model averaged over
the 16 perturbed copies.

| | base Qwen3-1.7B | gated `step_299` |
| --- | --- | --- |
| accuracy on the 32 prompts | 40.6% | 28.1% |
| soft-format rate | 0% | 97% |
| mean tokens | 340 | 292 |
| greedy margin (nats, mean) | 18.4 | 17.9 |
| tokens with margin < 0.5 | 1.5% | 1.5% |
| entropy (nats/token) | 0.063 | 0.070 |
| **sigma = 1e-3** (the training sigma) | | |
| text changed by a perturbation | 99.6% | 99.8% |
| first divergence (fraction of output) | 8% | 11% |
| correctness fixed / broke | 8.0% / 11.1% | 13.7% / 9.4% |
| perturbed accuracy | 37.5% | 32.4% |
| pairs disagreeing on correctness | 26% | 27% |
| pairs disagreeing on format | 16% | 37% |
| identical text within a pair | 0% | 0% |
| **sigma = 3e-3** | | |
| perturbed accuracy | 14.6% | 12.5% |
| soft-format among perturbed outputs | 39% | 67% |
| **sigma >= 1e-2** | | |
| perturbed accuracy | 0% | 0% |

What this settles:

1. **No entropy collapse.** Margin, entropy and the share of close-race tokens
   are the same before and after 300 gated steps; no antithetic pair produced
   identical text at any sigma; a perturbation changes the text 99.8% of the
   time. The run did not go deterministic. "Run longer" is not blocked by
   collapse.
2. **Correctness signal is plentiful, not absent.** At the training sigma,
   26% of pair-prompt comparisons disagree on correctness. With 128 pairs x 8
   prompts that is ~260 correctness disagreements per ES step. The problem is
   not that ES cannot see correctness.
3. **sigma = 1e-3 is already in the destructive regime, so a larger sigma is
   off the table.** A random perturbation of that size breaks 27% of the
   answers the base gets right and fixes 13% of the ones it gets wrong; the
   average perturbed model is 3 points *worse* than the centre. At 3e-3 it is
   26 points worse, at 1e-2 everything is wrong. The fitness response to
   +eps / -eps is dominated by symmetric breakage (both sides get worse), which
   the antithetic difference cancels but which leaves the sign of each pair's
   difference close to a coin flip. That is the low signal-to-noise picture:
   lots of disagreement, little consistent direction. Combined with the
   held-out churn (63 fixed / 63 broke, net 0), it says ES is random-walking on
   the correctness axis. The arm worth trying is a *smaller* sigma (3e-4) with
   lr scaled down to match, which needs the fp32 master to survive bf16.
4. **Format is partly "learnable by damage".** Perturbing the base at 3e-3
   collapses accuracy to 15% yet 39% of those broken outputs match the soft
   format (base: 0%). The likely mechanism is the degraded model echoing the
   format template that the system prompt spells out. Directions that damage
   the model therefore look good on the format axis. `--save-examples` on the
   probe now stores output triples so this can be read directly.
5. **Format at step_299 is fragile** (21% of perturbations break it, 37% of
   pairs disagree on it), so under the gated reward the format axis does not
   "saturate" and stop contributing: the update keeps being spent on repairing
   format. The professor's sequential picture is right in spirit but the
   hand-off to correctness may never come under that objective.
6. `step_299` is worse than base on training prompts here (28% vs 41% on 32
   prompts; noisy, but the sign matches held-out) and a random perturbation of
   it *improves* correctness on average. Format training pushed it downhill on
   the correctness axis.

Consequences for the plan: the correctness-only horizon-2 run is now the
priority (no format axis to spend the update on; the probe says the signal is
there), with the fp32-master A/B, plus a sigma = 3e-4 arm. The gated resume is
still the literal test of "run longer", but points 4-5 predict it will keep
paying for format.

## 3.5 Is the fp32 master "against the point" of EGGROLL? (2026-09-15)

Checked against the authors' own code rather than the paper text (arxiv is
blocked from the sandbox; `ESHyperscale/nano-egg` was cloned read-only).

* **This repo is the official LLM code** (README: "official code for the
  transformer LLM experiments"), and its update path is the same bf16 in-place
  add we measured. The official launch recipe (`slurm_launch_base_n1.sh`:
  Qwen3-4B, sigma 1e-3, lr 2e-4, pop 1024, prompt batch 16, r 1,
  `--normalize-with-std`, no `--scale-lr-in-grad`) gives a per-entry step of
  roughly lr/sqrt(P) ~ 6e-6, i.e. *smaller* than our 256-member runs (~1.2e-5),
  against a bf16 half-ulp of ~6e-5 for |w| ~ 0.02. So the paper's own LLM
  results were obtained under the same rounding: the update that lands is a
  sparsified, magnitude-thresholded version of the intended one (only entries
  with |w| below a few 1e-3 can move at all in one step). Learning still
  happened there, so the rounding is lossy, not fatal; the A/B is informative,
  not decisive by construction.
* **The paper's pure-int8 result does not use silent rounding.** `nano-egg`'s
  update (`_common_update`, `convert_fitnesses`) is a deliberately discrete
  rule: per antithetic pair the fitness is sign(F+ - F-); per weight entry an
  integer vote Z = sum_k sign(dF_k) * sign(noise_k) is accumulated; the weight
  moves by exactly one integer quantum in the direction of Z if |Z| exceeds a
  Z-test threshold scaled by sqrt(pop), otherwise it does not move. The
  README calls `alpha` (the fraction of parameters allowed to move per step)
  "analogous to learning rate". No fp32 copy, no accumulator, but also no
  lost updates: the step is always one representable unit. bf16
  round-to-nearest with sub-ulp steps is an *uncontrolled* version of the same
  idea, with the threshold set by |w| (ulp grows with the weight) rather than by
  statistical significance, and with no control of the moving fraction.
* **Verdict.** The fp32 master does not touch what EGGROLL is for (low-rank
  perturbations so the population is cheap; forward passes only, so it runs on
  an inference engine in bf16). It only changes how the population *centre* is
  stored between steps, which is the standard mixed-precision recipe. It does
  give up "no extra state" (4 bytes/param on the CPU). If the fp32 arm wins,
  the in-spirit follow-ups are (a) stochastic rounding of the bf16 add
  (unbiased, no extra memory) or (b) porting the int8 thresholded one-ulp rule
  to bf16; (c) `--scale-lr-in-grad` (multiplies the step by sqrt(P), so the
  per-entry step is ~lr instead of lr/sqrt(P)) lifts steps above the ulp but is
  a learning-rate change, not a precision fix.
* Related: "EGGROLL, Unrolled" (arXiv 2609.10980, Sep 2026) analyses the
  rank-1 geometry and proposes a leave-one-out estimator (LOO-ROLL) that halves
  estimator MSE at equal cost; not read in full (blocked), flagged for later.

## 4. Suggested order

1. ~~`sbatch submit_probe_sensitivity.sh`~~ **done** (Section 3.4): no
   collapse, plenty of correctness signal, sigma cannot go up.
2. `sbatch submit_h1_stage2_len2_correctonly.sh` three ways (4 GPU each,
   separate output dirs, derived automatically as
   `runs/h1_curriculum_len2_correctonly_{fp32master|bf16}_sigma<sigma>`):
   `FP32_MASTER=1` (default), `FP32_MASTER=0` (bf16 A/B), and
   `SIGMA=0.0003 LEARNING_RATE=0.00006` (smaller sigma, fp32 master).
   Watch `reward/frac_correct`, `diag/frac_correct/cos_with_fitness` (should be
   ~1 since correctness is the only axis), `diag/update/applied_frac`.
3. `sbatch submit_h1_stage2_len2_gated_resume.sh` (4 GPU) if budget allows:
   the literal "run longer" test. Prediction from the probe: format keeps
   absorbing the update (`diag/frac_format_ok/pairs_with_signal` stays high).
4. `es_reference.py probe --save-examples 8` on 8 prompts (1 GPU, minutes) to
   read what the sigma = 3e-3 "formatted but wrong" outputs look like.
5. `es_reference.py train --task gsm` small run vs. the pipeline at equal
   settings (1 GPU). The "own code" comparison proper.

Then update the decision tables in Sections 1.3 and 2.2.

---

## 5. Files added / changed in this pass

| File | Change |
| --- | --- |
| `es_diagnostics.py` | new: pure-numpy per-axis signal decomposition, greedy-collapse stats, entropy/margin from top-k logprobs. Unit-tested (`python -m unittest tests.test_es_diagnostics`). |
| `es_lora_multinode.py` | logs `diag/*` every step; `--entropy-topk N` requests vLLM logprobs; `--fp32-master` keeps fp32 master weights; `diag/update/*` measures how much of each step survives bf16. Default behaviour unchanged when the flags are off. |
| `tasks.py` | `GSMLongHorizonTask.get_state/restore_state` so `--resume-from` continues through the dataset instead of restarting at question 0. |
| `run_h1_curriculum.sh` | `ENTROPY_TOPK`, `FP32_MASTER`, `RESUME_FROM` passthroughs. |
| `es_reference.py` | new: independent re-implementation (`probe`, `train`). Core tested in `tests/test_es_reference_core.py` (perturbation scale, antithetic symmetry, exact restore in bf16, update aligned with the true negative gradient and descending on a quadratic, fp32 master accumulation). |
| `submit_probe_sensitivity.sh`, `submit_h1_stage2_len2_correctonly.sh`, `submit_h1_stage2_len2_gated_resume.sh` | new Slurm jobs for Section 4. |

What was **not** verified here: this environment has no GPU, no vLLM and no
model weights, so the trainer changes were checked by compilation and by
re-reading the patched regions only. `es_reference.py` was run end to end
(`probe`, `train --task sanity`, `train --task gsm` with checkpoint saving) on
a tiny randomly-initialised Qwen2 model on CPU, and its ES core passes the
unit tests above; it has not yet been run on a real model. Run
`run_h1_smoke.sh` (with `ENTROPY_TOPK=20 FP32_MASTER=1`) before the long jobs.
