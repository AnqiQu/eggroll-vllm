"""
es_diagnostics.py -- pure-numpy diagnostics for greedy antithetic ES (EGGROLL).

Everything here is side-effect free and independent of vLLM/Ray so it can be
unit-tested on a laptop (see tests/test_es_diagnostics.py). es_lora_multinode.py
calls `step_diagnostics` once per ES step and logs the result to W&B under the
`diag/` prefix.

Population layout convention (same as es_lora_multinode.py): population member
2k is the +noise member and 2k+1 the -noise member of antithetic pair k. Arrays
are shaped (population_size, num_prompts).

What the metrics answer
-----------------------
1. "Is the fitness signal about correctness or about format?"
   `signal_decomposition`: the ES update is  sum_k (F+_k - F-_k) * eps_k, i.e. a
   linear function of the vector of pair fitness differences d_fit. For every
   logged reward component c we form the same vector d_c and report
     cos(d_c, d_fit)          -- how aligned the update is with that axis
     share_of_update          -- <d_c, d_fit> / ||d_fit||^2 (regression share)
     pairs_with_signal        -- fraction of pairs whose +/- members differ on c
   If cos(d_correct, d_fit) ~ 0 while cos(d_format, d_fit) ~ 1, the update
   literally contains no correctness information that step.

2. "Has exploration collapsed?" (greedy decoding: the perturbation is the ONLY
   source of diversity, so if it stops flipping tokens the gradient is dead)
   `greedy_collapse_stats`:
     pair_identical_rate      -- +/- members produced byte-identical text
     pair_first_div_frac      -- where (fraction of length) the pair first differs
     distinct_outputs_frac    -- distinct outputs per prompt / population size
   `entropy_stats_from_rows` (needs top-k logprobs from the engine):
     entropy_topk, top1_prob, margin (logp1 - logp2), frac_margin_lt_0p5
   A rising margin / falling entropy while pair_identical_rate rises is the ES
   analogue of policy-entropy collapse.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np


# ----------------------------------------------------------------------------
# Antithetic-pair helpers
# ----------------------------------------------------------------------------
def pair_split(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(P, ...) -> (+members (P/2, ...), -members (P/2, ...))."""
    arr = np.asarray(arr)
    assert arr.shape[0] % 2 == 0, f"population must be even, got {arr.shape[0]}"
    return arr[0::2], arr[1::2]


def pair_disagreement(arr: np.ndarray) -> dict[str, float]:
    plus, minus = pair_split(np.asarray(arr, dtype=float))
    return {
        "disagree_rate": float(np.mean(plus != minus)),
        "absdiff_mean": float(np.mean(np.abs(plus - minus))),
    }


def center_per_prompt(arr: np.ndarray) -> np.ndarray:
    """Per-prompt (column) mean-centering then average over prompts -> (P,).
    Mirrors the fitness shaping in es_lora_multinode.py (before optional std)."""
    arr = np.asarray(arr, dtype=float)
    return np.mean(arr - arr.mean(axis=0, keepdims=True), axis=1)


def pair_diff_vector(arr: np.ndarray) -> np.ndarray:
    """d = F+ - F- for each pair, after per-prompt centering -> (P/2,)."""
    f = center_per_prompt(arr)
    plus, minus = pair_split(f)
    return plus - minus


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def signal_decomposition(fitness: np.ndarray, components: dict[str, np.ndarray]) -> dict[str, float]:
    """Decompose the ES update's fitness-difference vector along reward axes.

    fitness:    (P, N) the scalar the ES actually optimises
    components: name -> (P, N) any logged reward component (e.g. correctness,
                format); names may carry a 'reward/' prefix which is stripped.
    """
    fitness = np.asarray(fitness, dtype=float)
    d_fit = pair_diff_vector(fitness)
    out: dict[str, float] = {}
    plus_f, minus_f = pair_split(fitness)
    out["fitness/pairs_with_signal"] = float(np.mean(np.any(plus_f != minus_f, axis=1)))
    out["fitness/pair_diff_norm"] = float(np.linalg.norm(d_fit))
    denom = float(np.dot(d_fit, d_fit))
    for raw_name, comp in components.items():
        name = raw_name.split("/", 1)[1] if raw_name.startswith("reward/") else raw_name
        comp = np.asarray(comp, dtype=float)
        if comp.shape != fitness.shape:
            continue
        d_c = pair_diff_vector(comp)
        plus_c, minus_c = pair_split(comp)
        out[f"{name}/pairs_with_signal"] = float(np.mean(np.any(plus_c != minus_c, axis=1)))
        out[f"{name}/pair_disagree_rate"] = float(np.mean(plus_c != minus_c))
        out[f"{name}/pair_diff_norm"] = float(np.linalg.norm(d_c))
        out[f"{name}/cos_with_fitness"] = _cos(d_c, d_fit)
        out[f"{name}/share_of_update"] = float(np.dot(d_c, d_fit) / denom) if denom > 0 else 0.0
    # Correctness-vs-format alignment, if both are present.
    if "reward/frac_correct" in components and "reward/frac_format_ok" in components:
        dc = pair_diff_vector(components["reward/frac_correct"])
        df = pair_diff_vector(components["reward/frac_format_ok"])
        out["correct_vs_format/cos"] = _cos(dc, df)
    return out


# ----------------------------------------------------------------------------
# Greedy-decoding collapse statistics (text / token based; no logprobs needed)
# ----------------------------------------------------------------------------
def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    """Index of the first differing token; None if identical."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) == len(b):
        return None
    return n


def greedy_collapse_stats(texts: Sequence[Sequence[str]],
                          token_ids: Sequence[Sequence[Sequence[int]]] | None = None) -> dict[str, float]:
    """texts[p][n] is the (first-sample) output of population member p on prompt n."""
    P = len(texts)
    if P == 0:
        return {}
    N = len(texts[0])
    out: dict[str, float] = {}
    if P % 2 == 0 and P >= 2:
        identical = []
        for k in range(P // 2):
            for n in range(N):
                identical.append(texts[2 * k][n] == texts[2 * k + 1][n])
        out["pair_identical_rate"] = float(np.mean(identical)) if identical else 0.0
        if token_ids is not None:
            fracs, toks = [], []
            for k in range(P // 2):
                for n in range(N):
                    a, b = token_ids[2 * k][n], token_ids[2 * k + 1][n]
                    j = first_divergence(a, b)
                    if j is not None:
                        L = max(len(a), len(b), 1)
                        fracs.append(j / L)
                        toks.append(j)
            out["pair_first_div_frac"] = float(np.mean(fracs)) if fracs else 1.0
            out["pair_first_div_tok"] = float(np.mean(toks)) if toks else 0.0
    distinct = [len({texts[p][n] for p in range(P)}) / P for n in range(N)]
    out["distinct_outputs_frac"] = float(np.mean(distinct))
    return out


# ----------------------------------------------------------------------------
# Entropy / margin from top-k logprobs
# ----------------------------------------------------------------------------
def _lp_value(v: Any) -> float:
    """vLLM returns dict[token_id -> Logprob(logprob=..)] per token; also accept floats."""
    if isinstance(v, (int, float)):
        return float(v)
    lp = getattr(v, "logprob", None)
    if lp is None and isinstance(v, dict):
        lp = v.get("logprob")
    return float(lp)


def topk_logprob_rows(per_token: Iterable[Any] | None, k: int | None = None) -> list[list[float]]:
    """Convert per-token {id: Logprob} dicts into rows of descending logprobs."""
    rows: list[list[float]] = []
    if per_token is None:
        return rows
    for tok in per_token:
        if tok is None:
            continue
        vals = sorted((_lp_value(v) for v in tok.values()), reverse=True)
        if k is not None:
            vals = vals[:k]
        if vals:
            rows.append(vals)
    return rows


def entropy_stats_from_rows(rows: Sequence[Sequence[float]], low_margin: float = 0.5) -> dict[str, float]:
    """Per-response token-averaged stats from rows of descending logprobs.

    entropy_topk : -sum p_i log p_i over the returned top-k (a lower bound on the
                   full entropy; tail mass is ignored)
    top1_prob    : exp(logp1)
    margin       : logp1 - logp2 (nats). Under greedy decoding a token can only
                   flip if a perturbation shifts logits by more than this.
    frac_margin_lt_0p5 : fraction of tokens with margin < `low_margin`
    """
    ents, p1s, margins, low = [], [], [], []
    for r in rows:
        if not r:
            continue
        lp = np.asarray(r, dtype=float)
        p = np.exp(lp)
        ents.append(float(-np.sum(p * lp)))
        p1s.append(float(p[0]))
        if len(lp) > 1:
            m = float(lp[0] - lp[1])
            margins.append(m)
            low.append(m < low_margin)
    n = len(ents)
    if n == 0:
        return {"n_tokens": 0.0}
    return {
        "n_tokens": float(n),
        "entropy_topk": float(np.mean(ents)),
        "top1_prob": float(np.mean(p1s)),
        "margin": float(np.mean(margins)) if margins else float("nan"),
        "frac_margin_lt_0p5": float(np.mean(low)) if low else float("nan"),
    }


def aggregate_entropy(per_response: Sequence[dict[str, float]]) -> dict[str, float]:
    """Token-weighted average of `entropy_stats_from_rows` outputs."""
    keys = ("entropy_topk", "top1_prob", "margin", "frac_margin_lt_0p5")
    tot = 0.0
    acc = {k: 0.0 for k in keys}
    for d in per_response:
        n = float(d.get("n_tokens", 0.0))
        if n <= 0:
            continue
        tot += n
        for k in keys:
            v = d.get(k, float("nan"))
            if not (isinstance(v, float) and math.isnan(v)):
                acc[k] += n * v
    if tot == 0.0:
        return {}
    out = {f"entropy/{k}": acc[k] / tot for k in keys}
    out["entropy/n_tokens"] = tot
    return out


# ----------------------------------------------------------------------------
# One-call per-step summary
# ----------------------------------------------------------------------------
def step_diagnostics(fitness: np.ndarray,
                     components: dict[str, np.ndarray] | None = None,
                     texts: Sequence[Sequence[str]] | None = None,
                     token_ids: Sequence[Sequence[Sequence[int]]] | None = None,
                     entropy_per_response: Sequence[dict[str, float]] | None = None,
                     prefix: str = "diag/") -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(signal_decomposition(fitness, components or {}))
    if texts is not None and len(texts) > 0:
        out.update(greedy_collapse_stats(texts, token_ids))
    if entropy_per_response:
        out.update(aggregate_entropy(entropy_per_response))
    return {f"{prefix}{k}": float(v) for k, v in out.items()}


def format_diagnostics(d: dict[str, float]) -> str:
    """Compact one-line summary for stdout."""
    keys = [
        "diag/frac_correct/cos_with_fitness", "diag/frac_format_ok/cos_with_fitness",
        "diag/frac_correct/pairs_with_signal", "diag/frac_format_ok/pairs_with_signal",
        "diag/pair_identical_rate", "diag/pair_first_div_frac", "diag/distinct_outputs_frac",
        "diag/entropy/margin", "diag/entropy/entropy_topk",
    ]
    parts = [f"{k.replace('diag/', '')}={d[k]:.3f}" for k in keys if k in d]
    return "DIAG: " + ", ".join(parts)
