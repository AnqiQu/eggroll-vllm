#!/usr/bin/env python
"""
es_reference.py -- an INDEPENDENT, minimal re-implementation of EGGROLL
(antithetic low-rank Evolution Strategies on LoRA-shaped perturbations) in plain
PyTorch + transformers. No vLLM, no Ray, no PEFT, and no code shared with
es_lora_multinode.py. The only thing imported from the repo is h1_rewards
(the objective/data side of the experiment, deliberately kept identical).

Why this exists ("try it on your own code")
-------------------------------------------
The research question has three ingredients: the ALGORITHM (EGGROLL), the DATA
(GSM-LongHorizon + h1 rewards) and the CODE (the authors' vLLM pipeline).
A code review can only say "I did not find a bug". An independent
re-implementation that reproduces the same behaviour on the same data says
"the behaviour is a property of the algorithm + data, not of the borrowed
code" -- and if it does NOT reproduce it, one of the two implementations is
wrong and the difference is a lead. Known differences here, by design:
  * updates accumulate in fp32 master weights (the vLLM path adds the fp32
    step into bf16 weights, where sub-half-ulp entries round to zero);
  * perturbations are applied to the dense weight (W + B@A) instead of through
    vLLM's separate LoRA path;
  * plain batched HF greedy decoding instead of vLLM.

Modes
-----
probe   No training. Measures how often a sigma-sized perturbation changes the
        greedy text / format / correctness on N prompts, at several sigmas, and
        the greedy top1-top2 log-prob margin + entropy along the path (the
        entropy-collapse monitor). Run it on the base model AND on merged
        checkpoints (runs/.../merged/step_N) to compare.

train   The ES loop at small scale on one GPU, logging per step: fitness,
        frac_correct, frac_format_ok, antithetic-pair disagreement split by
        axis, greedy-collapse stats, and update stats. --task sanity is a
        trivially learnable objective (fraction of '0' characters) that must go
        up if the loop works at all; --task gsm is the h1 objective.

Examples
--------
  python es_reference.py probe --model Qwen/Qwen3-1.7B \
      --data GSM-LongHorizon/train_len_2.jsonl --reward-mode float \
      --n-prompts 32 --n-perturb 8 --sigmas 0.001 0.003 0.01 --out results/probe_base.json

  python es_reference.py train --task sanity --model Qwen/Qwen3-1.7B \
      --pop 16 --n-prompts 2 --iters 30 --max-new-tokens 64

  python es_reference.py train --task gsm --model Qwen/Qwen3-1.7B \
      --data GSM-LongHorizon/train_len_2.jsonl --reward-mode float \
      --fitness-mode correctness --pop 32 --n-prompts 4 --iters 100 \
      --max-new-tokens 1024 --save-every 25 --out runs/es_reference_len2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# =============================================================================
# Model helpers
# =============================================================================
def load_model_and_tokenizer(name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)        # transformers >= 4.56
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)  # older transformers
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def es_target_params(model) -> list[tuple[str, torch.nn.Parameter]]:
    """The same 7 projection matrices per layer that es_lora_multinode perturbs."""
    out = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and name.endswith(".weight") and any(f".{t}." in name for t in TARGET_MODULES):
            out.append((name, p))
    if not out:
        raise RuntimeError("no ES target parameters found (expected *_proj weights)")
    return out


def stop_token_ids(model, tok) -> set[int]:
    ids = set()
    eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif isinstance(eos, (list, tuple)):
        ids.update(int(x) for x in eos)
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    if tok.pad_token_id is not None:
        ids.add(int(tok.pad_token_id))
    return ids


# =============================================================================
# The ES algorithm (written from the paper's description, not from the repo)
# =============================================================================
class LowRankES:
    """Antithetic low-rank ES over a list of 2-D weights with fp32 master weights.

    Perturbation for pair k on weight W (out x in):  eps_k = B_k @ A_k with
      A_k ~ N(0, 1)^(r x in)  * sqrt(sigma)
      B_k ~ N(0, 1)^(out x r) * sqrt(sigma / r)
    so every entry of eps_k has std sigma. Member 2k gets +eps_k, 2k+1 gets -eps_k.
    Update:  W <- W + lr / (pop * sigma) * sum_k (F+_k - F-_k) * eps_k
    """

    def __init__(self, params, sigma: float, rank: int, seed: int, master_device: str | None = None):
        self.params = params
        self.sigma, self.rank, self.seed = float(sigma), int(rank), int(seed)
        self.master = [p.detach().to(master_device or p.device, torch.float32).clone() for _, p in params]

    def _factors(self, step: int, pair: int, li: int, shape):
        out_dim, in_dim = shape
        g = torch.Generator().manual_seed(((self.seed * 1_000_003 + step) * 100_003 + pair) * 1_009 + li)
        A = torch.randn(self.rank, in_dim, generator=g) * math.sqrt(self.sigma)
        B = torch.randn(out_dim, self.rank, generator=g) * math.sqrt(self.sigma / self.rank)
        return A, B

    def delta(self, step: int, pair: int, li: int, device) -> torch.Tensor:
        A, B = self._factors(step, pair, li, tuple(self.params[li][1].shape))
        return B.to(device) @ A.to(device)  # fp32 (out, in)

    @torch.no_grad()
    def set_member(self, step: int, pair: int, sign: int) -> None:
        for li, (_, p) in enumerate(self.params):
            d = self.delta(step, pair, li, p.device)
            p.copy_((self.master[li].to(p.device) + sign * d).to(p.dtype))

    @torch.no_grad()
    def restore(self) -> None:
        for li, (_, p) in enumerate(self.params):
            p.copy_(self.master[li].to(p.device, p.dtype))

    @torch.no_grad()
    def update(self, step: int, pair_diffs: np.ndarray, lr: float, pop_size: int) -> dict[str, float]:
        coef = lr / (pop_size * self.sigma)
        n_tot, changed, sq = 0, 0.0, 0.0
        for li, (_, p) in enumerate(self.params):
            acc = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
            for k, d in enumerate(pair_diffs):
                if d == 0.0:
                    continue
                acc.add_(self.delta(step, k, li, p.device), alpha=float(d))
            acc.mul_(coef)
            m = self.master[li]
            m.add_(acc.to(m.device))
            before = p.detach().clone()
            p.copy_(m.to(p.device, p.dtype))
            n_tot += p.numel()
            changed += float((p != before).sum())
            sq += float((acc * acc).sum())
            del acc, before
        return {"update_applied_frac": changed / max(n_tot, 1), "update_rms": math.sqrt(sq / max(n_tot, 1))}


def shape_fitness(F: np.ndarray, normalize_with_std: bool) -> np.ndarray:
    """(pop, n_prompts) -> (pop,): per-prompt mean-centred, averaged over prompts."""
    Fc = (F - F.mean(axis=0, keepdims=True)).mean(axis=1)
    if normalize_with_std:
        Fc = Fc / (Fc.std() + 1e-8)
    return Fc


def pair_diffs(Fc: np.ndarray) -> np.ndarray:
    return Fc[0::2] - Fc[1::2]


# =============================================================================
# Generation + path statistics
# =============================================================================
@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int, stop_ids: set[int]):
    texts, ids = [], []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        gen = out[:, enc["input_ids"].shape[1]:]
        for r in range(gen.shape[0]):
            row = gen[r].tolist()
            cut = next((j for j, t in enumerate(row) if t in stop_ids), len(row))
            row = row[:cut]
            ids.append(row)
            texts.append(tok.decode(row, skip_special_tokens=True))
    return texts, ids


@torch.no_grad()
def path_stats(model, tok, prompt: str, gen_ids: list[int], low_margin: float = 0.5) -> dict[str, float]:
    """Teacher-forced pass over prompt+generation: full-vocab entropy and the
    top1-top2 log-prob margin at every generated position. Under greedy decoding
    the margin is the logit shift a perturbation must exceed to flip that token."""
    if not gen_ids:
        return {"n_tokens": 0}
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    full = torch.tensor([p_ids + gen_ids], device=model.device)
    logits = model(full).logits[0, len(p_ids) - 1: len(p_ids) - 1 + len(gen_ids)].float()
    lp = torch.log_softmax(logits, dim=-1)
    top2 = torch.topk(lp, 2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    ent = -(lp.exp() * lp).sum(-1)
    return {
        "n_tokens": len(gen_ids),
        "entropy": float(ent.mean()),
        "margin": float(margin.mean()),
        "frac_margin_lt_0p5": float((margin < low_margin).float().mean()),
        "top1_prob": float(top2[:, 0].exp().mean()),
    }


def aggregate_path_stats(stats: list[dict]) -> dict[str, float]:
    keys = ("entropy", "margin", "frac_margin_lt_0p5", "top1_prob")
    tot = sum(s.get("n_tokens", 0) for s in stats)
    if tot == 0:
        return {}
    out = {k: sum(s[k] * s["n_tokens"] for s in stats if s.get("n_tokens", 0) > 0) / tot for k in keys}
    out["n_tokens"] = float(tot)
    return out


def first_divergence_frac(a: list[int], b: list[int]) -> float | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i / max(len(a), len(b), 1)
    if len(a) == len(b):
        return None
    return n / max(len(a), len(b), 1)


# =============================================================================
# Tasks / objectives
# =============================================================================
class GSMTask:
    """h1 GSM-LongHorizon prompts + h1 rewards (identical to GSMLongHorizonTask)."""

    def __init__(self, data_path: str, tok, reward_mode: str, fitness_mode: str, strict_format: bool,
                 n_prompts: int, enable_thinking: bool = False):
        from h1_rewards import SYSTEM_PROMPT
        with open(data_path) as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        self.tok = tok
        self.system_prompt = SYSTEM_PROMPT
        self.float_mode = reward_mode == "float"
        self.fitness_mode = fitness_mode
        self.strict = strict_format
        self.n_prompts = n_prompts
        self.enable_thinking = enable_thinking
        self.idx = 0

    def _prompt(self, q: str) -> str:
        msgs = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": q.strip()}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=self.enable_thinking)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def batch(self, start: int | None = None):
        if start is None:
            start, self.idx = self.idx, self.idx + self.n_prompts
        rows = [self.rows[(start + i) % len(self.rows)] for i in range(self.n_prompts)]
        return [self._prompt(r["question"]) for r in rows], [r["final_answer"] for r in rows]

    def score(self, text: str, answer) -> tuple[float, dict[str, float]]:
        from h1_rewards import correctness_reward, format_reward, format_ok
        c = correctness_reward(text, answer)
        f = format_reward(text, self.float_mode)
        ok = format_ok(text, strict=self.strict)
        if self.fitness_mode == "total":
            fit = c + f
        elif self.fitness_mode == "gated":
            fit = 1.0 if (c > 0 and ok) else 0.0
        else:
            fit = c
        return float(fit), {"correct": float(c > 0), "format_ok": float(ok), "format_reward": float(f)}


class SanityTask:
    """Trivially learnable objective: fraction of '0' characters in the output.
    (The repo's own 'zeros' debugging task.) Must go up if the ES loop works."""

    def __init__(self, tok, n_prompts: int):
        self.tok = tok
        self.n_prompts = n_prompts
        self.questions = ["Print the digit 0 as many times as you can.",
                          "Write a long line of zeros.",
                          "Output only zeros.",
                          "Repeat the character 0."]

    def _prompt(self, q: str) -> str:
        msgs = [{"role": "user", "content": q}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except (TypeError, ValueError):
            return q + "\n"

    def batch(self, start: int | None = None):
        qs = [self.questions[i % len(self.questions)] for i in range(self.n_prompts)]
        return [self._prompt(q) for q in qs], [None] * self.n_prompts

    def score(self, text: str, answer) -> tuple[float, dict[str, float]]:
        fit = (sum(ch == "0" for ch in text) / len(text)) if text else 0.0
        return float(fit), {"correct": fit, "format_ok": 1.0 if text else 0.0}


# =============================================================================
# probe
# =============================================================================
def run_probe(args) -> None:
    dtype = getattr(torch, args.dtype)
    model, tok = load_model_and_tokenizer(args.model, args.device, dtype)
    params = es_target_params(model)
    stop_ids = stop_token_ids(model, tok)
    task = GSMTask(args.data, tok, args.reward_mode, args.fitness_mode, args.format_check == "strict", args.n_prompts)
    prompts, answers = task.batch(start=args.prompt_offset)
    N = len(prompts)
    es = LowRankES(params, sigma=args.sigmas[0], rank=args.rank, seed=args.seed, master_device=args.master_device)

    t0 = time.time()
    base_texts, base_ids = generate(model, tok, prompts, args.max_new_tokens, args.batch_size, stop_ids)
    base_scores = [task.score(t, a) for t, a in zip(base_texts, answers)]
    base_correct = np.array([s[1]["correct"] for s in base_scores])
    base_ok = np.array([s[1]["format_ok"] for s in base_scores])
    base_path = aggregate_path_stats([path_stats(model, tok, p, i) for p, i in zip(prompts, base_ids)])
    result = {
        "model": args.model, "data": args.data, "n_prompts": N, "rank": args.rank, "n_perturb": args.n_perturb,
        "base": {"accuracy": float(base_correct.mean()), "format_rate": float(base_ok.mean()),
                 "mean_tokens": float(np.mean([len(i) for i in base_ids])), **base_path},
        "sigmas": {},
    }
    print(f"[probe] base: acc={base_correct.mean():.3f} format={base_ok.mean():.3f} "
          f"tokens={result['base']['mean_tokens']:.0f} margin={base_path.get('margin', float('nan')):.3f} "
          f"entropy={base_path.get('entropy', float('nan')):.3f} ({time.time() - t0:.0f}s)", flush=True)

    for sigma in args.sigmas:
        es.sigma = float(sigma)
        t0 = time.time()
        text_changed, c_fixed, c_broke, ok_changed, first_div = [], [], [], [], []
        pair_fit, pair_c, pair_ok, pair_same = [], [], [], []
        for j in range(args.n_perturb):
            per_sign = {}
            for sign in (+1, -1):
                es.set_member(0, j, sign)
                texts, ids = generate(model, tok, prompts, args.max_new_tokens, args.batch_size, stop_ids)
                per_sign[sign] = [(t, i, task.score(t, a)) for t, i, a in zip(texts, ids, answers)]
            es.restore()
            for n in range(N):
                (tp, ip, (fp, cp)), (tm, im, (fm, cm)) = per_sign[+1][n], per_sign[-1][n]
                for t, i, comp in ((tp, ip, cp), (tm, im, cm)):
                    text_changed.append(t != base_texts[n])
                    c_fixed.append(base_correct[n] == 0 and comp["correct"] == 1)
                    c_broke.append(base_correct[n] == 1 and comp["correct"] == 0)
                    ok_changed.append(comp["format_ok"] != base_ok[n])
                    fd = first_divergence_frac(i, base_ids[n])
                    if fd is not None:
                        first_div.append(fd)
                pair_fit.append(fp != fm)
                pair_c.append(cp["correct"] != cm["correct"])
                pair_ok.append(cp["format_ok"] != cm["format_ok"])
                pair_same.append(tp == tm)
        summary = {
            "p_text_changed": float(np.mean(text_changed)),
            "p_correct_fixed": float(np.mean(c_fixed)),      # base wrong -> perturbed right
            "p_correct_broke": float(np.mean(c_broke)),      # base right -> perturbed wrong
            "p_correct_changed": float(np.mean(c_fixed) + np.mean(c_broke)),
            "p_format_changed": float(np.mean(ok_changed)),
            "first_div_frac_mean": float(np.mean(first_div)) if first_div else 1.0,
            "pair_fitness_disagree": float(np.mean(pair_fit)),
            "pair_correct_disagree": float(np.mean(pair_c)),
            "pair_format_disagree": float(np.mean(pair_ok)),
            "pair_identical_text": float(np.mean(pair_same)),
            "seconds": time.time() - t0,
        }
        result["sigmas"][str(sigma)] = summary
        print(f"[probe] sigma={sigma:g}: text_changed={summary['p_text_changed']:.3f} "
              f"correct fixed/broke={summary['p_correct_fixed']:.3f}/{summary['p_correct_broke']:.3f} "
              f"format_changed={summary['p_format_changed']:.3f} first_div={summary['first_div_frac_mean']:.2f} | "
              f"pairs disagree fit/correct/format={summary['pair_fitness_disagree']:.3f}/"
              f"{summary['pair_correct_disagree']:.3f}/{summary['pair_format_disagree']:.3f} "
              f"identical={summary['pair_identical_text']:.3f} ({summary['seconds']:.0f}s)", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[probe] wrote {args.out}")


# =============================================================================
# train
# =============================================================================
def run_train(args) -> None:
    dtype = getattr(torch, args.dtype)
    model, tok = load_model_and_tokenizer(args.model, args.device, dtype)
    params = es_target_params(model)
    stop_ids = stop_token_ids(model, tok)
    if args.task == "sanity":
        task = SanityTask(tok, args.n_prompts)
    else:
        task = GSMTask(args.data, tok, args.reward_mode, args.fitness_mode, args.format_check == "strict", args.n_prompts)
    es = LowRankES(params, sigma=args.sigma, rank=args.rank, seed=args.seed, master_device=args.master_device)
    assert args.pop % 2 == 0, "--pop must be even (antithetic pairs)"
    P, N = args.pop, args.n_prompts
    log_path = None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        log_path = os.path.join(args.out, "train_log.jsonl")
        with open(os.path.join(args.out, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    print(f"[train] {len(params)} target weights, pop={P}, prompts/step={N}, sigma={args.sigma}, "
          f"lr={args.lr}, rank={args.rank}, std-normalise={not args.no_normalize_std}", flush=True)

    for step in range(args.iters):
        t0 = time.time()
        noise_step = step // args.steps_per_adapter
        prompts, answers = task.batch()
        F = np.zeros((P, N))
        comp = {"correct": np.zeros((P, N)), "format_ok": np.zeros((P, N))}
        texts = [[None] * N for _ in range(P)]
        for member in range(P):
            pair, sign = member // 2, (+1 if member % 2 == 0 else -1)
            es.set_member(noise_step, pair, sign)
            out_texts, _ = generate(model, tok, prompts, args.max_new_tokens, args.batch_size, stop_ids)
            for n, (t, a) in enumerate(zip(out_texts, answers)):
                fit, c = task.score(t, a)
                F[member, n] = fit
                comp["correct"][member, n] = c["correct"]
                comp["format_ok"][member, n] = c["format_ok"]
                texts[member][n] = t
        es.restore()

        Fc = shape_fitness(F, not args.no_normalize_std)
        d = pair_diffs(Fc)
        ustats = es.update(noise_step, d, args.lr, P)

        plus, minus = F[0::2], F[1::2]
        rec = {
            "step": step,
            "mean_fitness": float(F.mean()),
            "frac_correct": float(comp["correct"].mean()),
            "frac_format_ok": float(comp["format_ok"].mean()),
            "pair_disagree_rate": float(np.mean(plus != minus)),
            "pairs_with_correct_signal": float(np.mean(np.any(comp["correct"][0::2] != comp["correct"][1::2], axis=1))),
            "pairs_with_format_signal": float(np.mean(np.any(comp["format_ok"][0::2] != comp["format_ok"][1::2], axis=1))),
            "pair_identical_rate": float(np.mean([texts[2 * k][n] == texts[2 * k + 1][n] for k in range(P // 2) for n in range(N)])),
            "distinct_outputs_frac": float(np.mean([len({texts[p][n] for p in range(P)}) / P for n in range(N)])),
            "mean_chars": float(np.mean([len(texts[p][n]) for p in range(P) for n in range(N)])),
            **ustats,
            "seconds": time.time() - t0,
        }
        print(f"[train] step {step}: fit={rec['mean_fitness']:.4f} correct={rec['frac_correct']:.3f} "
              f"format={rec['frac_format_ok']:.3f} pair_disagree={rec['pair_disagree_rate']:.3f} "
              f"(correct {rec['pairs_with_correct_signal']:.2f} / format {rec['pairs_with_format_signal']:.2f}) "
              f"identical={rec['pair_identical_rate']:.2f} distinct={rec['distinct_outputs_frac']:.2f} "
              f"upd_rms={rec['update_rms']:.2e} applied={rec['update_applied_frac']:.2f} ({rec['seconds']:.0f}s)", flush=True)
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if args.out and args.save_every and (step + 1) % args.save_every == 0:
            ckpt = os.path.join(args.out, f"step_{step}")
            model.save_pretrained(ckpt, safe_serialization=True)
            tok.save_pretrained(ckpt)
            print(f"[train] saved {ckpt}", flush=True)


# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--model", required=True)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
        p.add_argument("--master-device", default=None, help="where fp32 master weights live (default: model device)")
        p.add_argument("--rank", type=int, default=1)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--max-new-tokens", type=int, default=1024)
        p.add_argument("--batch-size", type=int, default=32, help="prompts per generate() call")
        p.add_argument("--data", default=None, help="GSM-LongHorizon jsonl split")
        p.add_argument("--reward-mode", default="float", choices=["int", "float"])
        p.add_argument("--fitness-mode", default="correctness", choices=["correctness", "total", "gated"])
        p.add_argument("--format-check", default="soft", choices=["soft", "strict"])
        p.add_argument("--n-prompts", type=int, default=8)
        p.add_argument("--out", default=None)

    pp = sub.add_parser("probe", help="perturbation-sensitivity + margin/entropy probe (no training)")
    common(pp)
    pp.add_argument("--sigmas", type=float, nargs="+", default=[0.001, 0.003, 0.01])
    pp.add_argument("--n-perturb", type=int, default=8, help="antithetic pairs per sigma")
    pp.add_argument("--prompt-offset", type=int, default=0)

    pt = sub.add_parser("train", help="small-scale ES loop")
    common(pt)
    pt.add_argument("--task", default="gsm", choices=["gsm", "sanity"])
    pt.add_argument("--pop", type=int, default=16)
    pt.add_argument("--iters", type=int, default=20)
    pt.add_argument("--sigma", type=float, default=0.001)
    pt.add_argument("--lr", type=float, default=0.0002)
    pt.add_argument("--steps-per-adapter", type=int, default=4, help="reuse the same noise for this many steps (as the repo does)")
    pt.add_argument("--no-normalize-std", action="store_true")
    pt.add_argument("--save-every", type=int, default=0)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "probe" or (args.mode == "train" and args.task == "gsm"):
        if not args.data:
            sys.exit("--data is required for the gsm objective")
    if args.mode == "probe":
        run_probe(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
