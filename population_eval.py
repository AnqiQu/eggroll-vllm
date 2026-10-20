#!/usr/bin/env python
"""
population_eval.py -- evaluate the CLOUD of models around a checkpoint, not
just the checkpoint ("treat the population as the result").

ES maximises the average fitness of theta + sigma*eps, so an ES checkpoint is
the centre of a cloud of models. This script samples that cloud with the same
EGGROLL-shaped noise ES trains with (eggroll_noise.py: rank-r, per-entry std
sigma, antithetic pairs, the 7 projection matrices of every layer), serves every
member as a LoRA adapter in ONE vLLM engine, and scores every member greedily on
held-out problems. For comparison it also samples the centre model at
temperature T (the token-space "population" GRPO explores with).

Run it on an ES checkpoint and on a GRPO checkpoint with the same --seed: both
get the same noise draws, so the comparison is paired on the perturbations too.

Questions it answers (per dataset, per sigma), all in the output "summary":
  coverage@k        fraction of problems solved by >=1 of k members (unbiased
                    pass@k estimator over the N members); the "set of solutions"
  rescued           centre wrong, >=1 member right (latent next-horizon skill)
  broken            centre right, most members wrong (how sharp the point is)
  member_acc_*      how much accuracy the cloud gives up vs the centre; compare
                    ES vs GRPO coverage at MATCHED member_acc_mean (sweep sigma)
  pair_disagree     antithetic pairs scoring differently: the raw ES signal
  group_signal@16   P(a random 16-member group is not all-right/all-wrong):
                    the fraction of GRPO-sized groups with a nonzero advantage.
                    The same fields on the "sampled" block give GRPO's own signal,
                    so on base + test_len_5 this doubles as the direct-jump probe.

Example (one GH200, ~30-60 min per dataset at N=32 on a 3B model):
  python population_eval.py --model runs/.../merged/step_500 --label es_s2_step500 \
      --datasets GSM-LongHorizon/test_len_2.jsonl GSM-LongHorizon/test_len_3.jsonl \
      --n-members 32 --sigmas 0.0005 0.001 0.002 --sample-temperature 1.0 \
      --out results/pop_es_s2_step500.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
import time


# =============================================================================
# Metrics (pure python; tested in tests/test_population_metrics.py)
# =============================================================================
def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k draws without replacement from n is correct), c correct."""
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def group_signal(n: int, c: int, k: int) -> float:
    """P(a random size-k group from n is mixed, i.e. has a nonzero group-relative advantage)."""
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    total = math.comb(n, k)
    return 1.0 - (math.comb(c, k) + math.comb(n - c, k)) / total


def _ks(n: int) -> list[int]:
    ks, k = [], 1
    while k < n:
        ks.append(k)
        k *= 2
    return ks + [n]


def summarize(correct: list[list[int]], centre: list[int], antithetic: bool = False,
              group_k: int = 16, preds: list[list] | None = None, centre_preds: list | None = None) -> dict:
    """correct: [member][problem] 0/1; centre: [problem] 0/1 (greedy centre model)."""
    n, p = len(correct), len(centre)
    assert n > 0 and all(len(r) == p for r in correct), "ragged correctness matrix"
    counts = [sum(correct[m][j] for m in range(n)) for j in range(p)]
    accs = [sum(r) / p for r in correct]
    mean = sum(accs) / n
    out = {
        "n_members": n,
        "n_problems": p,
        "centre_acc": sum(centre) / p,
        "member_acc_mean": mean,
        "member_acc_std": math.sqrt(sum((a - mean) ** 2 for a in accs) / n),
        "member_acc_min": min(accs),
        "member_acc_max": max(accs),
        "coverage": {str(k): sum(pass_at_k(n, c, k) for c in counts) / p for k in _ks(n)},
        "rescued": sum(1 for j in range(p) if not centre[j] and counts[j] > 0) / p,
        "broken": sum(1 for j in range(p) if centre[j] and counts[j] < n / 2) / p,
        "frac_problems_mixed": sum(1 for c in counts if 0 < c < n) / p,
        f"group_signal@{min(group_k, n)}": sum(group_signal(n, c, min(group_k, n)) for c in counts) / p,
        "counts": counts,
    }
    if antithetic:
        assert n % 2 == 0, "antithetic summary needs an even number of members"
        dis = [[int(correct[2 * k][j] != correct[2 * k + 1][j]) for j in range(p)] for k in range(n // 2)]
        out["pair_disagree"] = sum(map(sum, dis)) / (p * n // 2)
        out["frac_problems_with_pair_signal"] = sum(1 for j in range(p) if any(d[j] for d in dis)) / p
    if preds is not None and centre_preds is not None:
        changed = sum(1 for m in range(n) for j in range(p) if preds[m][j] != centre_preds[j])
        out["answer_changed_vs_centre"] = changed / (n * p)
    return out


# =============================================================================
# Generation
# =============================================================================
def load_jsonl(path: str, num_samples: int | None, seed: int):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    idx = list(range(len(rows)))
    if num_samples is not None and num_samples < len(rows):
        idx = sorted(random.Random(seed).sample(idx, num_samples))
    return idx, [rows[i] for i in idx]


def score(evaluator, text: str, gt: float):
    # same trimming + extraction + tolerance as h1_gsm_eval.GSM8KEvaluator
    if "</answer>" in text:
        text = text.split("</answer>")[0] + "</answer>"
    pred = evaluator.extract_answer_from_text(text.strip())
    return int(pred is not None and abs(pred - gt) < 1e-2), pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="merged HF dir or hub id (the centre)")
    ap.add_argument("--label", default=None, help="name stored in the output (default: basename of --model)")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--num-samples", type=int, default=None, help="random subset of problems per dataset")
    ap.add_argument("--n-members", type=int, default=32, help="population size (even; antithetic pairs)")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.001])
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="noise seed; keep fixed across checkpoints to pair them")
    ap.add_argument("--sample-temperature", type=float, default=1.0, help="0 disables the token-sampling baseline")
    ap.add_argument("--n-samples", type=int, default=None, help="samples of the centre (default: --n-members)")
    ap.add_argument("--group-k", type=int, default=16, help="GRPO group size for group_signal (h1: 16)")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-loras", type=int, default=16, help="adapters resident on GPU at once")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--adapter-dir", default=None, help="where member adapters are written (default: tmp)")
    ap.add_argument("--save-preds", action="store_true", help="store every member's extracted answer")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.n_members % 2:
        ap.error("--n-members must be even (antithetic pairs)")
    n_samples = args.n_samples or args.n_members

    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    from eggroll_noise import target_shapes, write_member_adapter
    from h1_gsm_eval import GSM8KEvaluator

    shapes = target_shapes(args.model)
    adapter_root = args.adapter_dir or tempfile.mkdtemp(prefix="pop_adapters_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    t0 = time.time()
    members = {}  # sigma -> [LoRARequest] in member order (+eps_0, -eps_0, +eps_1, ...)
    lid = 0
    for sigma in args.sigmas:
        reqs = []
        for m in range(args.n_members):
            path = write_member_adapter(os.path.join(adapter_root, f"sigma{sigma:g}", f"m{m}"), shapes,
                                        sigma, args.rank, args.seed, pair=m // 2, sign=1 if m % 2 == 0 else -1)
            lid += 1
            reqs.append(LoRARequest(f"sigma{sigma:g}_m{m}", lid, path))
        members[sigma] = reqs
    print(f"wrote {lid} adapters ({len(shapes)} target weights each) to {adapter_root} in {time.time() - t0:.0f}s", flush=True)

    ev = GSM8KEvaluator(args.model, is_instruct=True, llm_kwargs={
        "enable_lora": True, "max_lora_rank": max(8, args.rank), "max_loras": min(args.max_loras, args.n_members),
        "max_cpu_loras": lid, "gpu_memory_utilization": args.gpu_mem, "max_model_len": 1024 + args.max_new_tokens})
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    result = {"config": {**vars(args), "n_samples": n_samples, "n_target_weights": len(shapes)}, "datasets": {}}
    for ds in args.datasets:
        idx, rows = load_jsonl(ds, args.num_samples, 42)
        prompts = [ev.create_prompt(r["question"].strip()) for r in rows]
        gts = [float(r["final_answer"]) for r in rows]
        P = len(prompts)
        entry = {"indices": idx}
        print(f"== {ds}: {P} problems", flush=True)

        t = time.time()
        outs = ev.llm.generate(prompts, greedy)
        cscored = [score(ev, o.outputs[0].text, g) for o, g in zip(outs, gts)]
        centre, centre_preds = [c for c, _ in cscored], [p for _, p in cscored]
        entry["centre"] = {"correct": "".join(map(str, centre)), "acc": sum(centre) / P,
                           "mean_tokens": sum(len(o.outputs[0].token_ids) for o in outs) / P}
        if args.save_preds:
            entry["centre"]["preds"] = centre_preds
        print(f"   centre greedy acc {entry['centre']['acc']:.3f} ({time.time() - t:.0f}s)", flush=True)

        if args.sample_temperature > 0:
            t = time.time()
            sp = SamplingParams(temperature=args.sample_temperature, n=n_samples, seed=args.seed,
                                max_tokens=args.max_new_tokens)
            outs = ev.llm.generate(prompts, sp)
            sc = [[score(ev, o.outputs[s].text, g)[0] for o, g in zip(outs, gts)] for s in range(n_samples)]
            entry["sampled"] = {"temperature": args.sample_temperature,
                                "correct": ["".join(map(str, r)) for r in sc],
                                "summary": summarize(sc, centre, antithetic=False, group_k=args.group_k)}
            s = entry["sampled"]["summary"]
            print(f"   sampled T={args.sample_temperature}: mean {s['member_acc_mean']:.3f} "
                  f"pass@{n_samples} {s['coverage'][str(n_samples)]:.3f} ({time.time() - t:.0f}s)", flush=True)

        entry["sigma"] = {}
        for sigma, reqs in members.items():
            t = time.time()
            flat_prompts = [p for _ in reqs for p in prompts]
            flat_lora = [r for r in reqs for _ in prompts]
            outs = ev.llm.generate(flat_prompts, greedy, lora_request=flat_lora)
            scored = [score(ev, o.outputs[0].text, gts[i % P]) for i, o in enumerate(outs)]
            corr = [[scored[m * P + j][0] for j in range(P)] for m in range(len(reqs))]
            preds = [[scored[m * P + j][1] for j in range(P)] for m in range(len(reqs))]
            blk = {"correct": ["".join(map(str, r)) for r in corr],
                   "mean_tokens": sum(len(o.outputs[0].token_ids) for o in outs) / len(outs),
                   "summary": summarize(corr, centre, antithetic=True, group_k=args.group_k,
                                        preds=preds, centre_preds=centre_preds)}
            if args.save_preds:
                blk["preds"] = preds
            entry["sigma"][f"{sigma:g}"] = blk
            s = blk["summary"]
            print(f"   sigma={sigma:g}: members mean {s['member_acc_mean']:.3f} (min {s['member_acc_min']:.3f}) "
                  f"cov@{args.n_members} {s['coverage'][str(args.n_members)]:.3f} rescued {s['rescued']:.3f} "
                  f"broken {s['broken']:.3f} pair_disagree {s['pair_disagree']:.3f} ({time.time() - t:.0f}s)", flush=True)

        result["datasets"][ds] = entry
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:  # rewrite after every dataset so a timeout keeps what finished
            json.dump({"label": args.label or os.path.basename(args.model.rstrip("/")), "model": args.model, **result}, f)

    if args.adapter_dir is None:
        shutil.rmtree(adapter_root, ignore_errors=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
