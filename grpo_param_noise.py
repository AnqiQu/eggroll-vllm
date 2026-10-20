#!/usr/bin/env python
"""
grpo_param_noise.py -- h1's GRPO trainer (h1/h1/grpo.py, unchanged config) with
optional EGGROLL-shaped parameter noise: the "smoothing without a population"
control for EGGROLL.

With --param_noise_sigma 0 (default) this IS h1's grpo.py: same arguments, same
GRPOConfig, same rewards; it only swaps in a trainer factory. With sigma > 0,
every optimizer step t:

    1. on_step_begin:          W <- W_clean + sign_t * eps_t   on the 7 projection
                               matrices of every layer (eggroll_noise.lowrank_delta:
                               rank r, per-entry std sigma)
    2. rollouts + DrGRPO loss + backward all happen at the perturbed point
       (TRL 0.21 generates once per optimizer step, at its first micro-batch)
    3. on_pre_optimizer_step:  W <- W_clean (exact copy, no bf16 add/subtract
                               drift), then the optimizer applies the gradient
                               taken at the perturbed point to the clean weights.

That is a one-sample stochastic gradient of the SMOOTHED objective
J_sigma(W) = E_eps[J(W + sigma*eps)], the same objective ES follows, estimated
with backprop and a single perturbation per step instead of a population.
Consecutive steps use +eps_k / -eps_k (antithetic) unless --no_antithetic.
Checkpoints are saved after step 3, so saved weights are always the clean ones.

Extra flags (everything else is passed to h1's grpo.py):
  --param_noise_sigma S   0 = plain h1 GRPO
  --param_noise_rank R    default 1 (EGGROLL runs use r=1)
  --param_noise_seed N    default 0
  --no_antithetic
  --correctness_only      zero the four format rewards (match the ES fitness);
                          default keeps h1's reward so sigma=0 reproduces h1
  --resumable             also save optimizer/scheduler/RNG state in every checkpoint
                          (h1 sets save_only_model=True, so its runs cannot resume)
                          and resume from the newest complete checkpoint in
                          --output_dir, for runs longer than one job's walltime

Not supported: DeepSpeed ZeRO-3 / FSDP (partitioned parameters). Plain
single-GPU or DDP (accelerate launch) is fine: every rank draws the same eps
because the noise is seeded by (seed, global_step, layer).

  python grpo_param_noise.py --model Qwen/Qwen2.5-3B-Instruct \
      --dataset GSM-LongHorizon/train_len_1.jsonl --output_dir runs/grpo_noise/stage1 \
      --max_steps 300 --max_prompt_length 1024 --max_completion_length 2048 --save_steps 50 \
      --warmup_ratio 0.1 --seed 42 --param_noise_sigma 0.001
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch
from transformers import TrainerCallback

from eggroll_noise import TARGET_MODULES, is_target_weight, lowrank_delta


class ParamNoiseCallback(TrainerCallback):
    def __init__(self, sigma: float, rank: int = 1, seed: int = 0, antithetic: bool = True,
                 targets=TARGET_MODULES):
        self.sigma, self.rank, self.seed, self.antithetic, self.targets = sigma, rank, seed, antithetic, targets
        self._params = None
        self._clean = None

    def member(self, step: int) -> tuple[int, int]:
        """(pair index, sign) used at optimizer step `step`."""
        if self.antithetic:
            return step // 2, (1 if step % 2 == 0 else -1)
        return step, 1

    def _targets(self, model):
        if self._params is None:
            self._params = [(n, p) for n, p in model.named_parameters() if is_target_weight(n, p.ndim, self.targets)]
            if not self._params:
                raise RuntimeError(f"param noise: no weights matching {self.targets}")
            n_el = sum(p.numel() for _, p in self._params)
            print(f"param noise: sigma={self.sigma} rank={self.rank} antithetic={self.antithetic} "
                  f"on {len(self._params)} weights ({n_el / 1e9:.2f}B params)", flush=True)
        return self._params

    @torch.no_grad()
    def perturb(self, model, step: int) -> None:
        if self._clean is not None:
            raise RuntimeError("param noise: perturbation already applied (callback order changed?)")
        pair, sign = self.member(step)
        clean = []
        for li, (_, p) in enumerate(self._targets(model)):
            clean.append(p.detach().clone())
            d = lowrank_delta(p.shape, self.sigma, self.rank, self.seed, pair, li, device=p.device)
            p.copy_((p.float() + sign * d).to(p.dtype))
        self._clean = clean

    @torch.no_grad()
    def restore(self, model) -> None:
        if self._clean is None:
            raise RuntimeError("param noise: restore without a perturbation (callback order changed?)")
        for (_, p), c in zip(self._targets(model), self._clean):
            p.copy_(c)
        self._clean = None

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        self.perturb(model, state.global_step)

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        self.restore(model)


def last_complete_checkpoint(output_dir: str) -> str | None:
    """Newest checkpoint-N holding optimizer state and trainer_state.json (written last)."""
    steps = [int(d.rsplit("-", 1)[1]) for d in glob.glob(os.path.join(output_dir, "checkpoint-*"))
             if d.rsplit("-", 1)[1].isdigit()
             and all(os.path.isfile(os.path.join(d, f)) for f in ("optimizer.pt", "trainer_state.json"))]
    return os.path.join(output_dir, f"checkpoint-{max(steps)}") if steps else None


def _zero_reward(name: str):
    def f(completions, **kwargs):
        return [0.0] * len(completions)
    f.__name__ = f"zeroed_{name}"  # TRL logs rewards by function name
    return f


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--param_noise_sigma", type=float, default=0.0)
    ap.add_argument("--param_noise_rank", type=int, default=1)
    ap.add_argument("--param_noise_seed", type=int, default=0)
    ap.add_argument("--no_antithetic", action="store_true")
    ap.add_argument("--correctness_only", action="store_true")
    ap.add_argument("--resumable", action="store_true")
    ns, rest = ap.parse_known_args()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "h1", "h1"))
    import grpo as h1  # h1's trainer script; main() looks these names up at call time

    base_trainer = h1.GRPOTrainer

    def make_trainer(*args, **kwargs):
        trainer = base_trainer(*args, **kwargs)
        if ns.param_noise_sigma > 0:
            trainer.add_callback(ParamNoiseCallback(ns.param_noise_sigma, ns.param_noise_rank,
                                                    ns.param_noise_seed, antithetic=not ns.no_antithetic))
        if ns.resumable and (last := last_complete_checkpoint(trainer.args.output_dir)):
            print(f"grpo_param_noise: resuming from {last}", flush=True)
            train = trainer.train  # h1's main() calls trainer.train() with no arguments
            trainer.train = lambda *a, **k: train(*a, resume_from_checkpoint=last, **k)
        return trainer

    h1.GRPOTrainer = make_trainer
    if ns.resumable:
        base_config = h1.GRPOConfig

        def make_config(*args, **kwargs):
            return base_config(*args, **{**kwargs, "save_only_model": False})

        h1.GRPOConfig = make_config
    if ns.correctness_only:
        for name in ("xmlcount_reward_func", "soft_format_reward_func", "strict_format_reward_func",
                     "int_reward_func", "float_reward_func"):
            setattr(h1, name, _zero_reward(name))
    print(f"grpo_param_noise: {vars(ns)}", flush=True)
    sys.argv = [sys.argv[0]] + rest
    h1.main()


if __name__ == "__main__":
    main()
