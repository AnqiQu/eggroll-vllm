"""
eggroll_noise.py -- EGGROLL-shaped parameter noise, shared by the experiments
that need "the same kind of perturbation ES trains with" outside the ES trainer:

  * population_eval.py   evaluates the cloud of models around a checkpoint
                          (each member = one rank-r LoRA adapter served by vLLM)
  * grpo_param_noise.py   GRPO whose rollouts + gradients are taken at a
                          perturbed point theta + sigma*eps every optimizer step

The distribution is the one es_lora_multinode.py uses (get_rng_noise +
generate_local_adapters): for every targeted 2-D weight W (out x in)

    A ~ N(0, 1)^(r x in)  * sqrt(sigma)
    B ~ N(0, 1)^(out x r) * sqrt(sigma / r)
    eps = B @ A           (every entry has std sigma)

applied as W + sign * eps, sign = +1/-1 for the two members of an antithetic
pair, on the 7 projection matrices of every decoder layer, with LoRA scaling
lora_alpha / r = 1. The random stream is NOT the trainer's (different seeding),
only the distribution: nothing here needs to reproduce a specific training step.
"""
from __future__ import annotations

import json
import math
import os

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _seed(seed: int, pair: int, layer_idx: int) -> int:
    return ((int(seed) * 1_000_003 + int(pair)) * 100_003 + int(layer_idx)) % (2**63 - 1)


def lowrank_factors(shape, sigma: float, rank: int, seed: int, pair: int, layer_idx: int):
    """(A, B) in fp32 on CPU with A: (r, in), B: (out, r); B @ A has per-entry std sigma."""
    import torch
    out_dim, in_dim = int(shape[0]), int(shape[1])
    g = torch.Generator().manual_seed(_seed(seed, pair, layer_idx))
    A = torch.randn(rank, in_dim, generator=g) * math.sqrt(sigma)
    B = torch.randn(out_dim, rank, generator=g) * math.sqrt(sigma / rank)
    return A, B


def lowrank_delta(shape, sigma: float, rank: int, seed: int, pair: int, layer_idx: int, device=None):
    """Dense eps = B @ A (fp32) for one weight."""
    A, B = lowrank_factors(shape, sigma, rank, seed, pair, layer_idx)
    if device is not None:
        A, B = A.to(device), B.to(device)
    return B @ A


def is_target_weight(name: str, ndim: int, targets=TARGET_MODULES) -> bool:
    return ndim == 2 and name.endswith(".weight") and any(f".{t}." in name for t in targets)


def target_shapes(model_name: str, targets=TARGET_MODULES) -> dict[str, tuple[int, int]]:
    """{hf param name (without .weight): (out, in)} for every targeted weight, sorted
    by name so layer_idx is stable across models of the same architecture.
    Built on the meta device, so no weights are loaded."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    out = {n[: -len(".weight")]: tuple(p.shape) for n, p in m.named_parameters() if is_target_weight(n, p.ndim, targets)}
    if not out:
        raise RuntimeError(f"no target weights {targets} found in {model_name}")
    return dict(sorted(out.items()))


def write_member_adapter(path: str, shapes: dict[str, tuple[int, int]], sigma: float, rank: int,
                         seed: int, pair: int, sign: int, targets=TARGET_MODULES) -> str:
    """Write member (pair, sign) as a PEFT LoRA adapter dir vLLM can load.
    Key names follow es_lora_multinode.generate_local_adapters
    ('base_model.model.<hf name>.lora_{A,B}.weight'), lora_alpha = r."""
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    sd = {}
    for li, (name, shape) in enumerate(shapes.items()):
        A, B = lowrank_factors(shape, sigma, rank, seed, pair, li)
        sd[f"base_model.model.{name}.lora_A.weight"] = A.contiguous()
        sd[f"base_model.model.{name}.lora_B.weight"] = (B * float(sign)).contiguous()
    save_file(sd, os.path.join(path, "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": int(rank), "lora_alpha": int(rank),
           "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False, "use_rslora": False,
           "target_modules": list(targets), "base_model_name_or_path": None}
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump(cfg, f)
    return path
