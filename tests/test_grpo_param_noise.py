"""ParamNoiseCallback inside a real HF Trainer loop (tiny random Qwen2, CPU).
Requires torch + transformers; skipped otherwise.
Run:  python -m unittest tests.test_grpo_param_noise"""
import ast
import os
import unittest

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, Trainer, TrainingArguments

    import eggroll_noise as N
    from grpo_param_noise import ParamNoiseCallback, unstripped_system_prompt
except Exception:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch/transformers not installed")
class TestParamNoiseCallback(unittest.TestCase):
    def _model(self, dtype):
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
        return Qwen2ForCausalLM(cfg).to(dtype)

    def _run(self, dtype, steps=4, grad_accum=2, lr=0.0):
        model = self._model(dtype)
        cb = ParamNoiseCallback(sigma=1e-2, rank=1, seed=7, antithetic=True)
        targets = [(n, p) for n, p in model.named_parameters() if N.is_target_weight(n, p.ndim)]
        clean0 = [p.detach().clone() for _, p in targets]
        seen = []  # (global_step, delta of first target weight) at every micro-batch forward

        class Recorder(Trainer):
            def training_step(self, model, inputs, *a, **k):
                seen.append((self.state.global_step, (targets[0][1].detach().float() - clean0[0].float()).clone()))
                return super().training_step(model, inputs, *a, **k)

        data = [{"input_ids": torch.randint(0, 64, (12,)), "labels": torch.randint(0, 64, (12,))}
                for _ in range(steps * grad_accum)]
        args = TrainingArguments(output_dir="/tmp/pn_test", per_device_train_batch_size=1,
                                 gradient_accumulation_steps=grad_accum, max_steps=steps, learning_rate=lr,
                                 weight_decay=0.0, report_to=[], save_strategy="no", logging_strategy="no",
                                 use_cpu=True, disable_tqdm=True)
        Recorder(model=model, args=args, train_dataset=data, callbacks=[cb]).train()
        return model, targets, clean0, seen, cb

    def test_perturbed_during_step_and_exactly_restored_bf16(self):
        model, targets, clean0, seen, cb = self._run(torch.bfloat16)
        # lr = 0: training must leave the weights bit-identical (restore is a copy, not a subtraction)
        for (_, p), c in zip(targets, clean0):
            self.assertTrue(torch.equal(p, c))
        self.assertIsNone(cb._clean)
        # every micro-batch of a step saw that step's perturbation, and it matches eggroll_noise
        self.assertEqual(len(seen), 8)
        for step, d in seen:
            pair, sign = cb.member(step)
            want = (clean0[0].float() + sign * N.lowrank_delta(clean0[0].shape, 1e-2, 1, 7, pair, 0)).to(torch.bfloat16).float() - clean0[0].float()
            self.assertTrue(torch.equal(d, want), f"step {step}")
            self.assertGreater(float(d.abs().max()), 0.0)
        # antithetic: steps 0 and 1 use +eps_0 / -eps_0 (fp32 check avoids bf16 rounding asymmetry)
        self.assertEqual(cb.member(0), (0, 1))
        self.assertEqual(cb.member(1), (0, -1))
        self.assertEqual(cb.member(2), (1, 1))

    def test_gradient_update_lands_on_clean_weights_fp32(self):
        model, targets, clean0, seen, cb = self._run(torch.float32, steps=2, lr=1e-3)
        # weights moved (optimizer ran) but carry no leftover perturbation: the change is
        # far from any +/-eps pattern (sigma=1e-2 would dominate an lr=1e-3 Adam step)
        moved = targets[0][1].detach() - clean0[0]
        self.assertGreater(float(moved.abs().max()), 0.0)
        self.assertLess(float(moved.abs().max()), 5e-3)


H1_GRPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "h1", "h1", "grpo.py")


@unittest.skipIf(torch is None or not os.path.exists(H1_GRPO), "needs torch and the h1 clone (h1/h1/grpo.py)")
class TestUnstrippedSystemPrompt(unittest.TestCase):
    def test_matches_h1_literal_before_strip(self):
        # h1/h1/grpo.py: SYSTEM_PROMPT = """...""".strip(); read the literal itself from the source
        tree = ast.parse(open(H1_GRPO).read())
        call = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", None) == "SYSTEM_PROMPT")
        literal = call.func.value.value
        self.assertEqual(unstripped_system_prompt(literal.strip()), literal)
        self.assertTrue(literal.endswith("</answer>\n"))


if __name__ == "__main__":
    unittest.main()
