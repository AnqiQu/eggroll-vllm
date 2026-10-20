#!/usr/bin/env python
"""Check from a GRPO run's own log that every prompt reached the model whole.

TRL 0.21 keeps only the LAST max_prompt_length tokens of a prompt, silently. Its
log has, per step, the cumulative token count num_tokens (prompt + completion
tokens of all num_generations samples) and completions/mean_length, so

    implied prompt length = delta num_tokens / num_generations - completions/mean_length

h1's grpo.py trains on one question per step in file order (shuffle_dataset False),
so step k used question (k-1) mod N, whose full length is computed here with h1's
prompt builder and TRL's chat-template call. Truncated steps show implied < full
(exactly max_prompt_length). Exit 1 if any step does not match.

  python check_prompt_lengths.py --state runs/.../checkpoint-10/trainer_state.json \
      --dataset GSM-LongHorizon/train_len_5.jsonl --tokenizer Qwen/Qwen2.5-3B-Instruct
"""
import argparse
import json
import os
import sys


def implied_prompt_lengths(log_history, num_generations):
    """{step: implied prompt length} for every step whose previous step is also logged."""
    rows = [h for h in log_history if "num_tokens" in h and "completions/mean_length" in h]
    out, prev_step, prev_tokens = {}, 0, 0
    for h in sorted(rows, key=lambda h: h["step"]):
        if h["step"] == prev_step + 1:
            out[h["step"]] = (h["num_tokens"] - prev_tokens) / num_generations - h["completions/mean_length"]
        prev_step, prev_tokens = h["step"], h["num_tokens"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="trainer_state.json of the last checkpoint")
    ap.add_argument("--dataset", required=True, help="the training jsonl")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--num_generations", type=int, default=16)
    ap.add_argument("--max_prompt_length", type=int, default=None, help="only for the report")
    ap.add_argument("--show", type=int, default=10, help="print the first N steps")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "h1", "h1"))
    from grpo import get_gsm8k_questions  # h1's prompt builder (system prompt + question)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    data = get_gsm8k_questions(args.dataset, None, None)
    # as TRL 0.21: apply_chat_template(add_generation_prompt=True), then tokenize without special tokens
    full = [len(tok(tok.apply_chat_template(x["prompt"], tokenize=False, add_generation_prompt=True),
                    add_special_tokens=False)["input_ids"]) for x in data]

    implied = implied_prompt_lengths(json.load(open(args.state))["log_history"], args.num_generations)
    if not implied:
        sys.exit("no consecutive logged steps with num_tokens and completions/mean_length")
    bad = []
    print(f"{'step':>5} {'question':>8} {'full':>6} {'implied':>8}")
    for i, step in enumerate(sorted(implied)):
        q = (step - 1) % len(full)
        if abs(implied[step] - full[q]) > 0.01:
            bad.append(step)
        if i < args.show or step in bad[-1:]:
            print(f"{step:5d} {q:8d} {full[q]:6d} {implied[step]:8.1f}" + ("  MISMATCH" if step in bad[-1:] else ""))
    steps = sorted(implied)
    m_imp = sum(implied[s] for s in steps) / len(steps)
    m_full = sum(full[(s - 1) % len(full)] for s in steps) / len(steps)
    over = sum(full[(s - 1) % len(full)] > (args.max_prompt_length or 10**9) for s in steps)
    print(f"{len(steps)} steps: mean implied prompt length {m_imp:.1f}, mean full length {m_full:.1f}, "
          f"longest full {max(full[(s - 1) % len(full)] for s in steps)}"
          + (f", {over} longer than max_prompt_length {args.max_prompt_length}" if args.max_prompt_length else ""))
    if bad:
        sys.exit(f"PROMPT CHECK FAILED: {len(bad)} of {len(steps)} steps saw a different prompt length "
                 f"than the full prompt (first: step {bad[0]})")
    print("PROMPT CHECK OK: every step saw its full prompt")


if __name__ == "__main__":
    main()
