"""h1 format-reward components of the completions an ES run logged, per window of steps.

The trainer logs, every step, the full completion of the first antithetic pair on a
few prompts ("------SAMPLE 1: <text> || FIT=<fitness>"). This re-scores those texts with
h1's reward functions (h1_rewards.py) and reports, per window:
  correct      h1 correctness (the training scorer)
  strict/soft  h1 strict / soft format reward fires (0.5 each)
  xml          mean xmlcount reward (max 0.5); xml_full = share with >= 0.49
  float        float reward fires (0.5)
  fmt2         share with the full format reward (>= 1.99 of 2.0)
  blocks>1     more than one <answer> block
  tail         text after the last </answer> (chars, mean)
  chars        median completion length
  =step1       the extracted answer equals the answer of sub-problem 1 (from train_len_1)
--correct-only restricts to correct completions; --why breaks down why correct ones miss
strict format.

python analyze_format_logs.py WINDOW data.jsonl log1 [log2 ...] [--correct-only] [--why]
(run from the repo root; logs are merged by step, later files win)"""
import collections
import json
import re
import statistics
import sys

from h1_rewards import (correctness_reward, extract_answer_from_text, float_reward_func,
                        soft_format_reward_func, strict_format_reward_func, xmlcount_reward_func)

flags = {a for a in sys.argv[1:] if a.startswith("--")}
args = [a for a in sys.argv[1:] if not a.startswith("--")]
win, data_path, logs = int(args[0]), args[1], args[2:]

gt_of = {json.loads(l)["question"].strip(): json.loads(l)["final_answer"] for l in open(data_path)}
step1_of = {}   # GSM8K problem text -> its answer (train_len_1 = all of GSM8K train)
for l in open("GSM-LongHorizon/train_len_1.jsonl"):
    r = json.loads(l)
    step1_of[r["question"].split("step by step:", 1)[-1].strip()] = r["final_answer"]


def step1_answer(q):
    m = re.match(r"Step 1: Solve the following math problem step by step:\s*(.*?)\n\nStep 2:", q, re.S)
    return step1_of.get(m.group(1).strip()) if m else None


samples = {}   # (step, prompt, text) -> (gt, step-1 answer, logged fitness); the log prints each batch twice
for path in logs:
    step, buf = None, open(path, errors="replace").read()
    for m in re.finditer(r"======= ES Step (\d+) / \d+ =======|\[PROMPT \d+\]: (.*?)------SAMPLE 1: (.*?) \|\| FIT=([\d.]+)", buf, re.S):
        if m.group(1):
            step = int(m.group(1)); continue
        q = m.group(2).split("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0].strip()
        if q in gt_of and step is not None:
            samples[(step, q, m.group(3))] = (gt_of[q], step1_answer(q), float(m.group(4)))
print(f"{len(samples)} logged completions, {sum(v[1] is not None for v in samples.values())} with a known step-1 answer")


def one(fn, text):
    return fn([[{"content": text}]])[0]


rows = collections.defaultdict(list)
why = collections.Counter()
unmatched = 0   # logged fitness is neither our correctness nor correctness + format: the logged text differs
for (step, q, text), (gt, s1, fit) in samples.items():
    c = correctness_reward(text, gt) > 0
    st, so, xm, fl = (one(f, text) for f in (strict_format_reward_func, soft_format_reward_func,
                                             xmlcount_reward_func, float_reward_func))
    unmatched += min(abs(fit - 2 * c), abs(fit - 2 * c - st - so - xm - fl)) > 1e-6
    if "--correct-only" in flags and not c:
        continue
    n_blk = text.count("<answer>")
    tail = len(text.split("</answer>")[-1]) if "</answer>" in text else 0
    ans = extract_answer_from_text(text)
    try:
        eq1 = s1 is not None and abs(float(ans) - float(s1)) < 1e-9
    except (TypeError, ValueError):
        eq1 = False
    rows[step // win * win].append(dict(c=c, st=st > 0, so=so > 0, xm=xm, xf=xm >= 0.49, fl=fl > 0,
                                        f2=st + so + xm + fl >= 1.99, mb=n_blk > 1, tail=tail,
                                        chars=len(text), eq1=eq1, known=s1 is not None))
    if "--why" in flags and c and st == 0:
        if n_blk == 0:
            why["no <answer> block"] += 1
        elif n_blk > 1:
            why["several <answer> blocks"] += 1
        elif not text.startswith("<reasoning>\n"):
            why["does not start with '<reasoning>\\n'"] += 1
        elif text.rstrip("\n").endswith("</answer>") and not text.endswith("</answer>\n"):
            why["ends '</answer>' without the newline"] += 1
        elif tail > 1:
            why["text after </answer>"] += 1
        else:
            why["other layout (newlines inside the blocks)"] += 1

print(f"logged fitness not reproduced from the logged text: {unmatched}")
pct = lambda rs, k: 100 * sum(r[k] for r in rs) / len(rs)
print(f"{'steps':11s} {'n':>5s} {'correct':>7s} {'strict':>6s} {'soft':>5s} {'xml':>5s} {'xml_full':>8s} {'float':>5s} {'fmt2':>5s} "
      f"{'blocks>1':>8s} {'tail':>6s} {'chars':>6s} {'=step1':>6s}")
for w in sorted(rows):
    rs = rows[w]; known = [r for r in rs if r["known"]]
    print(f"{w:4d}-{w + win - 1:<6d} {len(rs):5d} {pct(rs, 'c'):7.1f} {pct(rs, 'st'):6.1f} {pct(rs, 'so'):5.1f} "
          f"{statistics.mean(r['xm'] for r in rs):5.2f} {pct(rs, 'xf'):8.1f} {pct(rs, 'fl'):5.1f} {pct(rs, 'f2'):5.1f} "
          f"{pct(rs, 'mb'):8.1f} {statistics.mean(r['tail'] for r in rs):6.0f} {statistics.median(r['chars'] for r in rs):6.0f} "
          f"{pct(known, 'eq1') if known else float('nan'):6.1f}")
if why:
    tot = sum(why.values())
    print(f"\ncorrect completions that miss strict format: {tot}")
    for k, v in why.most_common():
        print(f"  {v:5d} ({100 * v / tot:4.1f} %)  {k}")
