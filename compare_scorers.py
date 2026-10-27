"""Score the training-log samples of an ES run with both scorers and compare.
Training scorer: h1_rewards.correctness_reward on the full text (last <answer> block).
Eval scorer: h1_gsm_eval: text cut at the first </answer>, numeric-only XML regex, legacy
fallbacks, |pred - gt| < 1e-2.
python compare_scorers.py WINDOW data.jsonl log1 [log2 ...]   (run from the repo root)"""
import json, re, sys, collections
sys.path.insert(0, ".")
from h1_rewards import correctness_reward, extract_answer_from_text as train_extract
import h1_gsm_eval

win, data_path, logs = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
gt_of = {json.loads(l)["question"].strip(): json.loads(l)["final_answer"] for l in open(data_path)}
ev = h1_gsm_eval.GSM8KEvaluator.__new__(h1_gsm_eval.GSM8KEvaluator); ev.is_instruct = True  # extraction methods only, no model (eval runs with --instruct)


def eval_verdict(text, gt):
    if "</answer>" in text:
        text = text.split("</answer>")[0] + "</answer>"
    pred = ev.extract_answer_from_text(text.strip())
    return pred is not None and abs(pred - float(gt)) < 1e-2, pred


samples = []   # (step, text, gt, fit)
for path in logs:
    step, buf = None, open(path, errors="replace").read()
    for m in re.finditer(r"======= ES Step (\d+) / \d+ =======|\[PROMPT \d+\]: (.*?)------SAMPLE 1: (.*?) \|\| FIT=([\d.]+)", buf, re.S):
        if m.group(1):
            step = int(m.group(1)); continue
        q = m.group(2).split("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0].strip()
        if q in gt_of and step is not None:
            samples.append((step, m.group(3), gt_of[q], float(m.group(4))))
samples = list(dict.fromkeys(samples))   # the log prints each logged batch twice
print(f"{len(samples)} logged samples")

# format over training: how many <answer> blocks, and is there a closing tag at all
blk = collections.defaultdict(collections.Counter)
for step, text, gt, fit in samples:
    n_ans = text.count("<answer>")
    c = blk[step // win * win]
    c["n"] += 1; c["multi"] += n_ans > 1; c["none"] += n_ans == 0; c["unclosed"] += n_ans > 0 and "</answer>" not in text.split("<answer>")[-1]
    if fit > 0:
        c["correct"] += 1; c["correct_multi"] += n_ans > 1
print("steps        n   >1 block%  no block%  last block unclosed%  | correct: n  >1 block%")
for w in sorted(blk):
    c = blk[w]
    print(f"{w:4d}-{w + win - 1:<5d} {c['n']:5d} {100 * c['multi'] / c['n']:9.1f} {100 * c['none'] / c['n']:9.1f} {100 * c['unclosed'] / c['n']:12.1f}"
          f"          | {c['correct']:5d} {100 * c['correct_multi'] / max(c['correct'], 1):7.1f}")

tab = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(list)
for step, text, gt, fit in samples:
    tr = fit > 0
    assert tr == (correctness_reward(text, gt) > 0), "log FIT disagrees with the training scorer"
    ok, pred = eval_verdict(text, gt)
    w = step // win * win
    tab[w][(tr, ok)] += 1
    n_ans = text.count("<answer>")
    key = ("train only" if tr and not ok else "eval only" if ok and not tr else None)
    if key:
        tab[w][key + f" ({'1 block' if n_ans == 1 else f'{n_ans} blocks'})"] += 1
        examples[key].append((step, gt, train_extract(text), pred, n_ans, text))

print("steps       n    train%  eval%   both  train-only  eval-only")
for w in sorted(tab):
    c = tab[w]; n = sum(v for k, v in c.items() if isinstance(k, tuple))
    tr = c[(True, True)] + c[(True, False)]; e = c[(True, True)] + c[(False, True)]
    print(f"{w:4d}-{w + win - 1:<5d} {n:5d} {100 * tr / n:6.1f} {100 * e / n:6.1f} {c[(True, True)]:6d} {c[(True, False)]:9d} {c[(False, True)]:9d}")
reasons = collections.Counter()
for c in tab.values():
    for k, v in c.items():
        if isinstance(k, str):
            reasons[k] += v
print("disagreements by number of <answer> blocks:", dict(reasons))
for key, ex in examples.items():
    print(f"\n===== {key}: {len(ex)} =====")
    for step, gt, tx, pred, n_ans, text in ex[:6]:
        tail = text[text.find("<answer>") - 150 if "<answer>" in text else -300:][:500]
        print(f"--- step {step} gt={gt} train_extract={tx!r} eval_pred={pred} blocks={n_ans}\n{tail}")
