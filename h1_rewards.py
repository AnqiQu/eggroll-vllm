"""
h1 reward + answer-extraction functions, vendored for EGGROLL training.

The block marked "VERBATIM FROM h1/grpo.py" below is copied unchanged from the
h1 repo (``h1/h1/grpo.py``) so that the EGGROLL fitness reproduces h1's DrGRPO
total reward exactly (same system prompt, same format rewards, same answer
extraction). Do not edit those functions -- if h1 updates them, re-copy.

The small "EGGROLL ADAPTERS" section at the bottom is the only new code: it
lets the per-sample EGGROLL fitness call the verbatim h1 reward functions (which
expect TRL's ``completions`` batch format) on a single response string, and it
wires the correctness check through the vendored ``h1_math_utils.grade_answer``
(the answer comparator requested for training/eval parity).

See ``GSMLongHorizonTask`` in ``tasks.py`` for how these are combined into a
scalar fitness per rollout.
"""

import re

# ======================================================================
# ===================  VERBATIM FROM h1/grpo.py  =======================
# (copied unchanged so training reward == h1 reward; do not modify)
# ======================================================================

SYSTEM_PROMPT = """
Respond in the following format, with only the numerical answer between the <answer> tags:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
""".strip()

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer> 
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def legacy_extract_answer(text: str) -> str | None:
    text = text.replace(",", "")
    answer_pattern = r"[Tt]he answer is:?\s*([+-]?\d+(?:\.\d+)?)"
    if m := re.search(answer_pattern, text):
        try:
            m_float = float(m.group(1))
            return m.group(1)
        except ValueError:
            pass

    if "####" in text:
        tail = text.split("####")[-1].strip()
        if m := re.search(r"([+-]?\d+(?:\.\d+)?)", tail):
            try:
                m_float = float(m.group(1))
                return m.group(1)
            except ValueError:
                pass

    # last-chunk heuristics
    parts = re.split(r"answer", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", parts[-1])
        if numbers:
            try:
                numbers_float = float(numbers[0])
                return numbers[0]
            except ValueError:
                pass

    lines = text.strip().splitlines()
    if lines:
        numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", lines[-1])
        if numbers:
            try:
                numbers_float = float(numbers[-1])
                return numbers[-1]
            except ValueError:
                pass
    return None

def extract_answer_from_text(text: str) -> str:
    if '<answer>' in text:
        return extract_xml_answer(text)
    else:
        legacy_answer = legacy_extract_answer(text)
        if legacy_answer is not None:
            return legacy_answer
        else:
            return text

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def float_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer_from_text(r) for r in responses]
    results = []
    for r in extracted_responses:
        try:
            r_float = float(r)
            results.append(0.5)
        except ValueError:
            results.append(0.0)
    return results

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# The correctness reward in h1/grpo.py (reproduced here as a comment for
# reference) extracts with ``extract_answer_from_text`` and then compares floats
# with ``abs(float(r) - float(a)) < 1e-9``, returning 2.0 for a match:
#
#     def correctness_reward_func(prompts, completions, answer, **kwargs):
#         responses = [c[0]['content'] for c in completions]
#         extracted = [extract_answer_from_text(r) for r in responses]
#         return [2.0 if <float match> else 0.0 for r, a in zip(extracted, answer)]
#
# For training we keep h1's extractor but route the *comparison* through the
# vendored ``h1_math_utils.grade_answer`` (see ``correctness_reward`` below), as
# requested, so the correctness signal matches h1's answer comparator.

# ======================================================================
# ==========================  EGGROLL ADAPTERS  ========================
# (new: evaluate the verbatim h1 rewards on ONE response string and sum
#  them into the scalar-per-rollout fitness EGGROLL expects.)
# ======================================================================

# h1's total DrGRPO reward is the SUM of these functions. The numeric-format
# reward is int_reward_func for horizon 1 (integer answers) and
# float_reward_func for horizons 2+ (float answers) -- selected by h1's
# ``--float_reward_func`` flag, which we mirror with ``float_mode`` here.
CORRECT_REWARD = 2.0  # h1 correctness_reward_func returns 2.0 on a correct answer


def _as_completions(response: str):
    """Wrap a single response string in TRL's completions format so the verbatim
    h1 reward functions above can be called unchanged."""
    return [[{"content": response}]]


def format_reward(response: str, float_mode: bool) -> float:
    """Sum of h1's format-related rewards for a single response.

    Mirrors h1's reward_funcs list minus the correctness term:
      xmlcount + soft_format + strict_format + (float_reward | int_reward).
    """
    comps = _as_completions(response)
    numeric_reward_func = float_reward_func if float_mode else int_reward_func
    return (
        xmlcount_reward_func(comps)[0]
        + soft_format_reward_func(comps)[0]
        + strict_format_reward_func(comps)[0]
        + numeric_reward_func(comps)[0]
    )


def correctness_reward(response: str, answer) -> float:
    """h1's correctness term (2.0 / 0.0) using h1's extractor + the vendored
    ``h1_math_utils.grade_answer`` comparator."""
    # Lazy import so the format-only path does not require sympy/pylatexenc.
    from h1_math_utils import grade_answer
    extracted = extract_answer_from_text(response)
    return CORRECT_REWARD if grade_answer(extracted, str(answer)) else 0.0


def total_reward(response: str, answer, float_mode: bool) -> float:
    """h1's full DrGRPO total reward for a single (response, answer): correctness
    + all format rewards. This scalar is the EGGROLL fitness for the rollout."""
    return correctness_reward(response, answer) + format_reward(response, float_mode)
