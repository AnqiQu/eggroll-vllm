#!/usr/bin/env python3
# Vendored from LongHorizonReasoning/h1 (h1/gsm_eval.py) for the EGGROLL
# GSM-LongHorizon replication, so a clone of this repo is self-contained for
# between-stage checkpoint selection. The ONLY change from upstream is passing
# enable_thinking=False in create_prompt() so evaluation matches the
# (non-thinking) training format on Qwen3 models (no-op for Qwen2.5).

import re
import json
import torch
import random
import argparse
import warnings
from typing import List, Dict, Tuple, Optional
from time import sleep
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import gc
from datetime import datetime
import os
import shutil

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#                                 CONSTANTS                                   #
# --------------------------------------------------------------------------- #

XML_SYSTEM_PROMPT = """
Respond in the following format, with only the numerical answer between the <answer> tags:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
""".strip()

XML_SYSTEM_PROMPT_BOXED = """
Always respond in the following format, with only the final answer between the <answer> tags and always put your answer in boxed:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
""".strip()

# --------------------------------------------------------------------------- #
#                               EVALUATOR                                     #
# --------------------------------------------------------------------------- #


class GSM8KEvaluator:
    """Evaluator for GSM8K using vLLM."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        is_instruct: bool = False,
        temperatures: List[float] = [0.0],
        num_generations: int = 1,
        boxed_system_prompt: bool = False,
        top_p: float = 1.0,
    ):
        print(f"Loading model with vLLM: {model_name}  (TP={tensor_parallel_size})")
        self.model_name = model_name
        self.is_instruct = is_instruct

        if model_name != "random":
            # ------ Tokenizer --------------------------------------------------- #
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # ------ vLLM runtime ------------------------------------------------- #
            self.llm = LLM(model_name, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.6)

        # ------ Prompt template --------------------------------------------- #
        self.system_prompt = XML_SYSTEM_PROMPT_BOXED if boxed_system_prompt else XML_SYSTEM_PROMPT if is_instruct else ""

        self.temperatures = temperatures
        self.num_generations = num_generations
        self.top_p = top_p
    # ------------------------------------------------------------------ #
    #                  Helper / extraction utilities                     #
    # ------------------------------------------------------------------ #

    # 1) Primary path: XML tags
    @staticmethod
    def _extract_xml_answer(text: str) -> Optional[float]:
        m = re.search(
            r"<answer>\s*([+-]?\d+(?:\.\d+)?)\s*</answer>", text, flags=re.DOTALL
        )
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                return None
        return None

    # 2) Fallbacks – keep the old heuristics around just in case
    def _legacy_extract_answer(self, text: str) -> Optional[float]:
        text = text.replace(",", "")
        answer_pattern = r"[Tt]he answer is:?\s*([+-]?\d+(?:\.\d+)?)"
        if m := re.search(answer_pattern, text):
            try:
                return float(m.group(1))
            except ValueError:
                pass

        if self.is_instruct and "####" in text:
            tail = text.split("####")[-1].strip()
            if m := re.search(r"([+-]?\d+(?:\.\d+)?)", tail):
                try:
                    return float(m.group(1))
                except ValueError:
                    pass

        # last-chunk heuristics
        parts = re.split(r"answer", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", parts[-1])
            if numbers:
                try:
                    return float(numbers[0])
                except ValueError:
                    pass

        lines = text.strip().splitlines()
        if lines:
            numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", lines[-1])
            if numbers:
                try:
                    return float(numbers[-1])
                except ValueError:
                    pass
        return None

    def extract_answer_from_text(self, text: str) -> Optional[float]:
        """Try XML first, fall back to legacy."""
        try:
            text_float = float(text)
            return text_float
        except ValueError:
            pass
        return self._extract_xml_answer(text) or self._legacy_extract_answer(text)

    @staticmethod
    def extract_answer_from_solution(solution: str) -> float:
        """Ground-truth extractor (#### 123)."""
        parts = solution.split("####")
        if len(parts) >= 2:
            s = parts[-1].strip().replace(",", "")
            if m := re.search(r"([+-]?\d+(?:\.\d+)?)", s):
                return float(m.group())
        raise ValueError(f"Could not extract answer from solution: {solution}")

    # ------------------------------------------------------------------ #
    #                       Prompt construction                          #
    # ------------------------------------------------------------------ #

    def create_prompt(self, question: str) -> str:
        """Return the string we feed to vLLM."""
        if self.is_instruct and self.model_name != "random":
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                # Qwen3: disable thinking so eval matches the (non-thinking)
                # training format. No-op for Qwen2.5. Set True to eval with thinking.
                enable_thinking=False,
            )
        # non-chat baseline
        return f"Q: {question}\nA:"

    # ------------------------------------------------------------------ #
    #                    vLLM-based generation                           #
    # ------------------------------------------------------------------ #

    def generate_answers(self, questions: List[str], max_new_tokens: int = 2048) -> Tuple[List[List[str]], float]:
        prompts = [self.create_prompt(question) for question in questions]
        answers = [[] for _ in range(len(questions))]

        num_tokens_in_responses = []

        for temperature in self.temperatures:
            params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                n=(1 if abs(temperature) < 1e-6 else self.num_generations),
                top_p=self.top_p,
            )

            if self.model_name == "random":
                answers = [[str(random.randint(0, 1000)) for i1 in range(self.num_generations)] for i2 in range(len(prompts))]
            else:
                outputs = self.llm.generate(prompts, params)

                for i in range(len(outputs)):
                    for output in outputs[i].outputs:
                        answer = output.text

                        num_tokens_in_responses.append(len(output.token_ids))

                        # Stop once we hit </answer> to avoid trailing noise.
                        if self.is_instruct and "</answer>" in answer:
                            answer = answer.split("</answer>")[0] + "</answer>"

                        # For non-instruct fall back to previous trimming logic
                        elif not self.is_instruct:
                            for stop in ["\n\n", "\nQ:", "Question:"]:
                                if stop in answer:
                                    answer = answer.split(stop)[0]
                                    break

                        answers[i].append(answer.strip())
                        if abs(temperature) < 1e-6:
                            for _ in range(self.num_generations - 1):
                                answers[i].append(answer.strip())

        avg_num_tokens_in_responses = sum(num_tokens_in_responses) / len(num_tokens_in_responses)
        return answers, avg_num_tokens_in_responses

    # ------------------------------------------------------------------ #
    #                       Evaluation logic                             #
    # ------------------------------------------------------------------ #

    def evaluate_sample(
        self, question: str, final_answer: str, gen_texts: List[str]
    ) -> Tuple[bool, float, Optional[float], str]:
        gt = float(final_answer)
        preds = [self.extract_answer_from_text(gen_text) for gen_text in gen_texts]
        correct = [(pred is not None and abs(pred - gt) < 1e-2) for pred in preds]
        return correct, gt, preds, gen_texts

    def evaluate_dataset(
        self,
        dataset,
        num_samples: int = None,
        seed: int = 42,
        start_idx: int = 0,
        max_new_tokens: int = 2048,
    ) -> Dict:
        random.seed(seed)
        torch.manual_seed(seed)

        if num_samples is None:
            num_samples = len(dataset)

        if start_idx > 0:
            indices = list(
                range(start_idx, min(start_idx + num_samples, len(dataset)))
            )
        else:
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

        answers, avg_num_tokens_in_responses = self.generate_answers([dataset[idx]["question"].strip() for idx in indices], max_new_tokens=max_new_tokens)

        results = {
            "correct": 0,
            "total": 0,
            "no_answer": 0,
            "samples": [],
            "num_samples": 0,
            "avg_num_tokens_in_responses": avg_num_tokens_in_responses,
        }

        ok_all = []

        for idx, answer in tqdm(zip(indices, answers), desc="Evaluating"):
            sample = dataset[idx]
            try:
                ok, gt, pred, gen = self.evaluate_sample(
                    sample["question"].strip(), sample["final_answer"], answer
                )

                results["num_samples"] += 1
                ok_all.append([int(o) for o in ok])

                for i in range(len(pred)):
                    results["total"] += 1
                    if pred[i] is None:
                        results["no_answer"] += 1
                    elif ok[i]:
                        results["correct"] += 1

                    results["samples"].append(
                        {
                            "index": idx,
                            "question": sample["question"],
                            "ground_truth": gt,
                            "predicted": pred[i],
                            "is_correct": ok[i],
                            "generated_text": gen[i],
                        }
                    )

            except Exception as e:
                print(f"\nError processing sample {idx}: {e}")

        if results["total"] > 0:
            results["accuracy"] = results["correct"] / results["total"]
            results["answer_rate"] = (
                (results["total"] - results["no_answer"]) / results["total"]
            )
        else:
            results["accuracy"] = results["answer_rate"] = 0.0

        results["ok_all"] = ok_all
        return results


def evaluate_model_on_datasets(model_name, datasets, tp, num_samples_by_dataset, seed, start_idx, instruct, temperatures, num_generations, max_new_tokens, boxed_system_prompt, top_p):
    evaluator = GSM8KEvaluator(
        model_name=model_name,
        tensor_parallel_size=tp,
        is_instruct=instruct,
        temperatures=temperatures,
        num_generations=num_generations,
        boxed_system_prompt=boxed_system_prompt,
        top_p=top_p,
    )
    results = {}
    for dataset_name, dataset in datasets.items():
        print("=" * 60)
        print("GSM8K Evaluation with vLLM")
        print(f"Model:           {model_name}")
        print(f"Dataset:         {dataset_name}")
        print(f"Instruct-tuned:  {instruct}")
        print(f"Tensor parallel: {tp}")
        print(f"Samples:         {num_samples_by_dataset[dataset_name]}")
        print("=" * 60)
        print(f"\nEvaluating {model_name} on {dataset_name} with {num_samples_by_dataset[dataset_name]} samples …")
        results[dataset_name] = evaluator.evaluate_dataset(dataset, num_samples=num_samples_by_dataset[dataset_name], seed=seed, start_idx=start_idx, max_new_tokens=max_new_tokens)
        print(f"Results for {model_name} on {dataset_name}:")
        print(f"Length of dataset:   {len(dataset)}")
        print(f"Total samples evaluated:   {results[dataset_name]['total']}")
        print(f"Correct:         {results[dataset_name]['correct']}")
        print(f"No answer:       {results[dataset_name]['no_answer']}")
        print(f"Accuracy:        {results[dataset_name]['accuracy']*100:.2f}%")
        print(f"Answer rate:     {results[dataset_name]['answer_rate']*100:.2f}%")
        print(f"Avg num tokens in responses: {results[dataset_name]['avg_num_tokens_in_responses']}")

    del evaluator
    gc.collect()
    torch.cuda.empty_cache()

    return results

# --------------------------------------------------------------------------- #
#                                    CLI                                      #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K evaluation with vLLM")
    parser.add_argument("--models", help="HF repo or local paths or random", nargs="+")
    parser.add_argument("--datasets", help="Paths to datasets", nargs="+")
    parser.add_argument("--out_file", default=None, help="Path to output file")
    parser.add_argument("--tp", type=int, default=1, help="#GPUs for vLLM sharding")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--temperatures", type=float, default=[0.0], nargs="+")
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--boxed_system_prompt", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--instruct",
        action="store_true",
        default=True,
        help="set if model was trained with chat / system prompt (XML format)",
    )
    args = parser.parse_args()

    if "llama" in args.models[0].lower():
        warnings.warn("For llama models better use math500_eval.py even for GSM8K")

    dataset_by_name = {}
    for dataset_name in args.datasets:
        print(f"\nLoading {dataset_name} dataset …")
        try:
            with open(dataset_name, "r") as f:
                test_set = [json.loads(line) for line in f]
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
        print(f"Total samples: {len(test_set)}")
        dataset_by_name[dataset_name] = test_set
    print(f"Loaded {len(dataset_by_name)} datasets")
    
    num_samples_by_dataset = {}
    for dataset_name in args.datasets:
        num_samples_by_dataset[dataset_name] = args.num_samples if args.num_samples is not None else len(dataset_by_name[dataset_name])
    print(f"Number of samples per dataset: {num_samples_by_dataset}")

    results = {}
    for model_name in args.models:
        for _ in range(10):
            try:
                current_results = evaluate_model_on_datasets(model_name, dataset_by_name, args.tp, num_samples_by_dataset, args.seed, args.start_idx, args.instruct, args.temperatures, args.num_generations, args.max_new_tokens, args.boxed_system_prompt, args.top_p)
                results[model_name] = current_results
                for dataset_name, dataset in current_results.items():
                    print(f"Results for {model_name} on {dataset_name}:")
                    print(f"Length of dataset:   {len(dataset_by_name[dataset_name])}")
                    print(f"Total samples evaluated:   {current_results[dataset_name]['total']}")
                    print(f"Correct:         {current_results[dataset_name]['correct']}")
                    print(f"No answer:       {current_results[dataset_name]['no_answer']}")
                    print(f"Accuracy:        {current_results[dataset_name]['accuracy']*100:.2f}%")
                    print(f"Answer rate:     {current_results[dataset_name]['answer_rate']*100:.2f}%")
                    print(f"Avg num tokens in responses: {current_results[dataset_name]['avg_num_tokens_in_responses']}")
                break
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                sleep(100)

    # ---------------------- summary ----------------------------------- #
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    for model_name in results.keys():
        for dataset_name in results[model_name].keys():
            print("=" * 60)
            print(f"Results for {model_name} on {dataset_name}:")
            print(f"Length of dataset:   {len(dataset_by_name[dataset_name])}")
            print(f"Total samples evaluated:   {results[model_name][dataset_name]['total']}")
            print(f"Correct:         {results[model_name][dataset_name]['correct']}")
            print(f"No answer:       {results[model_name][dataset_name]['no_answer']}")
            print(f"Accuracy:        {results[model_name][dataset_name]['accuracy']*100:.2f}%")
            print(f"Answer rate:     {results[model_name][dataset_name]['answer_rate']*100:.2f}%")
            print(f"Avg num tokens in responses: {results[model_name][dataset_name]['avg_num_tokens_in_responses']}")

    # ---------------------- dump JSON --------------------------------- #
    
    if args.out_file is not None:
        with open(args.out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.out_file}")
    else:
        print(f"File path not provided. Results not saved.")


if __name__ == "__main__":
    main()
