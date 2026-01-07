import argparse
import asyncio
import copy
import json
from pathlib import Path
from typing import List, Tuple, Dict

from tqdm.asyncio import tqdm_asyncio

from sampler import init_model, get_model
from async_main_question import PLANNER_INSTRUCTION, J1_INSTRUCTION, HospSumm_INSTRUCTION
from score import DataScorer


# Approximate pricing per 1K tokens (keep consistent with PMC)
PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-5-mini": {
        "input": 0.00025,
        "output": 0.002,
    },
    "gemini-3-flash-preview": {
        "input": 0.0005,
        "output": 0.003,
    },
    "deepseek-v3.2": {
        "input": 0.000284,
        "output": 0.000426,
    },
    "qwen3-30b-a3b-instruct-2507": {
        "input": 0.0001065,
        "output": 0.000426,
    },
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="CoT baseline")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name")
    parser.add_argument("--model", type=str, required=True,
                        help="Generation model name (must exist in sampler.model_init_map)")
    parser.add_argument("--save_dir", type=str, default="./cot_results/",
                        help="Directory to save responses and logs")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to evaluate (default: all)")
    parser.add_argument("--given_examples", type=int, nargs="*",
                        help="Optional subset of example indices to run")
    parser.add_argument("--max_concurrent", type=int, default=50,
                        help="Maximum number of concurrent CoT calls")

    args = parser.parse_args()
    return args


def load_examples(dataset: str) -> List[dict]:
    """Load benchmark examples following async_main_question.py paths."""
    if dataset == "travelplanner":
        path = Path("/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/travelplanner_test_16.jsonl")
    elif dataset == "HospSumm" or dataset == "hospsumm" or dataset == "hosp_summ":
        path = Path("/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/hosp_summ_test_16.jsonl")
    elif dataset == "j1eval":
        path = Path("/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/j1eval_test_16.jsonl")
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def build_question_and_answer(dataset: str, example: dict) -> Tuple[str, str]:
    """Build the single CoT question prompt and ground-truth answer."""
    if "HospSumm" in dataset or "hospsumm" in dataset or "hosp_summ" in dataset:
        patient_info = example["instruct"]

        base_prompt = HospSumm_INSTRUCTION.format(
            patient_info=patient_info
        )
        # Explicit CoT instruction + final "Answer:" line for easier extraction
        cot_suffix = (
            # "\n\nPlease think step by step and explain your reasoning. "
            "At the end, provide your final answer in a new line starting with "
            "'Answer:' followed by a summary."
        )
        question = base_prompt + cot_suffix
        answer = example["answer"]
    elif "travelplanner" in dataset:
        query = example["query"]
        ref_info = example["reference_information"]
        base_prompt = PLANNER_INSTRUCTION.format(text=ref_info, query=query)
        cot_suffix = (
            # "\n\nPlease think step by step when constructing the travel plan. "
            "Ensure all details come strictly from the given information."
        )
        question = base_prompt + cot_suffix
        # TravelPlanner uses metric-based scoring, no direct string answer here
        answer = ""  # placeholder, scoring reads ground truth by instance_id
    elif "j1eval" in dataset:
        drop_keys = {"id", "court_information"}
        obj = copy.deepcopy(example)
        for k in drop_keys:
            obj.pop(k, None)
        text = json.dumps(obj, ensure_ascii=False)
        base_prompt = J1_INSTRUCTION.format(text=text)
        cot_suffix = (
            # "\n\n请逐步推理案件事实和法律适用过程，"
            "最后在新的一行以 'Answer:' 开头给出简洁的最终裁判主文。"
        )
        question = base_prompt + cot_suffix
        answer = example["court_information"]["ground_truth"]["court_judgment"]
    else:
        raise NotImplementedError

    return question, answer


async def cot_single_example(
    example_id: int,
    example: dict,
    dataset: str,
    model_name: str,
    scorer: DataScorer,
    save_root: Path,
):
    """Run CoT for a single example and evaluate. Also return approximate generation cost."""
    sampler = get_model(model_name)

    question, answer = build_question_and_answer(dataset, example)
    messages = [dict(role="user", content=question)]

    # Run single CoT generation
    response_text, usage = await sampler(messages, response_format="normal")

    # Approximate cost from usage
    def _extract_usage_value(u, key: str) -> int:
        if u is None:
            return 0
        if hasattr(u, key):
            return getattr(u, key) or 0
        if isinstance(u, dict):
            return u.get(key, 0) or 0
        return 0

    prompt_tokens = _extract_usage_value(usage, "prompt_tokens")
    completion_tokens = _extract_usage_value(usage, "completion_tokens")
    total_tokens = prompt_tokens + completion_tokens

    # Normalize model key for pricing lookup (substring match)
    def _resolve_model_key(name: str) -> str:
        for k in PRICING.keys():
            if k in name or name in k:
                return k
        return ""

    model_key = _resolve_model_key(model_name)
    call_cost = 0.0
    if model_key in PRICING:
        pricing = PRICING[model_key]
        call_cost = (prompt_tokens / 1000.0) * pricing["input"] + (completion_tokens / 1000.0) * pricing["output"]

    # Paths for logging (reuse DataScorer interface)
    # Structure: save_root/{dataset}/{example_id}/...
    example_dir = save_root / dataset / str(example_id)
    example_dir.mkdir(parents=True, exist_ok=True)

    judge_path = example_dir / f"{model_name}_cot_judge.txt"
    response_path = example_dir / f"{model_name}_cot_response.json"

    # DataScorer.score expects some bookkeeping fields
    prompt_message = messages
    sub_tasks_text = None
    use_oracle_verifier = True  # use dataset oracle evaluators
    response_dict: list = []
    instance_id = example_id
    code_snippet = None

    score_oracle, score_model, _ = await scorer.score(
        example_id=example_id,
        n=0,
        prompt_message=prompt_message,
        question=question,
        response_text=response_text,
        answer=answer,
        sub_tasks_text=sub_tasks_text,
        use_oracle_verifier=use_oracle_verifier,
        judge_path=str(judge_path),
        response_path=str(response_path),
        response_dict=response_dict,
        instance_id=instance_id,
        code_snippet=code_snippet,
    )

    # Oracle score is what we care about
    score = score_oracle if score_oracle is not None else score_model

    # Determine correctness flag similar to main_judge / score.py
    if "j1eval" in dataset:
        # J1Eval returns a float in [0,1]; treat >=0.5 as correct
        is_correct = score >= 0.5
    else:
        # travelplanner / HospSumm are binary (0/1) scores
        is_correct = score >= 0.5

    # Note: returned cost only accounts for main generation call, not scoring overhead
    return score, is_correct, call_cost, total_tokens


async def main(args):
    dataset = args.dataset
    model_name = args.model
    save_root = Path(args.save_dir)

    # Init generation model
    init_model(model_name)

    # Init scorer (uses oracle evaluators under the hood)
    scorer = DataScorer(dataset=dataset, technique="cot", mode_verifier=model_name)

    examples = load_examples(dataset)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    # Optional subset selection
    selected_indices = set(args.given_examples) if args.given_examples else None

    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def run_with_semaphore(eid: int, ex: dict):
        async with semaphore:
            return await cot_single_example(
                example_id=eid,
                example=ex,
                dataset=dataset,
                model_name=model_name,
                scorer=scorer,
                save_root=save_root,
            )

    tasks = []
    for example_id, example in enumerate(examples):
        if selected_indices is not None and example_id not in selected_indices:
            continue
        tasks.append(run_with_semaphore(example_id, example))

    scores = []
    total_cost = 0.0
    total_tokens = 0
    if tasks:
        for score, is_correct, call_cost, call_tokens in await tqdm_asyncio.gather(*tasks):
            scores.append(score)
            total_cost += call_cost
            total_tokens += call_tokens

    # Summary:统一按平均 score 计算
    if scores:
        # travelplanner / HospSumm 的 score 为 0/1，平均值即为准确率
        # j1eval 的 score 为 [0,1] 浮点数，平均值即为 JUD 平均分
        avg_score = round(sum(scores) / len(scores), 4)
    else:
        avg_score = 0.0


    print("=" * 80)
    print(f"CoT Baseline Results - Dataset: {dataset}, Model: {model_name}")
    print(f"Number of evaluated examples: {len(scores)}")
    print(f"Average score: {avg_score:.4f}")
    print(f"Total generation tokens (approx): {total_tokens}")
    print(f"Total generation cost (approx): ${total_cost:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    cli_args = parse_arguments()
    asyncio.run(main(cli_args))


