"""
Async evaluator for the J1Eval JUD (judgment) metric.

The score replicates the logic from `J1Bench/src/Eval/bench/CI/CI.py`:
- Extract the judgment portion from ground truth (`court_judgment`)
  while skipping boilerplate like “如不服本判决…”.
- Ask an LLM moderator to compare the model’s judgment text with
  the ground-truth decision using the provided scoring rubric.
- Parse the moderator’s “评分：X；原因：Y” response and normalize X to 0~1.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional, Union

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH = Path("/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/j1eval_test_16.jsonl")
DEFAULT_MODERATOR_MODEL = 'gpt-4o'
MAX_CONCURRENT_REQUESTS = 10

JUDGMENT_PROMPT = """你是一名法律专家。请根据“法律问题”和“标准答案”，判断“待评测答案”是否完全、准确地涵盖了“标准答案”的所有核心要点，由此给出0-10分的评分。你**不用考虑待评测答案表达是否简洁、重点是否突出、是否使用寒暄语、结构是否冗长等非实质性因素**，无须因不够简洁而扣分，无须考虑中英文语言形式的不同，只需要考虑内容意思是否一致。

标准答案：
{gt_answer}

待测评答案：
{model_answer}

以如下格式输出你的结果（中文括号分割，不要换行，不要带括号）：
评分：；原因：
"""

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_dataset_cache: Optional[list[dict]] = None
_async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
_semaphore: Optional[asyncio.Semaphore] = None


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def _load_dataset() -> list[dict]:
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"J1Eval dataset not found at {DATASET_PATH}")

    records: list[dict] = []
    with DATASET_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    _dataset_cache = records
    return records


def _resolve_case(instance_id: Union[int, str]) -> dict:
    """Resolve a case dict using numeric index or full case id."""
    cases = _load_dataset()

    # Accept either string id ("CI-12") or numeric index (0-based or 1-based)
    if isinstance(instance_id, str):
        for case in cases:
            if case.get("id") == instance_id:
                return case
        raise ValueError(f"instance_id '{instance_id}' not found in dataset.")

    if isinstance(instance_id, int):
        if instance_id < 0:
            instance_id = abs(instance_id)
        # try 0-based first
        if instance_id < len(cases):
            return cases[instance_id]
        # fallback: treat as 1-based index derived from case id suffix
        suffix_map = {int(case["id"].split("-")[-1]): case for case in cases if case.get("id")}
        if instance_id in suffix_map:
            return suffix_map[instance_id]
        raise IndexError(f"instance_id {instance_id} out of range (dataset size={len(cases)}).")

    raise TypeError("instance_id must be int or str.")


def _build_ground_truth_judgment(case: dict) -> str:
    """Format ground-truth judgment text similar to evaluator implementation."""
    judgments = case["court_information"]["ground_truth"]["court_judgment"]
    filtered = []
    idx = 1
    for item in judgments:
        if "如不服本判决" in item:
            continue
        filtered.append(f"{idx}. {item}")
        idx += 1
    return "\n".join(filtered).strip()


# ---------------------------------------------------------------------------
# LLM moderator helpers
# ---------------------------------------------------------------------------

def _ensure_clients(api_key: Optional[str] = None) -> None:
    global _async_client, _semaphore
    if _async_client is None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY is required for J1Eval evaluation.")
        _async_client = AsyncOpenAI(api_key=key)
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


async def _moderator_call(prompt: str, model: str) -> str:
    if _async_client is None or _semaphore is None:
        _ensure_clients()

    assert _async_client is not None
    assert _semaphore is not None

    async with _semaphore:
        if model == 'gpt-4o':
            response = await _async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一名严格的法律评审助理。"},
                    {"role": "user", "content": prompt},
                ],
                # reasoning_effort='low',
                # max_completion_tokens=512,
                temperature=1,
            )
        elif model == 'gpt-5':
            response = await _async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一名严格的法律评审助理。"},
                    {"role": "user", "content": prompt},
                ],
                reasoning_effort='medium',
                max_completion_tokens=512,
                temperature=1,
            )
        # print('response: ',response)
    return response.choices[0].message.content.strip()


def _parse_score(mod_output: str) -> float:
    """
    Parse "评分：X；原因：Y" style output and normalize X to [0,1].
    Returns 0 if parsing fails.
    """
    if not mod_output:
        return 0.0
    score_part = mod_output.split("；", 1)[0]
    score_part = score_part.replace("评分：", "").replace("分", "").strip()
    try:
        score_value = float(score_part)
        return max(0.0, min(1.0, score_value / 10.0))
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def eval_score_async(extracted_answer: str, instance_id: Union[int, str], *, api_key: Optional[str] = None,
                           moderator_model: Optional[str] = None) -> float:
    """
    Evaluate the JUD metric for a single sample.

    Args:
        extracted_answer: Judge's predicted judgment text (model output).
        instance_id: Either the dataset index (int) or case id string (e.g., "CI-12").
        api_key: Optional OpenAI API key override (defaults to OPENAI_API_KEY env var).
        moderator_model: Optional override for moderator model name.

    Returns:
        float: normalized JUD score in [0, 1].
    """
    _ensure_clients(api_key=api_key)

    try:
        case = _resolve_case(instance_id)
    except Exception as exc:
        print(f"[J1Eval] Failed to resolve case for instance_id={instance_id}: {exc}")
        return 0.0

    gt_judgment = _build_ground_truth_judgment(case)
    if not gt_judgment:
        return 0.0

    prompt = JUDGMENT_PROMPT.format(gt_answer=gt_judgment, model_answer=extracted_answer.strip())

    try:
        moderator_response = await _moderator_call(prompt, moderator_model or DEFAULT_MODERATOR_MODEL)
        return _parse_score(moderator_response)
    except Exception as exc:
        print(f"[J1Eval] Moderator call failed for instance_id={instance_id}: {exc}")
        return 0.0


def eval_score_sync(
    extracted_answer: str,
    instance_id: Union[int, str],
    *,
    api_key: Optional[str] = None,
    moderator_model: Optional[str] = None,
) -> float:
    """
    Synchronous wrapper for eval_score_async.

    Handles environments with/without an already running event loop.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Run in dedicated thread with its own event loop
            import concurrent.futures

            def _runner():
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    return new_loop.run_until_complete(
                        eval_score_async(
                            extracted_answer,
                            instance_id,
                            api_key=api_key,
                            moderator_model=moderator_model,
                        )
                    )
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_runner)
                return future.result()
        else:
            return loop.run_until_complete(
                eval_score_async(
                    extracted_answer,
                    instance_id,
                    api_key=api_key,
                    moderator_model=moderator_model,
                )
            )
    except RuntimeError:
        # No event loop yet
        return asyncio.run(
            eval_score_async(
                extracted_answer,
                instance_id,
                api_key=api_key,
                moderator_model=moderator_model,
            )
        )

