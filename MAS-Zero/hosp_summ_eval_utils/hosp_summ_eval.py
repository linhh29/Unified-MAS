"""
Hospital Summarization evaluation utilities
Evaluates discharge summary answers using LLM-as-judge
"""
import json
import os
import asyncio
import re
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

# Dataset paths
DATASET_PATHS = {
    "test": Path("/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/hosp_summ_test_16.jsonl"),
    "validate": Path("/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/hosp_summ_validate.jsonl"),
}

# LLM-as-judge configuration
DEFAULT_JUDGE_MODEL = "gpt-4o"
MAX_CONCURRENT_REQUESTS = 10

# Judge prompts (from HospSumm.py)
JUDGE_SYSTEM_PROMPT = """You are an expert Board-Certified Physician and a specialized Medical Scribe evaluator. Your task is to evaluate the quality of a generated "Hospitalization Summary" by comparing it against a ground-truth "Reference Summary" written by a human doctor."""

JUDGE_PROMPT_TEMPLATE = """The goal of the summarization task is to: "Summarize the key diagnostic information and significant results based on the patient's multiple health records during hospitalization."

Please evaluate the [Generated Summary] based on the [Reference Summary] across the following four dimensions. For each dimension, assign a score from 0 to 10.

### Evaluation Dimensions:

**1. Factual Accuracy (0-10)**
*   **Definition:** Does the generated summary contain any factual errors, hallucinations, or contradictions compared to the reference?
*   **Scoring:**
    *   10: Absolutely no factual errors. All claims are supported by the reference.
    *   0: Contains dangerous medical errors (e.g., stating a patient has a disease they do not have, or flipping a negative result to positive).

**2. Completeness of Diagnostics (0-10)**
*   **Definition:** Does the generated summary capture all the "Key Diagnostic Information" and "Significant Results" found in the reference? Look for specific diseases, procedures, and critical lab values.
*   **Scoring:**
    *   10: Captures 100% of the key diagnoses and significant findings mentioned in the reference.
    *   0: Misses the primary reason for hospitalization or the main diagnosis.

**3. Coherence & Medical Professionalism (0-10)**
*   **Definition:** Is the summary well-structured, logically organized, and written in professional medical language?
*   **Scoring:**
    *   10: Reads like a professional discharge summary. Perfect terminology and flow.
    *   0: Incoherent, disjointed, or uses unprofessional/layman language inappropriately.

**4. Conciseness (0-10)**
*   **Definition:** Does the summary avoid redundancy and irrelevant details?
*   **Scoring:**
    *   10: Very concise. Every sentence adds value to the clinical picture.
    *   0: Extremely repetitive or filled with irrelevant nursing care details (e.g., "patient slept well") that are not diagnostic.

Reference Summary:
{reference}

Generated Summary:
{hypothesis}

Please output your evaluation in the following JSON format only, without additional commentary outside the JSON:

{{
"reasoning": "A brief explanation of the evaluation (max 3 sentences), highlighting specific missing facts or errors.",
"scores": {{
    "factual_accuracy": <float>,
    "completeness": <float>,
    "coherence": <float>,
    "conciseness": <float>
}}
}}
"""

# Cache for loaded datasets
_dataset_cache = {}

# Global async client and semaphore
_async_client: Optional[AsyncOpenAI] = None
_semaphore: Optional[asyncio.Semaphore] = None


# ---------------------------------------------------------------------------
# LLM-as-judge helpers
# ---------------------------------------------------------------------------

def _ensure_clients(api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
    """Initialize async OpenAI client and semaphore if not already initialized."""
    global _async_client, _semaphore
    if _async_client is None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        base = base_url or os.getenv("OPENAI_API_BASE")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY is required for LLM-as-judge evaluation.")
        _async_client = AsyncOpenAI(api_key=key, base_url=base)
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


async def _llm_judge_call(hypothesis: str, reference: str, model: str) -> str:
    """Call LLM judge to evaluate the summary."""
    if _async_client is None or _semaphore is None:
        _ensure_clients()
    
    assert _async_client is not None
    assert _semaphore is not None
    
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(reference=reference, hypothesis=hypothesis)
    
    async with _semaphore:
        if model == 'gpt-4o':
            response = await _async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=1,
                response_format={"type": "json_object"}
            )
        else:
            # Fallback for other models
            response = await _async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=1,
            )
    
    return response.choices[0].message.content.strip()


def _parse_judge_score(judge_output: str) -> float:
    """
    Parse JSON response from LLM judge and calculate weighted average score.
    
    Args:
        judge_output: JSON string from LLM judge
    
    Returns:
        float: Weighted average score normalized to 0-1 range (scores are 0-10, so divide by 10)
    """
    if not judge_output:
        return 0.0
    
    try:
        # Try to parse JSON response
        response_json = json.loads(judge_output)
        scores = response_json.get("scores", {})
        
        factual_accuracy = scores.get("factual_accuracy", 0.0)
        completeness = scores.get("completeness", 0.0)
        coherence = scores.get("coherence", 0.0)
        conciseness = scores.get("conciseness", 0.0)
        
        # Calculate weighted average (equal weights for all dimensions)
        weighted_average = (factual_accuracy * 0.25 + completeness * 0.25 + 
                           coherence * 0.25 + conciseness * 0.25)
        
        # Normalize from 0-10 scale to 0-1 scale
        normalized_score = weighted_average / 10.0
        
        return max(0.0, min(1.0, normalized_score))
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract numbers from the response
        print(f"Warning: Could not parse JSON from judge response: {judge_output}")
        # Try to extract any number as fallback
        numbers = re.findall(r'\d+\.?\d*', judge_output)
        if numbers:
            try:
                score = float(numbers[0])
                # Assume it's on 0-10 scale, normalize to 0-1
                return max(0.0, min(1.0, score / 10.0))
            except ValueError:
                pass
        return 0.0
    except Exception as e:
        print(f"Error parsing judge score: {e}")
        return 0.0


def load_dataset(set_type: str = "test") -> list:
    """Load dataset from JSONL file"""
    global _dataset_cache
    
    if set_type in _dataset_cache:
        return _dataset_cache[set_type]
    
    dataset_path = DATASET_PATHS.get(set_type)
    if dataset_path is None:
        raise ValueError(f"Unknown set_type: {set_type}. Must be 'test' or 'validate'")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    scenarios = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    
    _dataset_cache[set_type] = scenarios
    return scenarios


def get_reference_summary(instance_id: int, set_type: str = "test") -> str:
    """Get reference summary for a given instance_id"""
    scenarios = load_dataset(set_type)
    
    if instance_id < 0 or instance_id >= len(scenarios):
        raise ValueError(f"instance_id {instance_id} is out of range. Dataset has {len(scenarios)} examples.")
    
    scenario = scenarios[instance_id]
    answer = scenario.get("answer", "")
    
    # Handle both dict and str types
    if isinstance(answer, dict):
        return str(answer)
    else:
        return str(answer)


async def eval_score_async(
    extracted_answer: str,
    instance_id: int,
    set_type: str = "test",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    judge_model: Optional[str] = None
) -> float:
    """
    Evaluate a summary answer against the reference summary using LLM-as-judge (async version)
    
    Args:
        extracted_answer: The extracted summary answer to evaluate
        instance_id: The ID/index of the example in the dataset
        set_type: "test" or "validate" (default: "test")
        api_key: Optional OpenAI API key override (defaults to OPENAI_API_KEY env var)
        base_url: Optional OpenAI API base URL override (defaults to OPENAI_API_BASE env var)
        judge_model: Optional judge model name override (defaults to "gpt-4o")
    
    Returns:
        float: LLM-as-judge score between 0.0 and 1.0
    """
    # Initialize clients
    try:
        _ensure_clients(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"Error initializing LLM judge client for instance_id {instance_id}: {e}")
        return 0.0
    
    # Get reference summary
    try:
        reference_summary = get_reference_summary(instance_id, set_type)
    except Exception as e:
        print(f"Error loading reference summary for instance_id {instance_id}: {e}")
        return 0.0
    
    # Validate inputs
    hypothesis = extracted_answer.strip()
    reference = reference_summary.strip()
    if not hypothesis or not reference:
        print(f"Warning: Empty hypothesis or reference for instance_id {instance_id}")
        return 0.0
    
    # Call LLM judge
    try:
        model = judge_model or DEFAULT_JUDGE_MODEL
        judge_response = await _llm_judge_call(hypothesis, reference, model)
        score = _parse_judge_score(judge_response)
        return score
    except Exception as e:
        print(f"Error in LLM judge evaluation for instance_id {instance_id}: {e}")
        return 0.0


def eval_score_sync(
    extracted_answer: str,
    instance_id: int,
    set_type: str = "test",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    judge_model: Optional[str] = None
) -> float:
    """
    Evaluate a summary answer against the reference summary using LLM-as-judge (synchronous version)
    
    Args:
        extracted_answer: The extracted summary answer to evaluate
        instance_id: The ID/index of the example in the dataset
        set_type: "test" or "validate" (default: "test")
        api_key: Optional OpenAI API key override (defaults to OPENAI_API_KEY env var)
        base_url: Optional OpenAI API base URL override (defaults to OPENAI_API_BASE env var)
        judge_model: Optional judge model name override (defaults to "gpt-4o")
    
    Returns:
        float: LLM-as-judge score between 0.0 and 1.0
    """
    # Run async version in sync context
    try:
        return asyncio.run(eval_score_async(
            extracted_answer, instance_id, set_type, api_key, base_url, judge_model
        ))
    except Exception as e:
        print(f"Error in synchronous LLM judge evaluation for instance_id {instance_id}: {e}")
        return 0.0

