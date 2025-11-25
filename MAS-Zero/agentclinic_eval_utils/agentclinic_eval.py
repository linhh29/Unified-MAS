"""
AgentClinic evaluation utilities
Evaluates medical diagnosis answers using LLM-based comparison
"""
import json
import re
import asyncio
import os
from pathlib import Path
from typing import Optional, Dict, Any
from openai import AsyncOpenAI
import anthropic


# Global async clients
_async_openai_client = None
_anthropic_client = None
_concurrency_semaphore = None

# Default moderator LLM for evaluation
DEFAULT_MODERATOR_LLM = "gpt4o"

# Dataset paths
DATASET_PATHS = {
    "test": Path("/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/agentclinic_medqa_test.jsonl"),
    "validate": Path("/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/agentclinic_medqa_validate.jsonl"),
}

# Cache for loaded datasets
_dataset_cache = {}


def init_clients(api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None, max_concurrent: int = 10):
    """Initialize async clients and concurrency control"""
    global _async_openai_client, _anthropic_client, _concurrency_semaphore
    if api_key:
        _async_openai_client = AsyncOpenAI(api_key=api_key)
    elif os.getenv("OPENAI_API_KEY"):
        _async_openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if anthropic_api_key:
        _anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
    elif os.getenv("ANTHROPIC_API_KEY"):
        _anthropic_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    if _concurrency_semaphore is None:
        _concurrency_semaphore = asyncio.Semaphore(max_concurrent)


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


def get_correct_diagnosis(instance_id: int, set_type: str = "test") -> str:
    """Get correct diagnosis for a given instance_id"""
    scenarios = load_dataset(set_type)
    
    if instance_id < 0 or instance_id >= len(scenarios):
        raise ValueError(f"instance_id {instance_id} is out of range. Dataset has {len(scenarios)} examples.")
    
    scenario = scenarios[instance_id]
    diagnosis = scenario["OSCE_Examination"]["Correct_Diagnosis"]
    
    # Handle both dict and str types
    if isinstance(diagnosis, dict):
        return str(diagnosis)
    else:
        return str(diagnosis)


async def query_model_async(
    model_str: str,
    prompt: str,
    system_prompt: str,
    tries: int = 30,
    timeout: float = 20.0
) -> str:
    """Async query model with concurrency control"""
    global _async_openai_client, _anthropic_client, _concurrency_semaphore
    
    # Initialize clients if not already initialized
    if _async_openai_client is None and _anthropic_client is None:
        init_clients()
    
    # Use semaphore to control concurrency
    if _concurrency_semaphore:
        async with _concurrency_semaphore:
            return await _query_model_impl(model_str, prompt, system_prompt, tries, timeout)
    else:
        return await _query_model_impl(model_str, prompt, system_prompt, tries, timeout)


async def _query_model_impl(
    model_str: str,
    prompt: str,
    system_prompt: str,
    tries: int = 30,
    timeout: float = 20.0
) -> str:
    """Internal implementation of async query_model"""
    for _ in range(tries):
        try:
            actual_model_name = model_str
            
            if model_str in ["gpt4", "gpt3.5", "gpt4o", "gpt-4o-mini", "gpt4v", "o1-preview"]:
                if not _async_openai_client:
                    raise Exception("OpenAI client not initialized. Set OPENAI_API_KEY environment variable.")
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                model_map = {
                    "gpt4": "gpt-4-turbo-preview",
                    "gpt3.5": "gpt-3.5-turbo",
                    "gpt4o": "gpt-4o",
                    "gpt-4o-mini": "gpt-4o-mini",
                    "gpt4v": "gpt-4-vision-preview",
                    "o1-preview": "o1-preview-2024-09-12"
                }
                actual_model_name = model_map.get(model_str, model_str)
                
                if model_str == "o1-preview":
                    # o1 models don't use system messages
                    messages = [{"role": "user", "content": system_prompt + "\n\n" + prompt}]
                    response = await _async_openai_client.chat.completions.create(
                        model=actual_model_name,
                        messages=messages,
                    )
                else:
                    response = await _async_openai_client.chat.completions.create(
                        model=actual_model_name,
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                
                answer = response.choices[0].message.content
                answer = re.sub(r"\s+", " ", answer).strip()
                
                if answer:
                    return answer.lower()
                    
            elif model_str == "claude3.5sonnet":
                if not _anthropic_client:
                    raise Exception("Anthropic client not initialized. Set ANTHROPIC_API_KEY environment variable.")
                
                actual_model_name = "claude-3-5-sonnet-20240620"
                message = await _anthropic_client.messages.create(
                    model=actual_model_name,
                    system=system_prompt,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = message.content[0].text
                answer = re.sub(r"\s+", " ", answer).strip()
                
                if answer:
                    return answer.lower()
            else:
                raise Exception(f"Unsupported model: {model_str}")
                
        except Exception as e:
            await asyncio.sleep(timeout)
            continue
    
    raise Exception("Max retries: timeout")


async def compare_results_async(
    diagnosis: str,
    correct_diagnosis: str,
    moderator_llm: str = DEFAULT_MODERATOR_LLM
) -> str:
    """
    Compare diagnosis with correct diagnosis using LLM moderator
    
    Args:
        diagnosis: The extracted diagnosis answer to evaluate
        correct_diagnosis: The correct diagnosis from the dataset
        moderator_llm: The LLM to use for comparison (default: "gpt4o")
    
    Returns:
        "yes" if diagnoses match, "no" otherwise
    """
    # Handle both dict and str types for correct_diagnosis
    if isinstance(correct_diagnosis, dict):
        correct_diagnosis_str = str(correct_diagnosis)
    else:
        correct_diagnosis_str = str(correct_diagnosis)

    print('11111111111111111111111')
    print(f"correct_diagnosis_str: {correct_diagnosis_str}")
    print(f"diagnosis: {diagnosis}")
    
    prompt = (
        f"\nHere is the correct diagnosis: {correct_diagnosis_str}\n"
        f"Here was the doctor dialogue: {diagnosis}\n"
        f"Are these the same?"
    )
    
    system_prompt = (
        "You are responsible for determining if the correct diagnosis and the doctor diagnosis "
        "are the same disease. Please respond only with Yes or No. Nothing else."
    )
    
    answer = await query_model_async(moderator_llm, prompt, system_prompt)
    return answer.lower()


async def eval_score_async(
    extracted_answer: str,
    instance_id: int,
    set_type: str = "test",
    moderator_llm: str = DEFAULT_MODERATOR_LLM,
    api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None
) -> float:
    """
    Evaluate a diagnosis answer against the correct diagnosis (async version)
    
    Args:
        extracted_answer: The extracted diagnosis answer to evaluate
        instance_id: The ID/index of the example in the dataset
        set_type: "test" or "validate" (default: "test")
        moderator_llm: The LLM to use for comparison (default: "gpt4o")
        api_key: OpenAI API key (optional, can use OPENAI_API_KEY env var)
        anthropic_api_key: Anthropic API key (optional, can use ANTHROPIC_API_KEY env var)
    
    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    # Initialize clients if API keys are provided
    if api_key or anthropic_api_key:
        init_clients(api_key=api_key, anthropic_api_key=anthropic_api_key)
    else:
        # Try to initialize from environment variables
        init_clients()
    
    # Get correct diagnosis
    try:
        correct_diagnosis = get_correct_diagnosis(instance_id, set_type)
    except Exception as e:
        print(f"Error loading correct diagnosis for instance_id {instance_id}: {e}")
        return 0.0
    
    # Compare results
    try:
        comparison_result = await compare_results_async(
            extracted_answer,
            correct_diagnosis,
            moderator_llm=moderator_llm
        )
        
        # Return 1.0 if "yes", 0.0 if "no"
        is_correct = comparison_result == "yes" or "yes" in comparison_result
        return 1.0 if is_correct else 0.0
        
    except Exception as e:
        print(f"Error evaluating answer for instance_id {instance_id}: {e}")
        return 0.0


def eval_score_sync(
    extracted_answer: str,
    instance_id: int,
    set_type: str = "test",
    moderator_llm: str = DEFAULT_MODERATOR_LLM,
    api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None
) -> float:
    """
    Evaluate a diagnosis answer against the correct diagnosis (synchronous wrapper)
    
    This is a synchronous wrapper around eval_score_async for use in non-async contexts.
    
    Args:
        extracted_answer: The extracted diagnosis answer to evaluate
        instance_id: The ID/index of the example in the dataset
        set_type: "test" or "validate" (default: "test")
        moderator_llm: The LLM to use for comparison (default: "gpt4o")
        api_key: OpenAI API key (optional, can use OPENAI_API_KEY env var)
        anthropic_api_key: Anthropic API key (optional, can use ANTHROPIC_API_KEY env var)
    
    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    try:
        # Check if there's already an event loop running
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If there's a running loop, we need to use a different approach
            # Create a new event loop in a thread
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        eval_score_async(
                            extracted_answer, instance_id, set_type,
                            moderator_llm, api_key, anthropic_api_key
                        )
                    )
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        else:
            # No running loop, we can use asyncio.run
            return asyncio.run(
                eval_score_async(
                    extracted_answer, instance_id, set_type,
                    moderator_llm, api_key, anthropic_api_key
                )
            )
    except RuntimeError:
        # No event loop, use asyncio.run
        return asyncio.run(
            eval_score_async(
                extracted_answer, instance_id, set_type,
                moderator_llm, api_key, anthropic_api_key
            )
        )

