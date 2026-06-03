"""
Demo inference: run the Unified-MAS search stage on a custom question (no optimize).

Usage:
    python demo_inference.py --question "Your task description here" --model gemini-3-pro-preview
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from Unified_MAS.llm_client import LLMClient
from Unified_MAS.web_search_llm import WebSearchLLM
from Unified_MAS.search_engines import SearchEngineFactory
from Unified_MAS.content_fetcher import fetch_urls_from_log
from Unified_MAS.strategy_analyzer import analyze_all_strategies
from Unified_MAS.prompts import (
    get_task_keywords_prompt,
    get_search_queries_prompt,
    get_node_generation_prompt,
)
from Unified_MAS.code_definition import code_template

# ---------------------------------------------------------------------------
# Default configuration — edit here to tune search behavior
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-3-pro-preview"
TEMPERATURE = 1.0
MAX_COMPLETION_TOKENS = 32768
MAX_SEARCH_RESULTS = 10
MAX_ROUNDS = 3
MAX_CONCURRENT = 1
RUN_NAME = "custom"
FORCE_RERUN = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified-MAS demo inference: custom question → search → generated nodes"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Custom task description or question",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args()


def _load_or_generate(path: Path, generator, force_rerun: bool):
    if not force_rerun and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"\n[Loaded] {path.name} from: {path}")
        return data

    data = generator()
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {path.name} to: {path}")
    return json.loads(data) if isinstance(data, str) else data


def main(args=None):
    if args is None:
        args = parse_args()

    question = args.question.strip()
    if not question:
        raise ValueError("--question must not be empty")

    run_name = RUN_NAME
    dataset_key = f"demo/{run_name}"
    force = FORCE_RERUN

    intermediate_dir = Path(__file__).parent / "intermediate_result"
    search_dir = intermediate_dir / "demo" / run_name / "search"
    search_dir.mkdir(parents=True, exist_ok=True)

    question_file = search_dir / "custom_question.txt"
    with open(question_file, "w", encoding="utf-8") as f:
        f.write(question)
    print(f"[Demo] Question saved to: {question_file}")

    llm_client = LLMClient(
        model=args.model,
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )

    google_engine = SearchEngineFactory.create_engine(
        "google", api_key=os.getenv("SERPER_API_KEY")
    )
    scholar_engine = SearchEngineFactory.create_engine(
        "scholar", api_key=os.getenv("SERPER_API_KEY")
    )
    github_engine = SearchEngineFactory.create_engine(
        "github", api_key=os.getenv("GITHUB_TOKEN")
    )

    def make_web_search_llm(engine):
        return WebSearchLLM(
            llm_client=llm_client,
            search_engine=engine,
            max_search_results=MAX_SEARCH_RESULTS,
            max_rounds=MAX_ROUNDS,
            dataset_name=dataset_key,
            mode="search",
        )

    llm_google = make_web_search_llm(google_engine)
    llm_scholar = make_web_search_llm(scholar_engine)
    llm_github = make_web_search_llm(github_engine)

    # Step 1: task keywords from custom question
    keywords_file = search_dir / "task_keywords.txt"

    def generate_task_keywords():
        summarize_system, summarize_user = get_task_keywords_prompt(question)
        messages = [
            {"role": "system", "content": summarize_system},
            {"role": "user", "content": summarize_user},
        ]
        return llm_client.chat(messages, response_format="json_object")

    task_keywords = _load_or_generate(keywords_file, generate_task_keywords, force)
    task_thinking = task_keywords.get("thinking", "")
    print(task_keywords)

    # Step 2: search queries
    keywords_json_str = json.dumps(task_keywords, ensure_ascii=False, indent=2)
    queries_file = search_dir / "search_queries.txt"

    def generate_search_queries():
        search_system, search_user = get_search_queries_prompt(keywords_json_str)
        messages = [
            {"role": "system", "content": search_system},
            {"role": "user", "content": search_user},
        ]
        return llm_client.chat(messages, response_format="json_object")

    search_queries = _load_or_generate(queries_file, generate_search_queries, force)

    print(search_queries["strategy_A"])
    print(search_queries["strategy_B"])
    print(search_queries["strategy_C"])
    print(search_queries["strategy_D"])

    strategy_details = {
        "strategy_A": "Strategy A - Background Knowledge",
        "strategy_B": "Strategy B - High-quality Academic Papers about System Architecture (Workflow & Design)",
        "strategy_C": "Strategy C - Technical Code Implementation",
        "strategy_D": "Strategy D - Evaluation & Metrics",
    }

    target_description_template = """
    Strategy: {Strategy}
    Query: {Query}
    """

    # Step 3: multi-turn web search
    tasks_list = []
    for strategy in search_queries:
        if strategy == "strategy_A":
            llms = [llm_google, llm_scholar]
        elif strategy == "strategy_B":
            llms = [llm_google, llm_scholar]
        elif strategy == "strategy_C":
            llms = [llm_github]
        elif strategy == "strategy_D":
            llms = [llm_google, llm_scholar]
        else:
            continue

        for query in search_queries[strategy]:
            for llm in llms:
                target_description = target_description_template.format(
                    Strategy=strategy_details[strategy],
                    Query=query["reasoning"],
                )
                tasks_list.append((llm, target_description, strategy, query["query"]))

    multi_turn_search_log = search_dir / "multi_turn_search_log.jsonl"

    async def async_multi_turn_search(llm, target_description, strategy, query_text, semaphore, pbar):
        async with semaphore:
            result = await llm.multi_turn_search_async(target_description)
            pbar.set_postfix(
                {
                    "strategy": strategy,
                    "query": query_text[:30] + "..." if len(query_text) > 30 else query_text,
                }
            )
            print(f"\nTarget Description: {target_description}")
            print("=" * 80)
            print(result)
            pbar.update(1)
            return result

    if force and multi_turn_search_log.exists():
        multi_turn_search_log.unlink()

    if multi_turn_search_log.exists():
        print(f"\n[Loaded] Multi-turn search log exists: {multi_turn_search_log}, skipping search.")
    else:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        pbar = tqdm(total=len(tasks_list), desc="Searching queries", unit="query")

        async def run_all_tasks():
            tasks = [
                async_multi_turn_search(llm, target_desc, strategy, query_text, semaphore, pbar)
                for llm, target_desc, strategy, query_text in tasks_list
            ]
            await asyncio.gather(*tasks)
            pbar.close()

        asyncio.run(run_all_tasks())

    # Step 4: fetch URL contents
    fetched_contents_cache_file = search_dir / "fetched_contents.json"
    if force and fetched_contents_cache_file.exists():
        fetched_contents_cache_file.unlink()

    if fetched_contents_cache_file.exists():
        print(f"\n[Fetch URLs] Using cache: {fetched_contents_cache_file}")
        with open(fetched_contents_cache_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        print("\n[Fetch URLs] Fetching URLs from search log...")
        all_results = fetch_urls_from_log(
            llm_google.log_file,
            dataset_key,
            github_token=os.getenv("GITHUB_TOKEN"),
        )
        with open(fetched_contents_cache_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"[Fetch URLs] Saved to: {fetched_contents_cache_file}")

    # Step 5: strategy analysis
    strategy_analysis_file = search_dir / "strategy_analysis.json"
    if force and strategy_analysis_file.exists():
        strategy_analysis_file.unlink()

    if strategy_analysis_file.exists():
        with open(strategy_analysis_file, "r", encoding="utf-8") as f:
            strategy_analysis = json.load(f)
        print(f"\n[Loaded] Strategy analysis from: {strategy_analysis_file}")
    else:
        strategy_analysis = analyze_all_strategies(
            all_results=all_results,
            task_thinking=task_thinking,
            llm_client=llm_client,
            dataset=dataset_key,
            intermediate_dir=search_dir,
        )

    # Step 6: node generation
    print(f"\n{'=' * 80}")
    print("[Node Generation] Starting node generation...")
    print(f"{'=' * 80}")

    strategy_analysis_str = json.dumps(strategy_analysis, ensure_ascii=False, indent=2)
    node_system, node_user = get_node_generation_prompt(
        task_thinking=task_thinking,
        strategy_analysis=strategy_analysis_str,
        code_template=code_template,
        task_samples=question,
    )
    node_messages = [
        {"role": "system", "content": node_system},
        {"role": "user", "content": node_user},
    ]

    nodes_output_file = search_dir / "generated_nodes.json"
    if force and nodes_output_file.exists():
        nodes_output_file.unlink()

    if nodes_output_file.exists():
        with open(nodes_output_file, "r", encoding="utf-8") as f:
            nodes_result = json.load(f)
        print(f"\n[Loaded] Generated nodes from: {nodes_output_file}")
    else:
        print("[Node Generation] Calling LLM to generate nodes...")
        nodes_result_str = llm_client.chat(node_messages, response_format="json_object")
        nodes_result = json.loads(nodes_result_str)
        with open(nodes_output_file, "w", encoding="utf-8") as f:
            json.dump(nodes_result, f, ensure_ascii=False, indent=2)

    for node in nodes_result.get("nodes", []):
        print(node.get("all_code", ""))
        print("=" * 80)
    print(nodes_result.get("Connections", {}))

    print(f"[Node Generation] Nodes saved to: {nodes_output_file}")
    print(f"[Node Generation] Total nodes: {len(nodes_result.get('nodes', []))}")

    total_cost = llm_client.total_cost
    cost_file = Path(__file__).parent / "cost.txt"
    with open(cost_file, "a", encoding="utf-8") as f:
        f.write(
            f"demo_inference.py - Model: {args.model}, Run: {run_name}, Total Cost: ${total_cost:.6f}\n"
        )

    print(f"\n{'=' * 80}")
    print(f"[Cost Summary] Total cost: ${total_cost:.6f}")
    print(f"[Cost Summary] Appended to: {cost_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main(parse_args())
