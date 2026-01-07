"""
主执行文件
使用示例
"""
import argparse
import os
import sys
import asyncio
from pathlib import Path
from tqdm import tqdm

# 添加父目录到路径，以便导入模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_client import LLMClient
from LLM_web_searcher.web_search_llm import WebSearchLLM
from LLM_web_searcher.search_engines import SearchEngineFactory
from LLM_web_searcher.content_fetcher import fetch_urls_from_log
from LLM_web_searcher.strategy_analyzer import analyze_all_strategies, clean_json_response, fix_unicode_surrogates
from LLM_web_searcher.prompts import get_task_keywords_prompt, get_search_queries_prompt, get_node_generation_prompt
from LLM_web_searcher.code_definition import code_template
import sys
import importlib.util
import json
import re
from LLM_web_searcher.utils import build_question_and_answer, J1_INSTRUCTION, HospSumm_INSTRUCTION, PLANNER_INSTRUCTION

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Web Search LLM 示例")
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-pro-preview",
        help="LLM 模型名称 (默认: gemini-3-pro-preview)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="LLM 温度参数 (默认: 0.5)"
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=8192,
        help="最大完成 tokens 数 (默认: 8192)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/j1eval_validate.jsonl",
        help="数据文件路径 (默认: /data/qin/lhh/Unified-MAS/MAS-Zero/data/src/j1eval_validate.jsonl)"
    )
    parser.add_argument(
        "--max_search_results",
        type=int,
        default=10,
        help="每轮搜索返回的最大结果数 (默认: 10)"
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=5,
        help="multi-turn 搜索的最大轮数 (默认: 5)"
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=3,
        help="异步执行时的最大并发数 (默认: 3)"
    )
    return parser.parse_args()


def main(args=None):
    """主函数示例"""
    if args is None:
        args = parse_args()
    
    if 'travelplanner' in args.data_path:
        dataset = 'travelplanner'
    elif 'hosp_summ' in args.data_path:
        dataset = 'hosp_summ'
    elif 'j1eval' in args.data_path:
        dataset = 'j1eval'
    else:
        raise ValueError(f"不支持的数据集: {args.data_path}")
    
    # 创建自定义 LLM 客户端
    llm_client = LLMClient(
        model=args.model,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens
    )

    # 实例化三个不同的搜索引擎
    google_engine = SearchEngineFactory.create_engine(
        "google",
        api_key=os.getenv("SERPER_API_KEY"),
    )
    scholar_engine = SearchEngineFactory.create_engine(
        "scholar",
        api_key=os.getenv("SERPER_API_KEY"),
    )
    github_engine = SearchEngineFactory.create_engine(
        "github",
        api_key=os.getenv("GITHUB_TOKEN"),
    )

    # 为每个搜索引擎创建一个 WebSearchLLM 实例
    llm_google = WebSearchLLM(
        llm_client=llm_client,
        search_engine=google_engine,
        max_search_results=args.max_search_results,
        max_rounds=args.max_rounds,
        dataset_name=dataset,
        mode="search",
    )
    llm_scholar = WebSearchLLM(
        llm_client=llm_client,
        search_engine=scholar_engine,
        max_search_results=args.max_search_results,
        max_rounds=args.max_rounds,
        dataset_name=dataset,
        mode="search",
    )
    llm_github = WebSearchLLM(
        llm_client=llm_client,
        search_engine=github_engine,
        max_search_results=args.max_search_results,
        max_rounds=args.max_rounds,
        dataset_name=dataset,
        mode="search",
    )
    

    # 从 JSONL 文件读取样本，并让 LLM 先总结"这个数据集的任务是什么"
    data_path = args.data_path
    print(data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        samples_lines = f.readlines()
    
    # 取前10个样本并合并成一个字符串
    samples_text = '\n'.join(samples_lines[:10])

    # ========= Step 1: 保存 task_keywords 到 intermediate_result 目录 =========
    intermediate_dir = Path(__file__).parent / "intermediate_result"
    intermediate_dir.mkdir(exist_ok=True)
    # 为每个dataset创建子目录，并在dataset下创建search子目录
    dataset_dir = intermediate_dir / dataset
    dataset_dir.mkdir(exist_ok=True)
    search_dir = dataset_dir / "search"
    search_dir.mkdir(exist_ok=True)
    output_file = search_dir / "task_keywords.txt"

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            task_keywords = json.load(f)
        print(f"\n[Loaded] Task keywords loaded from: {output_file}")
    else:
        summarize_system, summarize_user = get_task_keywords_prompt(samples_text)
        summarize_messages = [
            {"role": "system", "content": summarize_system},
            {"role": "user", "content": summarize_user},
        ]
        task_keywords_str = llm_client.chat(summarize_messages, response_format='json_object')
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(str(task_keywords_str))
        print(f"\n[Saved] Task keywords saved to: {output_file}")
        task_keywords = json.loads(task_keywords_str)

    # 打印思考过程和关键词结果
    print(task_keywords["thinking"])
    print(task_keywords["answer"])

    # ========= Step 2: 基于 task_keywords 生成检索 query 组合 =========
    keywords_json_str = json.dumps(task_keywords["answer"], ensure_ascii=False, indent=2)

    # 保存并打印检索 query 组合
    search_output_file = search_dir / "search_queries.txt"

    if os.path.exists(search_output_file):
        with open(search_output_file, "r", encoding="utf-8") as f:
            search_queries = json.load(f)
        print(f"\n[Loaded] Search queries loaded from: {search_output_file}")
    else:
        search_system, search_user = get_search_queries_prompt(keywords_json_str)
        search_messages = [
            {"role": "system", "content": search_system},
            {"role": "user", "content": search_user},
        ]
        search_queries_str = llm_client.chat(search_messages, response_format='json_object')
        with open(search_output_file, "w", encoding="utf-8") as f:
            f.write(str(search_queries_str))
        print(f"\n[Saved] Search queries saved to: {search_output_file}")
        search_queries = json.loads(search_queries_str)

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
    
    # 异步执行multi_turn_search的包装函数
    async def async_multi_turn_search(llm, target_description, strategy, query_text, semaphore, pbar):
        """异步包装multi_turn_search"""
        async with semaphore:
            # 在线程池中运行同步的multi_turn_search
            loop = asyncio.get_event_loop()
            # result = await loop.run_in_executor(None, llm.multi_turn_search, target_description)
            result = None
            pbar.set_postfix({
                "strategy": strategy, 
                "query": query_text[:30] + "..." if len(query_text) > 30 else query_text
            })
            print(f"\nTarget Description: {target_description}")
            print(f"{'=' * 80}")
            print(result)
            pbar.update(1)
            return result
    
    # 计算总的查询任务数量（用于tqdm进度条）
    total_tasks = 0
    tasks_list = []  # 存储所有任务
    
    for strategy in search_queries:
        if strategy == 'strategy_A':
            llms = [llm_google, llm_scholar]
        elif strategy == 'strategy_B':
            llms = [llm_google, llm_scholar]
        elif strategy == 'strategy_C':
            llms = [llm_github]
        elif strategy == 'strategy_D':
            llms = [llm_google, llm_scholar]
        
        for query in search_queries[strategy]:
            for llm in llms:
                target_description = target_description_template.format(
                    Strategy=strategy_details[strategy], 
                    Query=query['reasoning']
                )
                tasks_list.append((llm, target_description, strategy, query['query']))
                total_tasks += 1
    
    # 创建信号量控制并发数
    semaphore = asyncio.Semaphore(args.max_concurrent)
    
    # 创建进度条
    pbar = tqdm(total=total_tasks, desc="Searching queries", unit="query")
    
    # 创建所有异步任务
    async def run_all_tasks():
        tasks = [
            async_multi_turn_search(llm, target_desc, strategy, query_text, semaphore, pbar)
            for llm, target_desc, strategy, query_text in tasks_list
        ]
        await asyncio.gather(*tasks)
        pbar.close()
    
    # 运行所有异步任务
    asyncio.run(run_all_tasks())
    
    # ========= Step 3: 从日志文件获取URL并下载内容 =========
    # 检查是否存在缓存的结果文件
    fetched_contents_cache_file = search_dir / "fetched_contents.json"
    if fetched_contents_cache_file.exists():
        print(f"\n[Fetch URLs] Found cached results file: {fetched_contents_cache_file}")
        with open(fetched_contents_cache_file, 'r') as f:
            all_results = json.load(f)
    else:
        print(f"\n[Fetch URLs] No cached results found. Fetching URLs from log file...")
        all_results = fetch_urls_from_log(llms[0].log_file, dataset, github_token=os.getenv("GITHUB_TOKEN"))
        # 保存结果到缓存文件
        with open(fetched_contents_cache_file, 'w') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"[Fetch URLs] Results saved to cache: {fetched_contents_cache_file}")

    
    # ========= Step 4: 按Strategy分析文件 =========
    task_thinking = task_keywords.get("thinking", "")
    strategy_analysis = analyze_all_strategies(
        all_results=all_results,
        task_thinking=task_thinking,
        llm_client=llm_client,
        dataset=dataset,
        intermediate_dir=search_dir,
    )

    # ========= Step 5: 生成Nodes =========
    print(f"\n{'='*80}")
    print("[Node Generation] Starting node generation...")
    print(f"{'='*80}")
    
    # 将strategy_analysis转换为JSON字符串
    strategy_analysis_str = json.dumps(strategy_analysis, ensure_ascii=False, indent=2)
    
    # 生成prompt（包含任务样本）
    samples_text, _ = build_question_and_answer(dataset, json.loads(samples_lines[0]))
    node_system, node_user = get_node_generation_prompt(
        task_thinking=task_thinking,
        strategy_analysis=strategy_analysis_str,
        code_template=code_template,
        task_samples=samples_text
    )
    
    # 调用LLM生成nodes
    node_messages = [
        {"role": "system", "content": node_system},
        {"role": "user", "content": node_user},
    ]
    

    print("[Node Generation] Calling LLM to generate nodes...")
    nodes_result_str = llm_client.chat(node_messages, response_format='json_object')
    
    # 清理JSON响应
    # cleaned_json = clean_json_response(nodes_result_str)
    nodes_result = json.loads(nodes_result_str)
    
    for node in nodes_result['nodes']:
        print(node['all_code'])
        print("="*80)
    print(nodes_result['Connections'])

    
    # 保存生成的nodes
    nodes_output_file = search_dir / "generated_nodes.json"
    with open(nodes_output_file, 'w') as f:
        json.dump(nodes_result, f, ensure_ascii=False, indent=2)
    
    print(f"[Node Generation] Nodes generated and saved to: {nodes_output_file}")
    print(f"[Node Generation] Total nodes: {len(nodes_result.get('nodes', []))}")
    
    # ========= Step 6: 保存累计成本到 cost.txt =========
    total_cost = llm_client.total_cost
    cost_file = Path(__file__).parent / "cost.txt"
    
    with open(cost_file, 'a', encoding='utf-8') as f:
        f.write(f"run_search.py - Model: {args.model}, Dataset: {dataset}, Total Cost: ${total_cost:.6f}\n")
    
    print(f"\n{'='*80}")
    print(f"[Cost Summary] Total accumulated cost: ${total_cost:.6f}")
    print(f"[Cost Summary] Cost saved to: {cost_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
