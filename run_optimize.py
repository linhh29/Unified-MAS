"""
Pipeline 执行文件
从 generated_nodes.json 中加载节点代码和 Connections，并执行 pipeline
"""
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# 添加父目录到路径，以便导入模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from Unified_MAS.llm_client import LLMClient
from Unified_MAS.search_engines import SearchEngineFactory
from Unified_MAS.web_search_llm import WebSearchLLM
from Unified_MAS.prompts import get_debug_prompt
from Unified_MAS.utils import get_j_tilde, create_pipeline_executor, build_question_and_answer


def load_generated_nodes(json_path: str) -> Dict[str, Any]:
    """从 JSON 文件加载生成的节点"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="执行生成的 pipeline")
    parser.add_argument(
        "--nodes_json",
        type=str,
        default="xx/j1eval/generated_nodes.json",
        help="生成的节点 JSON 文件路径"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        help="输入数据 JSON 文件路径（可选，如果不提供则使用示例数据）"
    )
    parser.add_argument(
        "--meta_model",
        type=str,
        default="gemini-3-pro-preview",
        help="Meta-model 模型名称（用于 debug 过程）"
    )
    parser.add_argument(
        "--executor_model",
        type=str,
        default="gemini-3-flash-preview",
        help="Executor-model 模型名称（用于每个 node 里面的 executor）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="LLM 温度参数"
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=8192,
        help="最大完成 tokens 数"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="j1eval",
        help="数据集名称"
    )
    parser.add_argument(
        "--max_search_results",
        type=int,
        default=10,
        help="最大搜索结果数量"
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=10,
        help="最大搜索轮数"
    )
    parser.add_argument(
        "--max_debug_attempts",
        type=int,
        default=3,
        help="每个样本允许的最大 debug 尝试次数"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Epoch 数目；每 epoch 跑完整个 validation set，按平均 reward 选节点优化，仅用对应样本更新该节点"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="并行执行的 worker 数；1 表示顺序执行，>1 表示多线程并行"
    )
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=None,
        help="每 epoch 运行的样本数；不指定则用全部样本。用于快速验证 pipeline（如 4 样本 × 3 epochs 快速跑通）"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="不从断点续跑，强制从 epoch 1 重新开始（默认会扫描 rounds 目录从最新 epoch 续跑）"
    )
    args = parser.parse_args()
    
    # 加载生成的节点
    update_nodes_json_path = args.nodes_json.replace('/search/', '/optimize/')
    if os.path.exists(update_nodes_json_path):
        args.nodes_json = update_nodes_json_path
    print(f"Loading nodes from: {args.nodes_json}")
    nodes_data = load_generated_nodes(args.nodes_json)
    print(f"Loaded {len(nodes_data.get('nodes', []))} nodes")
    print(f"Pipeline description: {nodes_data.get('pipeline_description', 'N/A')}")
    
    # 创建 Meta LLM 客户端（用于 debug）
    meta_llm_client = LLMClient(
        model=args.meta_model,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens
    )
    
    # 创建 Executor LLM 客户端（用于节点执行）
    executor_llm_client = LLMClient(
        model=args.executor_model,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens
    )
    
    # 创建搜索引擎（使用 Google 作为默认，因为有些节点可能需要）
    search_engine = SearchEngineFactory.create_engine(
        "google",
        api_key=os.getenv("SERPER_API_KEY"),
    )
    llm_google = WebSearchLLM(
        llm_client=executor_llm_client,  # 使用 executor-model
        search_engine=search_engine,
        max_search_results=args.max_search_results,
        max_rounds=args.max_rounds,
        dataset_name=args.dataset_name,
        mode="optimize",
    )
    
    # 创建 pipeline executor
    print("\nCreating pipeline executor...")
    print(f"Meta-model: {args.meta_model} (for debug)")
    print(f"Executor-model: {args.executor_model} (for node execution)")
    print(f"Epoch mode: {args.num_epochs} epochs (每 epoch 跑完 validation set 后按平均 reward 选节点优化)")
    print(f"Max workers: {args.max_workers} ({'并行' if args.max_workers > 1 else '顺序'}执行)")
    executor = create_pipeline_executor(
        nodes_data, executor_llm_client, llm_google, meta_llm_client, args.dataset_name,
        args.num_epochs
    )
    
    # 加载输入数据
    print(f"Loading input data from: {args.input_data}")
    input_path = Path(args.input_data)
    
    # 读取所有样本
    all_questions = []
    all_answers = []
    with open(args.input_data, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                tmp = build_question_and_answer(args.dataset_name, json.loads(line))
                all_questions.append(tmp[0])
                all_answers.append(tmp[1])
    
    print(f"Loaded {len(all_questions)} questions")
    
    # 快速验证模式：每 epoch 仅用前 N 个样本
    if args.samples_per_epoch is not None:
        all_questions = all_questions[: args.samples_per_epoch]
        all_answers = all_answers[: args.samples_per_epoch]
        print(f"[Quick validation] Using first {len(all_questions)} samples per epoch")
    
    # 执行 pipeline 对所有问题
    print("\n" + "="*80)
    print("Executing pipeline for all questions...")
    print("="*80)
    
    # 保存所有结果到 optimize 目录
    intermediate_dir = Path(__file__).parent / "intermediate_result" / args.dataset_name
    optimize_dir = intermediate_dir / "optimize"
    optimize_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {optimize_dir}")
    
    success_count = 0
    error_count = 0
    all_results = []
    
    def _init_worker(nd, ecl, lg, mllm, ds, ne):
        """每个 worker 线程初始化时创建独立的 executor（避免共享 intermediate_outputs 等状态）"""
        worker_exec = create_pipeline_executor(nd, ecl, lg, mllm, ds, ne)
        threading.current_thread()._executor = worker_exec
    
    def _process_sample(args_tuple):
        """处理单个样本，返回 (idx, result_data, buffer_entry)"""
        idx, question, answer = args_tuple
        max_debug_attempts = args.max_debug_attempts
        worker_executor = getattr(threading.current_thread(), '_executor', executor)
        
        for attempt in range(max_debug_attempts):
            try:
                result, buffer_entry = worker_executor.execute_pipeline(question, answer, sample_index=idx)
                result_data = {
                    "sample_id": idx,
                    "input": question,
                    "answer": answer,
                    "output": result,
                    "status": "success"
                }
                return (idx, result_data, buffer_entry)
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                error_info = {
                    "error_type": error_type,
                    "error_message": error_msg,
                    "attempt": attempt + 1,
                    "traceback": None
                }
                if attempt == max_debug_attempts - 1:
                    import traceback
                    error_info["traceback"] = traceback.format_exc()
                    result_data = {
                        "sample_id": idx,
                        "input": question,
                        "answer": answer,
                        "output": None,
                        "status": "failed",
                        "error": error_info
                    }
                    return (idx, result_data, None)
                if worker_executor is executor:
                    try:
                        debug_success, _, _ = executor.debug_pipeline(question, nodes_data, max_iterations=1)
                        if debug_success:
                            continue
                    except Exception:
                        pass
                result_data = {
                    "sample_id": idx,
                    "input": question,
                    "answer": answer,
                    "output": None,
                    "status": "failed",
                    "error": error_info
                }
                return (idx, result_data, None)
    
    rounds_dir = optimize_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    
    # 断点续跑：扫描 rounds 目录，从最新 epoch 继续
    start_epoch = 0
    if not args.no_resume and rounds_dir.exists():
        epoch_files = list(rounds_dir.glob("epoch_*_generated_nodes.json"))
        if epoch_files:
            epoch_nums = []
            for f in epoch_files:
                m = re.match(r"epoch_(\d+)_generated_nodes\.json", f.name)
                if m:
                    epoch_nums.append(int(m.group(1)))
            if epoch_nums:
                start_epoch = max(epoch_nums)
                print(f"\n[Resume] Found {len(epoch_files)} epoch file(s) in {rounds_dir}")
                print(f"[Resume] Resuming from epoch {start_epoch + 1} (skip epochs 1~{start_epoch})")
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*80}")
        
        # 从第 2 个 epoch 开始，加载上一轮优化后的节点（epoch_1, epoch_2, ...）
        if epoch >= 1:
            prev_epoch_file = rounds_dir / f"epoch_{epoch}_generated_nodes.json"
            if prev_epoch_file.exists():
                print(f"Loading nodes from previous epoch: {prev_epoch_file}")
                nodes_data = load_generated_nodes(str(prev_epoch_file))
                print(f"Loaded {len(nodes_data.get('nodes', []))} nodes")
                executor = create_pipeline_executor(
                    nodes_data, executor_llm_client, llm_google, meta_llm_client, args.dataset_name,
                    args.num_epochs
                )
            else:
                print(f"[Warning] {prev_epoch_file} not found, using current nodes_data")
        
        # 每个 epoch 开始时用第一个样本做 debug（新加载的 node 可能有 bug）
        if all_questions:
            print(f"\n[Epoch {epoch + 1}] Debug with first sample...")
            first_question = all_questions[0]
            success, fixed_nodes_data, was_fixed = executor.debug_pipeline(first_question, nodes_data, max_iterations=10)
            if success:
                if was_fixed:
                    print(f"\n✓ [Epoch {epoch + 1}] Debug fixed bugs. Updating nodes_data and executor.")
                    nodes_data = fixed_nodes_data
                    executor = create_pipeline_executor(
                        nodes_data, executor_llm_client, llm_google, meta_llm_client, args.dataset_name,
                        args.num_epochs
                    )
                    # 保存修复，确保下次加载时含修复
                    if epoch >= 1:
                        fix_file = rounds_dir / f"epoch_{epoch}_generated_nodes.json"
                        with open(fix_file, 'w', encoding='utf-8') as f:
                            json.dump(nodes_data, f, ensure_ascii=False, indent=2)
                        print(f"  Saved fix to {fix_file}")
                    else:
                        fix_file = optimize_dir / "generated_nodes.json"
                        with open(fix_file, 'w', encoding='utf-8') as f:
                            json.dump(nodes_data, f, ensure_ascii=False, indent=2)
                        print(f"  Saved fix to {fix_file}")
                else:
                    print(f"\n✓ [Epoch {epoch + 1}] Debug OK: first sample runs without bugs.")
            else:
                print(f"\n✗ [Epoch {epoch + 1}] Debug failed. Proceeding with current nodes_data.")
        
        epoch_success_count = 0
        epoch_error_count = 0
        epoch_output_file = optimize_dir / f"validate_results_epoch_{epoch + 1}.jsonl"
        collected_buffer = []
        
        if args.max_workers <= 1:
            # 顺序执行
            with open(epoch_output_file, 'w', encoding='utf-8') as f:
                for idx, (question, answer) in enumerate(zip(all_questions, all_answers), 1):
                    print(f"\n--- Processing question {idx}/{len(all_questions)} ---")
                    idx_ret, result_data, buffer_entry = _process_sample((idx, question, answer))
                    if result_data["status"] == "success":
                        epoch_success_count += 1
                        print(f"✓ Question {idx} processed successfully")
                        if buffer_entry:
                            collected_buffer.append(buffer_entry)
                    else:
                        epoch_error_count += 1
                        err = result_data.get("error", {})
                        err_msg = err.get("error_message", str(err))
                        print(f"✗ Sample {idx} failed: {err_msg}")
                    f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
                    f.flush()
        else:
            # 并行执行：每个 worker 有独立 executor（每 epoch 重新创建 pool 以使用更新后的 nodes_data）
            init_args = (nodes_data, executor_llm_client, llm_google, meta_llm_client, args.dataset_name, args.num_epochs)
            with ThreadPoolExecutor(max_workers=args.max_workers, initializer=_init_worker, initargs=init_args) as pool:
                tasks = [(idx, q, a) for idx, (q, a) in enumerate(zip(all_questions, all_answers), 1)]
                results_by_idx = {}
                for future in as_completed(pool.submit(_process_sample, t) for t in tasks):
                    idx_ret, result_data, buffer_entry = future.result()
                    results_by_idx[idx_ret] = (result_data, buffer_entry)
                    status = result_data["status"]
                    if status == "success":
                        epoch_success_count += 1
                        print(f"✓ Question {idx_ret} done")
                        if buffer_entry:
                            collected_buffer.append(buffer_entry)
                    else:
                        epoch_error_count += 1
                        err = result_data.get("error", {})
                        err_msg = err.get("error_message", str(err))
                        print(f"✗ Sample {idx_ret} failed: {err_msg}")
            
            with open(epoch_output_file, 'w', encoding='utf-8') as f:
                for idx in sorted(results_by_idx.keys()):
                    result_data, _ = results_by_idx[idx]
                    f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
        
        success_count = epoch_success_count
        error_count = epoch_error_count
        
        # Epoch 模式：跑完本 epoch 所有样本后执行优化，保存到 rounds/epoch_{epoch+1}_generated_nodes.json
        # 下一轮 epoch 开始时会从该文件加载并重建 executor
        was_optimized = executor.perform_epoch_optimization(buffer=collected_buffer, epoch=epoch + 1)
        if was_optimized:
            print(f"\n[Epoch {epoch + 1}] Node optimized. Saved to rounds/epoch_{epoch + 1}_generated_nodes.json")
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {optimize_dir}/validate_results_epoch_*.jsonl")
    print(f"{'='*80}")
    
    # 打印统计信息
    print(f"\nSummary: {success_count} successful, {error_count} failed out of {len(all_questions)} total")
    
    # ========= 保存累计成本到 cost.txt =========
    total_cost = meta_llm_client.total_cost + executor_llm_client.total_cost
    cost_file = Path(__file__).parent / "cost.txt"
    
    with open(cost_file, 'a', encoding='utf-8') as f:
        f.write(
            f"run_optimize.py - Meta Model: {args.meta_model}, Executor Model: {args.executor_model}, "
            f"Dataset: {args.dataset_name}, Total Cost: ${total_cost:.6f} "
            f"(Meta: ${meta_llm_client.total_cost:.6f}, Executor: ${executor_llm_client.total_cost:.6f})\n"
        )
    
    print(f"\n{'='*80}")
    print(f"[Cost Summary] Total accumulated cost: ${total_cost:.6f}")
    print(f"[Cost Summary]   - Meta model ({args.meta_model}): ${meta_llm_client.total_cost:.6f}")
    print(f"[Cost Summary]   - Executor model ({args.executor_model}): ${executor_llm_client.total_cost:.6f}")
    print(f"[Cost Summary] Cost saved to: {cost_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

