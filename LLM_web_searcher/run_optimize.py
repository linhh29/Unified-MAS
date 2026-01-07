"""
Pipeline 执行文件
从 generated_nodes.json 中加载节点代码和 Connections，并执行 pipeline
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# 添加父目录到路径，以便导入模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from LLM_web_searcher.llm_client import LLMClient
from LLM_web_searcher.search_engines import SearchEngineFactory
from LLM_web_searcher.web_search_llm import WebSearchLLM
from LLM_web_searcher.prompts import get_debug_prompt
from LLM_web_searcher.utils import get_j_tilde, create_pipeline_executor, build_question_and_answer


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
        default="/data/qin/lhh/Unified-MAS/LLM_web_searcher/intermediate_result/j1eval/generated_nodes.json",
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
    executor = create_pipeline_executor(nodes_data, executor_llm_client, llm_google, meta_llm_client, args.dataset_name)
    
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
    
    # 使用第一个样本进行 debug
    if all_questions and '/optimize/' not in args.nodes_json:
        print("\n" + "="*80)
        print("Starting debug process with first question...")
        print("="*80)
        
        first_question = all_questions[0]
        success, fixed_nodes_data = executor.debug_pipeline(first_question, nodes_data, max_iterations=10)
        
        if success:
            print("\n✓ Debug successful! Pipeline is now working.")
            
            # 保存修复后的代码
            intermediate_dir = Path(__file__).parent / "intermediate_result" / args.dataset_name
            optimize_dir = intermediate_dir / "optimize"
            optimize_dir.mkdir(parents=True, exist_ok=True)
            
            debugged_output_file = optimize_dir / "generated_nodes.json"
            with open(debugged_output_file, 'w', encoding='utf-8') as f:
                json.dump(fixed_nodes_data, f, ensure_ascii=False, indent=2)
            print(f"\n✓ Fixed code saved to: {debugged_output_file}")
            
            # 更新 nodes_data 和 executor
            nodes_data = fixed_nodes_data
            executor = create_pipeline_executor(nodes_data, executor_llm_client, llm_google, meta_llm_client, args.dataset_name)
        else:
            print("\n✗ Debug failed. Using original code.")
    
    # 执行 pipeline 对所有问题
    print("\n" + "="*80)
    print("Executing pipeline for all questions...")
    print("="*80)
    
    # 保存所有结果到 optimize 目录
    # 构建 optimize 目录路径：intermediate_result/j1eval/optimize
    intermediate_dir = Path(__file__).parent / "intermediate_result" / args.dataset_name
    optimize_dir = intermediate_dir / "optimize"
    
    # 检查并创建 optimize 目录
    optimize_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {optimize_dir}")
    
    # 创建输出文件（使用 JSONL 格式，每行一个 JSON 对象）
    output_file = optimize_dir / f"validate_results.jsonl"
    
    success_count = 0
    error_count = 0
    all_results = []
    
    # 打开文件进行追加写入
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, (question, answer) in enumerate(zip(all_questions, all_answers), 1):
            print(f"\n--- Processing question {idx}/{len(all_questions)} ---")
            
            # 每个样本允许最多 max_debug_attempts 次 debug 尝试
            max_debug_attempts = args.max_debug_attempts
            success = False
            result = None
            error_info = None
            
            for attempt in range(max_debug_attempts):
                try:
                    # 执行 pipeline，传递 sample index
                    result = executor.execute_pipeline(question, answer, sample_index=idx)
                    success = True
                    break  # 成功则跳出循环
                    
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    error_info = {
                        "error_type": error_type,
                        "error_message": error_msg,
                        "attempt": attempt + 1,
                        "traceback": None
                    }
                    
                    # 如果是最后一次尝试，记录完整的 traceback
                    if attempt == max_debug_attempts - 1:
                        import traceback
                        error_info["traceback"] = traceback.format_exc()
                        print(f"\n✗ Question {idx} failed after {max_debug_attempts} attempts")
                        print(f"Error: {error_type}: {error_msg}")
                        break
                    
                    # 尝试 debug（使用 debug_pipeline 方法）
                    print(f"\n⚠ Question {idx} failed on attempt {attempt + 1}/{max_debug_attempts}")
                    print(f"Error: {error_type}: {error_msg}")
                    print(f"Attempting to debug and fix...")
                    
                    try:
                        # 调用 debug_pipeline 来修复代码
                        debug_success, updated_nodes_data = executor.debug_pipeline(
                            question, 
                            nodes_data, 
                            max_iterations=1  # 每次 debug 只尝试一次修复
                        )
                        
                        if debug_success:
                            print(f"✓ Debug successful, retrying execution...")
                            # debug_pipeline 已经更新了 executor 的方法，继续下一次尝试
                            continue
                        else:
                            print(f"✗ Debug failed, will retry with original code...")
                            # debug 失败，继续下一次尝试（使用原始代码）
                            continue
                            
                    except Exception as debug_error:
                        print(f"✗ Debug process itself failed: {debug_error}")
                        # debug 过程本身出错，继续下一次尝试
                        continue
            
            # 根据结果创建 result_data
            if success:
                result_data = {
                    "sample_id": idx,
                    "input": question,
                    "answer": answer,
                    "output": result,
                    "status": "success"
                }
                all_results.append(result_data)
                success_count += 1
                print(f"✓ Question {idx} processed successfully and saved")
            else:
                result_data = {
                    "sample_id": idx,
                    "input": question,
                    "answer": answer,
                    "output": None,
                    "status": "failed",
                    "error": error_info
                }
                all_results.append(result_data)
                error_count += 1
                print(f"✗ Sample {idx} failed after {max_debug_attempts} attempts and marked as failed")
            
            # 立即写入结果到文件（JSONL 格式：每行一个 JSON 对象）
            f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
            f.flush()  # 确保立即写入磁盘
            
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_file}")
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

