import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

from tqdm.asyncio import tqdm_asyncio

from sampler import init_model, get_model
from async_main_question import PLANNER_INSTRUCTION, J1_INSTRUCTION, HospSumm_INSTRUCTION
from score import DataScorer

# 添加路径以导入 LLM_web_searcher 模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from LLM_web_searcher.llm_client import LLMClient
from LLM_web_searcher.search_engines import SearchEngineFactory
from LLM_web_searcher.web_search_llm import WebSearchLLM
from LLM_web_searcher.prompts import get_debug_prompt


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
    parser = argparse.ArgumentParser(description="Multi-Agent System (DAG) evaluation")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name")
    parser.add_argument("--nodes_json", type=str, required=True,
                        help="Path to generated_nodes.json file")
    parser.add_argument("--executor_model", type=str, required=True,
                        help="Executor model name for node execution")
    parser.add_argument("--save_dir", type=str, default="./dag_results/",
                        help="Directory to save responses and logs")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to evaluate (default: all)")
    parser.add_argument("--given_examples", type=int, nargs="*",
                        help="Optional subset of example indices to run")
    parser.add_argument("--max_concurrent", type=int, default=10,
                        help="Maximum number of concurrent pipeline executions")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="LLM temperature parameter")
    parser.add_argument("--max_completion_tokens", type=int, default=8192,
                        help="Maximum completion tokens")
    parser.add_argument("--max_search_results", type=int, default=10,
                        help="Maximum search results")
    parser.add_argument("--max_rounds", type=int, default=10,
                        help="Maximum search rounds")
    parser.add_argument("--meta_model", type=str, default="gemini-3-pro-preview",
                        help="Meta-model name (for debug process)")
    parser.add_argument("--max_debug_attempts", type=int, default=3,
                        help="Maximum number of debug attempts per sample")

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


def load_generated_nodes(json_path: str) -> Dict[str, Any]:
    """从 JSON 文件加载生成的节点"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_pipeline_executor(nodes_data: Dict[str, Any], executor_llm_client: LLMClient, search_engine, meta_llm_client) -> Any:
    """
    创建 PipelineExecutor 类并动态添加所有节点方法和 execute_pipeline 方法
    
    Args:
        nodes_data: 从 generated_nodes.json 加载的数据
        executor_llm_client: Executor LLM 客户端实例（用于节点执行）
        search_engine: 搜索引擎实例
        meta_llm_client: Meta LLM 客户端实例（用于 debug，如果为 None 则使用 executor_llm_client）
    
    Returns:
        PipelineExecutor 实例
    """
    # 创建 PipelineExecutor 类
    class PipelineExecutor:
        def __init__(self, executor_llm_client, search_engine, meta_llm_client):
            self.llm_client = executor_llm_client  # executor-model，用于节点执行
            self.search_engine = search_engine
            self.meta_llm_client = meta_llm_client
    
    # 在类的命名空间中执行所有节点代码
    class_namespace = {}
    
    # 添加所有节点方法
    for node in nodes_data.get('nodes', []):
        node_name = node.get('node_name')
        all_code = node.get('all_code', '')
        
        if node_name and all_code:
            try:
                # 在类的命名空间中执行节点代码
                exec(all_code, {}, class_namespace)
                # 将方法绑定到类
                if node_name in class_namespace:
                    setattr(PipelineExecutor, node_name, class_namespace[node_name])
                else:
                    print(f"  ✗ Warning: Node {node_name} not found after execution")
            except Exception as e:
                print(f"  ✗ Warning: Failed to load node {node_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 添加 execute_pipeline 方法
    connections_code = nodes_data.get('Connections', '')
    if connections_code:
        try:
            # 在类的命名空间中执行 Connections 代码
            exec(connections_code, {}, class_namespace)
            if 'execute_pipeline' in class_namespace:
                setattr(PipelineExecutor, 'execute_pipeline', class_namespace['execute_pipeline'])
            else:
                print(f"  ✗ Warning: execute_pipeline not found after execution")
        except Exception as e:
            print(f"  ✗ Warning: Failed to load execute_pipeline: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ✗ Warning: No Connections code found in nodes_data")
    
    # 添加 debug 方法
    def debug_pipeline(self, sample_input: Dict, nodes_data: Dict, max_iterations: int = 5) -> tuple[bool, Dict[str, Any]]:
        """
        调试 pipeline，自动修复错误
        
        Args:
            sample_input: 样本输入
            nodes_data: 节点数据（用于更新）
            max_iterations: 最大调试次数
            
        Returns:
            (success: bool, fixed_nodes_data: Dict)
        """
        import traceback
        
        for iteration in range(max_iterations):
            print(f"\n{'='*80}")
            print(f"Debug iteration {iteration + 1}/{max_iterations}")
            print(f"{'='*80}")
            
            try:
                # 尝试执行 pipeline
                result = self.execute_pipeline(sample_input)
                print(f"\n✓ Pipeline executed successfully!")
                print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)[:500]}...")
                
                # 成功，返回修复后的代码
                return True, nodes_data
                
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                error_traceback = traceback.format_exc()
                
                print(f"\n✗ Error occurred: {error_type}: {error_msg}")
                print(f"Traceback:\n{error_traceback}")
                
                # 收集错误信息
                error_info = f"""
Error Type: {error_type}
Error Message: {error_msg}
Traceback:
{error_traceback}
"""
                
                # 找出出错的节点（从堆栈跟踪中）
                error_node = None
                error_node_code = None
                for node in nodes_data.get('nodes', []):
                    node_name = node.get('node_name', '')
                    if node_name in error_traceback:
                        error_node = node
                        error_node_code = node.get('all_code', '')
                        break
                
                # 如果找不到具体节点，尝试从错误信息推断
                if not error_node:
                    # 检查是否是 Connections 的问题
                    if 'execute_pipeline' in error_traceback:
                        error_node_code = nodes_data.get('Connections', '')
                    else:
                        # 默认修复第一个节点
                        error_node = nodes_data.get('nodes', [{}])[0] if nodes_data.get('nodes') else None
                        if error_node:
                            error_node_code = error_node.get('all_code', '')
                
                connections_code = nodes_data.get('Connections', '')
                
                # 调用 LLM 修复代码（使用 meta-model）
                print(f"\n[Debug] Calling LLM (meta-model) to fix the error...")
                debug_system, debug_user = get_debug_prompt(
                    error_info, error_node_code or '', connections_code, sample_input
                )
                
                debug_messages = [
                    {"role": "system", "content": debug_system},
                    {"role": "user", "content": debug_user}
                ]
                
                fix_response_str = self.meta_llm_client.chat(debug_messages, response_format='json_object')
                fix_response = json.loads(fix_response_str)
                
                print(f"[Debug] LLM provided fix explanation: {fix_response.get('explanation', 'N/A')}")
                
                # 更新代码
                fixed_node_code = fix_response.get('fixed_node_code', '')
                fixed_connections_code = fix_response.get('fixed_connections_code', '')
                
                if fixed_node_code and error_node:
                    error_node['all_code'] = fixed_node_code
                    print(f"[Debug] Updated node code: {error_node.get('node_name', 'Unknown')}")
                
                if fixed_connections_code:
                    nodes_data['Connections'] = fixed_connections_code
                    print(f"[Debug] Updated Connections code")
                
                # 重新创建 executor 并更新方法（使用 executor-model）
                print(f"[Debug] Recreating pipeline executor with fixed code...")
                new_executor = create_pipeline_executor(nodes_data, self.llm_client, self.search_engine, self.meta_llm_client)
                
                # 更新当前 executor 的所有方法（包括节点方法和 execute_pipeline）
                for attr_name in dir(new_executor):
                    if not attr_name.startswith('_') and attr_name not in ['llm_client', 'search_engine', 'meta_llm_client']:
                        try:
                            attr_value = getattr(new_executor, attr_name)
                            if callable(attr_value):
                                setattr(self, attr_name, attr_value)
                        except:
                            pass
        
        # 达到最大迭代次数，返回失败
        print(f"\n✗ Debug failed after {max_iterations} iterations")
        return False, nodes_data
    
    # 将 debug 方法添加到类
    setattr(PipelineExecutor, 'debug_pipeline', debug_pipeline)
    
    # 创建实例
    executor = PipelineExecutor(executor_llm_client, search_engine, meta_llm_client)
    return executor


def prepare_pipeline_input(dataset: str, example: dict) -> dict:
    """准备 pipeline 的输入数据"""
    if "j1eval" in dataset:
        # 对于 j1eval，去掉 id 和 court_information
        drop_keys = {"id", "court_information"}
        obj = copy.deepcopy(example)
        for k in drop_keys:
            obj.pop(k, None)
        return obj
    elif "HospSumm" in dataset or "hospsumm" in dataset or "hosp_summ" in dataset:
        # 对于 HospSumm，返回整个 example
        return example
    elif "travelplanner" in dataset:
        # 对于 travelplanner，返回整个 example
        return example
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")


def extract_pipeline_output(dataset: str, pipeline_output: Any) -> str:
    """从 pipeline 输出中提取最终答案文本"""
    if "j1eval" in dataset:
        # pipeline 输出是 drafted_judgment，可能是 JSON 字符串或字典
        # 先尝试解析 JSON 字符串
        if isinstance(pipeline_output, str):
            try:
                pipeline_output = json.loads(pipeline_output)
            except json.JSONDecodeError:
                return pipeline_output
        
        # 如果是字典，提取 court_judgment 字段
        if isinstance(pipeline_output, dict):
            court_judgment = pipeline_output.get('court_judgment', '')
            if isinstance(court_judgment, list):
                return '\n'.join(str(item) for item in court_judgment)
            return str(court_judgment) if court_judgment else ""
        return str(pipeline_output)
    elif "HospSumm" in dataset or "hospsumm" in dataset or "hosp_summ" in dataset:
        # 对于 HospSumm，可能需要从输出中提取
        if isinstance(pipeline_output, dict):
            return pipeline_output.get('answer', str(pipeline_output))
        return str(pipeline_output)
    elif "travelplanner" in dataset:
        # 对于 travelplanner，返回整个输出
        if isinstance(pipeline_output, dict):
            return json.dumps(pipeline_output, ensure_ascii=False)
        return str(pipeline_output)
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")


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


async def pipeline_single_example(
    example_id: int,
    example: dict,
    dataset: str,
    executor: Any,
    scorer: DataScorer,
    save_root: Path,
    executor_model: str,
    nodes_data: Dict[str, Any],
    max_debug_attempts: int = 3,
):
    """Run pipeline for a single example and evaluate. Also return approximate generation cost."""
    import asyncio
    
    # 准备 pipeline 输入
    pipeline_input = prepare_pipeline_input(dataset, example)
    
    # 每个样本允许最多 max_debug_attempts 次 debug 尝试
    success = False
    pipeline_output = None
    error_info = None
    
    for attempt in range(max_debug_attempts):
        try:
            # 在线程池中执行同步的 pipeline（因为 executor.execute_pipeline 是同步的）
            pipeline_output = await asyncio.to_thread(executor.execute_pipeline, pipeline_input)
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
                print(f"\n✗ Example {example_id} failed after {max_debug_attempts} attempts")
                print(f"Error: {error_type}: {error_msg}")
                break
            
            # 尝试 debug（使用 debug_pipeline 方法）
            print(f"\n⚠ Example {example_id} failed on attempt {attempt + 1}/{max_debug_attempts}")
            print(f"Error: {error_type}: {error_msg}")
            print(f"Attempting to debug and fix...")
            
            try:
                # 调用 debug_pipeline 来修复代码（在线程池中执行，因为它是同步的）
                debug_success, updated_nodes_data = await asyncio.to_thread(
                    executor.debug_pipeline,
                    pipeline_input,
                    nodes_data,
                    1  # 每次 debug 只尝试一次修复
                )
                
                if debug_success:
                    print(f"✓ Debug successful, retrying execution...")
                    # debug_pipeline 已经更新了 executor 的方法，继续下一次尝试
                    # 同时更新 nodes_data 以便后续使用
                    nodes_data.update(updated_nodes_data)
                    continue
                else:
                    print(f"✗ Debug failed, will retry with original code...")
                    # debug 失败，继续下一次尝试（使用原始代码）
                    continue
                    
            except Exception as debug_error:
                print(f"✗ Debug process itself failed: {debug_error}")
                # debug 过程本身出错，继续下一次尝试
                continue
    
    # 如果失败，设置错误输出
    if not success:
        pipeline_output = {"error": error_info}
    
    # 从 pipeline 输出中提取最终答案
    response_text = extract_pipeline_output(dataset, pipeline_output)
    
    # 获取 ground truth answer
    _, answer = build_question_and_answer(dataset, example)
    
    # 统计本次调用的成本（执行前后的差值）
    initial_cost = 0.0
    if hasattr(executor, 'llm_client') and hasattr(executor.llm_client, 'total_cost'):
        initial_cost = executor.llm_client.total_cost
    
    # 执行 pipeline 后获取最终成本
    final_cost = 0.0
    if hasattr(executor, 'llm_client') and hasattr(executor.llm_client, 'total_cost'):
        final_cost = executor.llm_client.total_cost
    
    call_cost = final_cost - initial_cost
    total_tokens = 0  # Token 统计需要从 LLMClient 中获取，暂时设为 0

    # Paths for logging (reuse DataScorer interface)
    # Structure: save_root/{dataset}/{example_id}/...
    example_dir = save_root / dataset / str(example_id)
    example_dir.mkdir(parents=True, exist_ok=True)

    judge_path = example_dir / f"{executor_model}_dag_judge.txt"
    response_path = example_dir / f"{executor_model}_dag_response.json"

    # 保存 pipeline 输出
    with open(response_path, 'w', encoding='utf-8') as f:
        json.dump({
            "pipeline_input": pipeline_input,
            "pipeline_output": pipeline_output,
            "extracted_response": response_text
        }, f, ensure_ascii=False, indent=2)

    # DataScorer.score expects some bookkeeping fields
    # 为了兼容，我们创建一个简单的 question（实际上 pipeline 不需要）
    question = json.dumps(pipeline_input, ensure_ascii=False)[:500]  # 截断以避免过长
    prompt_message = [dict(role="user", content=question)]
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
    executor_model = args.executor_model
    save_root = Path(args.save_dir)

    # 加载 generated_nodes.json
    print(f"Loading pipeline from: {args.nodes_json}")
    nodes_data = load_generated_nodes(args.nodes_json)
    print(f"Loaded {len(nodes_data.get('nodes', []))} nodes")
    print(f"Pipeline description: {nodes_data.get('pipeline_description', 'N/A')}")

    # 创建 Meta LLM 客户端（用于 debug）
    meta_llm_client = LLMClient(
        model=args.meta_model,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens
    )

    # 创建 Executor LLM 客户端
    executor_llm_client = LLMClient(
        model=executor_model,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens
    )

    # 创建搜索引擎
    search_engine = SearchEngineFactory.create_engine(
        "google",
        api_key=os.getenv("SERPER_API_KEY"),
    )
    llm_google = WebSearchLLM(
        llm_client=executor_llm_client,
        search_engine=search_engine,
        max_search_results=args.max_search_results,
        max_rounds=args.max_rounds,
        dataset_name=dataset,
        mode="test",
    )

    # 创建 pipeline executor
    print("\nCreating pipeline executor...")
    print(f"Meta-model: {args.meta_model} (for debug)")
    print(f"Executor-model: {executor_model} (for node execution)")
    executor = create_pipeline_executor(nodes_data, executor_llm_client, llm_google, meta_llm_client)
    print("Pipeline executor created successfully!")

    # Init scorer (uses oracle evaluators under the hood)
    scorer = DataScorer(dataset=dataset, technique="dag", mode_verifier=executor_model)

    examples = load_examples(dataset)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    # Optional subset selection
    selected_indices = set(args.given_examples) if args.given_examples else None

    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def run_with_semaphore(eid: int, ex: dict):
        async with semaphore:
            return await pipeline_single_example(
                example_id=eid,
                example=ex,
                dataset=dataset,
                executor=executor,
                scorer=scorer,
                save_root=save_root,
                executor_model=executor_model,
                nodes_data=nodes_data,
                max_debug_attempts=args.max_debug_attempts,
            )

    tasks = []
    for example_id, example in enumerate(examples):
        if selected_indices is not None and example_id not in selected_indices:
            continue
        tasks.append(run_with_semaphore(example_id, example))

    scores = []
    if tasks:
        for score, is_correct, call_cost, call_tokens in await tqdm_asyncio.gather(*tasks):
            scores.append(score)

    # Summary:统一按平均 score 计算
    if scores:
        # travelplanner / HospSumm 的 score 为 0/1，平均值即为准确率
        # j1eval 的 score 为 [0,1] 浮点数，平均值即为 JUD 平均分
        avg_score = round(sum(scores) / len(scores), 4)
    else:
        avg_score = 0.0

    # 获取最终的 accumulated cost
    final_accumulated_cost = 0.0
    if hasattr(executor, 'llm_client') and hasattr(executor.llm_client, 'total_cost'):
        final_accumulated_cost = executor.llm_client.total_cost

    print("=" * 80)
    print(f"Multi-Agent System (DAG) Results - Dataset: {dataset}, Executor Model: {executor_model}")
    print(f"Pipeline: {args.nodes_json}")
    print(f"Number of evaluated examples: {len(scores)}")
    print(f"Average score: {avg_score:.4f}")
    print(f"Total generation cost (accumulated): ${final_accumulated_cost:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    cli_args = parse_arguments()
    asyncio.run(main(cli_args))


