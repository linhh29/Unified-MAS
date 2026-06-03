"""
Debug 脚本：读取 nodes JSON，用对应数据集的第一个样本（build question 之后）跑通 pipeline，
并打印每个节点的输入输出。
"""
import json
import os

os.environ['OPENAI_API_KEY'] = 'xx'
os.environ['OPENAI_API_BASE'] = 'xx'
os.environ['SERPER_API_KEY'] = 'xx'

import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from Unified_MAS.llm_client import LLMClient
from Unified_MAS.search_engines import SearchEngineFactory
from Unified_MAS.web_search_llm import WebSearchLLM
from Unified_MAS.utils import build_question_and_answer, create_pipeline_executor


# 默认数据集路径映射
DEFAULT_DATASET_PATHS = {
    "deepfund": "xx/deepfund_validate.jsonl",
    "j1eval": "xx/j1eval_validate.jsonl",
    "healthbench": "xx/healthbench_validate.jsonl",
    "travelplanner": "xx/travelplanner_validate.jsonl",
    "hosp_summ": "xx/hosp_summ_validate.jsonl",
}


def extract_dataset_from_path(node_path: str) -> Optional[str]:
    """从 node path 中提取 dataset 名称，例如 intermediate_result/deepfund/optimize/rounds/xxx -> deepfund"""
    # 匹配 intermediate_result/<dataset>/...
    m = re.search(r"intermediate_result[/\\]([^/\\]+)", node_path)
    if m:
        return m.group(1)
    return None


def load_nodes(node_path: str) -> Dict[str, Any]:
    """加载 nodes JSON"""
    with open(node_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_first_sample(dataset_path: str) -> Dict[str, Any]:
    """加载数据集的第一个样本"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No valid sample found in {dataset_path}")


def wrap_node_with_io_print(executor, nodes_data: Dict[str, Any]) -> None:
    """包装 executor 的每个节点方法，在调用前后打印输入和输出"""
    for node in nodes_data.get("nodes", []):
        node_name = node.get("node_name")
        if not node_name:
            continue
        method = getattr(executor, node_name, None)
        if method is None or not callable(method):
            continue

        def make_wrapper(n: str, orig_method):
            def wrapped(self, input_data, *args, _n=n, _orig=orig_method, **kwargs):
                print(f"\n{'='*80}")
                print(f"[Node Input] {_n}")
                print(f"{'='*80}")
                if isinstance(input_data, (dict, list)):
                    print(json.dumps(input_data, ensure_ascii=False, indent=2))
                else:
                    print(str(input_data))
                print(f"{'='*80}\n")

                result = _orig(input_data, *args, **kwargs)

                print(f"\n{'='*80}")
                print(f"[Node Output] {_n}")
                print(f"{'='*80}")
                try:
                    if isinstance(result, (dict, list)):
                        s = json.dumps(result, ensure_ascii=False, indent=2)
                        print(s)
                    else:
                        print(str(result))
                except Exception:
                    print(repr(result))
                print(f"{'='*80}\n")

                return result

            return wrapped

        wrapped = make_wrapper(node_name, method)
        setattr(executor, node_name, wrapped.__get__(executor, type(executor)))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Debug pipeline: 读取 nodes JSON，用第一个样本跑通并打印每个节点的输入输出"
    )
    parser.add_argument(
        "--nodes_json",
        type=str,
        default="xx/deepfund/search/generated_nodes.json",
        help="Nodes JSON 文件路径（如 intermediate_result/deepfund/optimize/rounds/epoch_10_generated_nodes.json）",
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default=None,
        help="数据集 JSONL 路径（可选，不提供则根据 nodes 路径推断 dataset 并使用默认路径）",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="数据集名称（可选，不提供则从 nodes 路径推断）",
    )
    parser.add_argument(
        "--executor_model",
        type=str,
        default="gemini-3-flash-preview",
        help="Executor 模型",
    )
    parser.add_argument(
        "--meta_model",
        type=str,
        default="gemini-3-pro-preview",
        help="Meta 模型（debug 用，当前脚本主要跑 executor）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Temperature",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=8192,
        help="Max completion tokens",
    )
    args = parser.parse_args()

    node_path = args.nodes_json
    if not os.path.exists(node_path):
        print(f"Error: nodes file not found: {node_path}")
        sys.exit(1)

    # 推断 dataset 名称
    dataset_name = args.dataset_name or extract_dataset_from_path(node_path)
    if not dataset_name:
        print("Error: Cannot infer dataset name from path. Please provide --dataset_name.")
        sys.exit(1)

    # 确定数据集路径
    input_data_path = args.input_data or DEFAULT_DATASET_PATHS.get(dataset_name)
    if not input_data_path or not os.path.exists(input_data_path):
        print(f"Error: Input data not found: {input_data_path or 'N/A'}")
        print(f"  Dataset: {dataset_name}, please provide --input_data")
        sys.exit(1)

    print(f"Loading nodes from: {node_path}")
    nodes_data = load_nodes(node_path)
    print(f"Loaded {len(nodes_data.get('nodes', []))} nodes")
    print(f"Pipeline: {nodes_data.get('pipeline_description', 'N/A')}")

    print(f"\nLoading first sample from: {input_data_path}")
    first_sample = load_first_sample(input_data_path)
    question, answer = build_question_and_answer(dataset_name, first_sample)

    print(f"\n{'#'*80}")
    print("[Build Question] (first sample, after build_question_and_answer)")
    print(f"{'#'*80}")
    print(question[:2000])
    if len(question) > 2000:
        print("... (truncated)")
    print(f"\n[Expected Answer]: {answer[:500]}{'...' if len(str(answer)) > 500 else ''}")
    print(f"{'#'*80}\n")

    # 创建 LLM 客户端和搜索引擎
    meta_llm_client = LLMClient(
        model=args.meta_model,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )
    executor_llm_client = LLMClient(
        model=args.executor_model,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )
    search_engine = SearchEngineFactory.create_engine(
        "google",
        api_key=os.getenv("SERPER_API_KEY"),
    )
    llm_google = WebSearchLLM(
        llm_client=executor_llm_client,
        search_engine=search_engine,
        max_search_results=10,
        max_rounds=1,
        dataset_name=dataset_name,
        mode="test",
    )

    print("Creating pipeline executor...")
    executor = create_pipeline_executor(
        nodes_data,
        executor_llm_client,
        llm_google,
        meta_llm_client,
        dataset_name,
        num_epochs=1,
    )

    # 包装节点方法以打印输入输出
    wrap_node_with_io_print(executor, nodes_data)

    print("\n" + "=" * 80)
    print("Executing pipeline with first sample...")
    print("=" * 80)

    try:
        result, buffer_entry = executor.execute_pipeline(question, answer, sample_index=0)

        print("\n" + "=" * 80)
        print("[Final Result]")
        print("=" * 80)
        if isinstance(result, (dict, list)):
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result)
        print("=" * 80)

        # 汇总每个节点的输出（已在运行中打印，这里再列一遍简要信息）
        print("\n[Summary] Intermediate outputs:")
        for i, item in enumerate(buffer_entry.get("intermediate_outputs", [])):
            name = item.get("node_name", "?")
            out = item.get("output", "")
            out_preview = out
            print(f"  {i+1}. {name}: {out_preview}")

    except Exception as e:
        import traceback

        print(f"\nError during execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
