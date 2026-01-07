#!/usr/bin/env python3
"""
统计 PMC 实验结果的总成本和平均分数

用法:
    python statistics.py <results_dir>
    
例如:
    python statistics.py async_results
"""

import argparse
import re
from pathlib import Path
from typing import Tuple, Optional


def extract_cost(cost_file: Path) -> Optional[float]:
    """
    从 cost.txt 文件中提取总成本
    
    Args:
        cost_file: cost.txt 文件路径
        
    Returns:
        总成本（美元），如果文件不存在或无法解析则返回 None
    """
    if not cost_file.exists():
        return None
    
    try:
        with open(cost_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 匹配 "Total Cost: $X.XXXX" 格式
        match = re.search(r'Total Cost:\s*\$?([\d.]+)', content)
        if match:
            return float(match.group(1))
        else:
            print(f"Warning: Could not find 'Total Cost' in {cost_file}")
            return None
    except Exception as e:
        print(f"Error reading {cost_file}: {e}")
        return None


def extract_score(score_file: Path) -> Optional[float]:
    """
    从 score.txt 文件中提取最终分数
    
    Args:
        score_file: score.txt 文件路径
        
    Returns:
        最终分数，如果文件不存在或无法解析则返回 None
    """
    if not score_file.exists():
        return None
    
    try:
        with open(score_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 匹配 "Final Score: X" 格式
        match = re.search(r'Final Score:\s*([\d.]+)', content)
        if match:
            return float(match.group(1))
        else:
            print(f"Warning: Could not find 'Final Score' in {score_file}")
            return None
    except Exception as e:
        print(f"Error reading {score_file}: {e}")
        return None


def collect_task_data(results_dir: Path) -> Tuple[list, list]:
    """
    收集所有 task 的成本和分数数据
    
    Args:
        results_dir: 结果目录路径
        
    Returns:
        (costs, scores) 元组，每个都是列表
    """
    costs = []
    scores = []
    
    # 查找所有 task_X 目录
    task_dirs = sorted(results_dir.glob('task_*'), key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 0)
    
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
            
        task_id = task_dir.name
        cost_file = task_dir / 'cost.txt'
        score_file = task_dir / 'score.txt'
        
        # 提取成本
        cost = extract_cost(cost_file)
        if cost is not None:
            costs.append((task_id, cost))
        else:
            print(f"Warning: No cost data for {task_id}")
        
        # 提取分数
        score = extract_score(score_file)
        if score is not None:
            scores.append((task_id, score))
        else:
            print(f"Warning: No score data for {task_id}")
    
    return costs, scores


def print_statistics(costs: list, scores: list):
    """
    打印统计结果
    
    Args:
        costs: [(task_id, cost), ...] 列表
        scores: [(task_id, score), ...] 列表
    """
    print("=" * 80)
    print("PMC EXPERIMENT STATISTICS")
    print("=" * 80)
    print()
    
    # 成本统计
    if costs:
        total_cost = sum(cost for _, cost in costs)
        cost_values = [cost for _, cost in costs]
        avg_cost = sum(cost_values) / len(cost_values)
        min_cost = min(cost_values)
        max_cost = max(cost_values)
        
        print("COST STATISTICS")
        print("-" * 80)
        print(f"Number of tasks with cost data: {len(costs)}")
        print(f"Total Cost: ${total_cost:.4f}")
        print(f"Average Cost per task: ${avg_cost:.4f}")
        print(f"Min Cost: ${min_cost:.4f}")
        print(f"Max Cost: ${max_cost:.4f}")
        print()
        
        # 显示每个 task 的成本
        print("Cost breakdown by task:")
        for task_id, cost in costs:
            print(f"  {task_id}: ${cost:.4f}")
        print()
    else:
        print("COST STATISTICS")
        print("-" * 80)
        print("No cost data found.")
        print()
    
    # 分数统计
    if scores:
        score_values = [score for _, score in scores]
        avg_score = sum(score_values) / len(score_values)
        min_score = min(score_values)
        max_score = max(score_values)
        total_score = sum(score_values)
        
        print("SCORE STATISTICS")
        print("-" * 80)
        print(f"Number of tasks with score data: {len(scores)}")
        print(f"Average Score: {avg_score:.4f}")
        print(f"Total Score: {total_score:.4f}")
        print(f"Min Score: {min_score:.4f}")
        print(f"Max Score: {max_score:.4f}")
        print()
        
        # 显示每个 task 的分数
        print("Score breakdown by task:")
        for task_id, score in scores:
            print(f"  {task_id}: {score:.4f}")
        print()
    else:
        print("SCORE STATISTICS")
        print("-" * 80)
        print("No score data found.")
        print()
    
    # 综合统计
    print("SUMMARY")
    print("-" * 80)
    if costs and scores:
        # 找到同时有成本和分数的 task
        cost_dict = dict(costs)
        score_dict = dict(scores)
        common_tasks = set(cost_dict.keys()) & set(score_dict.keys())
        
        if common_tasks:
            print(f"Tasks with both cost and score data: {len(common_tasks)}")
            print(f"Total Cost (for tasks with scores): ${sum(cost_dict[t] for t in common_tasks):.4f}")
            print(f"Average Score: {sum(score_dict[t] for t in common_tasks) / len(common_tasks):.4f}")
        else:
            print("No tasks have both cost and score data.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='统计 PMC 实验结果的总成本和平均分数',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python statistics.py async_results
  python statistics.py /path/to/async_results
        """
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='/data/qin/lhh/Unified-MAS/PMC/async_results',
        help='结果目录路径（包含 task_X 子目录的目录）'
    )
    
    args = parser.parse_args()
    
    # 转换为 Path 对象
    results_dir = Path(args.results_dir)
    
    # 检查目录是否存在
    if not results_dir.exists():
        print(f"Error: Directory '{results_dir}' does not exist.")
        return 1
    
    if not results_dir.is_dir():
        print(f"Error: '{results_dir}' is not a directory.")
        return 1
    
    # 收集数据
    costs, scores = collect_task_data(results_dir)
    
    # 打印统计结果
    print_statistics(costs, scores)
    
    return 0


if __name__ == '__main__':
    exit(main())

