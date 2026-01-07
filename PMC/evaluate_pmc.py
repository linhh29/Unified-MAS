import os
import re
from pathlib import Path


def evaluate_pmc(results_dir="/data/qin/lhh/Unified-MAS/PMC/async_results"):
    """
    读取async_results文件夹下所有任务的final_score和cost，然后相加得到最终结果
    
    Args:
        results_dir: async_results文件夹的路径
        
    Returns:
        tuple: (total_score, total_cost, task_count)
    """
    results_dir = Path(results_dir)
    total_score = 0
    total_cost = 0
    task_count = 0
    failed_tasks = []
    
    # 遍历所有task目录
    for task_dir in sorted(results_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith('task_'):
            continue
        
        task_count += 1
        task_name = task_dir.name
        
        # 读取score.txt文件
        score_file = task_dir / "score.txt"
        if score_file.exists():
            try:
                with open(score_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 使用正则表达式提取Final Score
                    match = re.search(r'Final Score:\s*([\d.]+)', content)
                    if match:
                        score = float(match.group(1))
                        total_score += score
                    else:
                        print(f"Warning: Could not find Final Score in {score_file}")
                        failed_tasks.append(f"{task_name}/score.txt")
            except Exception as e:
                print(f"Error reading {score_file}: {e}")
                failed_tasks.append(f"{task_name}/score.txt")
        else:
            print(f"Warning: {score_file} does not exist")
            failed_tasks.append(f"{task_name}/score.txt")
        
        # 读取cost.txt文件
        cost_file = task_dir / "cost.txt"
        if cost_file.exists():
            try:
                with open(cost_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 使用正则表达式提取Total Cost
                    match = re.search(r'Total Cost:\s*\$([\d.]+)', content)
                    if match:
                        cost = float(match.group(1))
                        total_cost += cost
                    else:
                        print(f"Warning: Could not find Total Cost in {cost_file}")
                        failed_tasks.append(f"{task_name}/cost.txt")
            except Exception as e:
                print(f"Error reading {cost_file}: {e}")
                failed_tasks.append(f"{task_name}/cost.txt")
        else:
            print(f"Warning: {cost_file} does not exist")
            failed_tasks.append(f"{task_name}/cost.txt")
    
    # 打印结果
    print("=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Tasks Processed: {task_count}")
    print(f"Total Score: {total_score}")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"Average Score per Task: {total_score / task_count if task_count > 0 else 0:.4f}")
    print(f"Average Cost per Task: ${total_cost / task_count if task_count > 0 else 0:.4f}")
    
    if failed_tasks:
        print(f"\nFailed to read {len(failed_tasks)} file(s):")
        for failed in failed_tasks[:10]:  # 只显示前10个
            print(f"  - {failed}")
        if len(failed_tasks) > 10:
            print(f"  ... and {len(failed_tasks) - 10} more")
    
    print("=" * 80)
    
    return total_score, total_cost, task_count


if __name__ == "__main__":
    evaluate_pmc()

