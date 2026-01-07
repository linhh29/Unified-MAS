#!/usr/bin/env python3
"""
统计 AgentClinic 结果目录中所有 scenario 的 cost
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List


def load_json(json_file: str) -> Dict:
    """加载 JSON 文件"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


def calculate_statistics(results_dir: str) -> Dict:
    """计算统计信息"""
    results_dir = Path(results_dir)
    
    # 初始化统计变量
    total_doctor_cost = 0.0
    total_patient_cost = 0.0
    total_measurement_cost = 0.0
    total_moderator_cost = 0.0
    total_all_cost = 0.0
    
    total_doctor_tokens = 0
    total_patient_tokens = 0
    total_measurement_tokens = 0
    total_moderator_tokens = 0
    total_all_tokens = 0
    
    total_doctor_calls = 0
    total_patient_calls = 0
    total_measurement_calls = 0
    total_moderator_calls = 0
    
    # 诊断结果统计
    total_scenarios = 0
    correct_diagnoses = 0
    incorrect_diagnoses = 0
    no_diagnosis = 0
    
    # 存储每个 scenario 的详细信息
    scenario_details = []
    
    # 遍历所有 scenario 目录
    scenario_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('scenario_')])
    
    for scenario_dir in scenario_dirs:
        cost_file = scenario_dir / 'cost.json'
        result_file = scenario_dir / 'result.json'
        
        if not cost_file.exists():
            print(f"Warning: {cost_file} not found, skipping...")
            continue
        
        cost_data = load_json(cost_file)
        if cost_data is None:
            continue
        
        scenario_id = cost_data.get('scenario_id', scenario_dir.name)
        total_scenarios += 1
        
        # 累加各 agent 的 cost
        doctor_cost = cost_data.get('doctor', {}).get('total_cost', 0.0)
        patient_cost = cost_data.get('patient', {}).get('total_cost', 0.0)
        measurement_cost = cost_data.get('measurement', {}).get('total_cost', 0.0)
        moderator_cost = cost_data.get('moderator', {}).get('total_cost', 0.0)
        
        total_doctor_cost += doctor_cost
        total_patient_cost += patient_cost
        total_measurement_cost += measurement_cost
        total_moderator_cost += moderator_cost
        
        # 累加 tokens
        doctor_tokens = cost_data.get('doctor', {}).get('total_tokens', 0)
        patient_tokens = cost_data.get('patient', {}).get('total_tokens', 0)
        measurement_tokens = cost_data.get('measurement', {}).get('total_tokens', 0)
        moderator_tokens = cost_data.get('moderator', {}).get('total_tokens', 0)
        
        total_doctor_tokens += doctor_tokens
        total_patient_tokens += patient_tokens
        total_measurement_tokens += measurement_tokens
        total_moderator_tokens += moderator_tokens
        
        # 累加调用次数
        total_doctor_calls += cost_data.get('doctor', {}).get('num_calls', 0)
        total_patient_calls += cost_data.get('patient', {}).get('num_calls', 0)
        total_measurement_calls += cost_data.get('measurement', {}).get('num_calls', 0)
        total_moderator_calls += cost_data.get('moderator', {}).get('num_calls', 0)
        
        # 从 result.json 读取诊断结果统计
        is_correct = False
        diagnosis_made = False
        if result_file.exists():
            result_data = load_json(result_file)
            if result_data:
                is_correct = result_data.get('is_correct', False)
                diagnosis_made = result_data.get('diagnosis_made', False)
        
        if diagnosis_made:
            if is_correct:
                correct_diagnoses += 1
            else:
                incorrect_diagnoses += 1
        else:
            no_diagnosis += 1
        
        # 记录 scenario 详情 - 使用 cost.json 中的 total.total_cost
        scenario_total_cost = cost_data.get('total', {}).get('total_cost', 0.0)
        total_all_cost += scenario_total_cost
        total_all_tokens += cost_data.get('total', {}).get('total_tokens', 0)
        
        scenario_details.append({
            'scenario_id': scenario_id,
            'total_cost': scenario_total_cost,
            'doctor_cost': doctor_cost,
            'patient_cost': patient_cost,
            'measurement_cost': measurement_cost,
            'moderator_cost': moderator_cost,
            'is_correct': is_correct,
            'diagnosis_made': diagnosis_made
        })
    
    # 计算平均值
    avg_cost_per_scenario = total_all_cost / total_scenarios if total_scenarios > 0 else 0.0
    accuracy = (correct_diagnoses / total_scenarios * 100) if total_scenarios > 0 else 0.0
    
    return {
        'total_scenarios': total_scenarios,
        'costs': {
            'doctor': {
                'total_cost': total_doctor_cost,
                'total_tokens': total_doctor_tokens,
                'num_calls': total_doctor_calls,
                'avg_cost_per_scenario': total_doctor_cost / total_scenarios if total_scenarios > 0 else 0.0
            },
            'patient': {
                'total_cost': total_patient_cost,
                'total_tokens': total_patient_tokens,
                'num_calls': total_patient_calls,
                'avg_cost_per_scenario': total_patient_cost / total_scenarios if total_scenarios > 0 else 0.0
            },
            'measurement': {
                'total_cost': total_measurement_cost,
                'total_tokens': total_measurement_tokens,
                'num_calls': total_measurement_calls,
                'avg_cost_per_scenario': total_measurement_cost / total_scenarios if total_scenarios > 0 else 0.0
            },
            'moderator': {
                'total_cost': total_moderator_cost,
                'total_tokens': total_moderator_tokens,
                'num_calls': total_moderator_calls,
                'avg_cost_per_scenario': total_moderator_cost / total_scenarios if total_scenarios > 0 else 0.0
            },
            'total': {
                'total_cost': total_all_cost,
                'total_tokens': total_all_tokens,
                'avg_cost_per_scenario': avg_cost_per_scenario
            }
        },
        'diagnosis_results': {
            'correct': correct_diagnoses,
            'incorrect': incorrect_diagnoses,
            'no_diagnosis': no_diagnosis,
            'accuracy': accuracy
        },
        'scenario_details': scenario_details
    }


def print_statistics(stats: Dict):
    """打印统计结果"""
    print("=" * 80)
    print("Cost Statistics Summary")
    print("=" * 80)
    
    print(f"\nTotal Scenarios: {stats['total_scenarios']}")
    
    # 总 cost 和正确率（最重要的信息）
    total_stats = stats['costs']['total']
    diag_results = stats['diagnosis_results']
    
    print("\n" + "-" * 80)
    print("Total Cost Summary:")
    print("-" * 80)
    print(f"  Total Cost: ${total_stats['total_cost']:.6f}")
    print(f"  Total Tokens: {total_stats['total_tokens']:,}")
    print(f"  Average Cost per Scenario: ${total_stats['avg_cost_per_scenario']:.6f}")
    
    print("\n" + "-" * 80)
    print("Diagnosis Results:")
    print("-" * 80)
    print(f"  Correct Diagnoses: {diag_results['correct']}")
    print(f"  Incorrect Diagnoses: {diag_results['incorrect']}")
    print(f"  No Diagnosis: {diag_results['no_diagnosis']}")
    print(f"  Accuracy: {diag_results['accuracy']:.2f}%")
    
    print("\n" + "-" * 80)
    print("Cost Breakdown by Agent Type:")
    print("-" * 80)
    
    costs = stats['costs']
    for agent_type in ['doctor', 'patient', 'measurement', 'moderator']:
        agent_stats = costs[agent_type]
        print(f"\n{agent_type.capitalize()} Agent:")
        print(f"  Total Cost: ${agent_stats['total_cost']:.6f}")
        print(f"  Total Tokens: {agent_stats['total_tokens']:,}")
        print(f"  Number of Calls: {agent_stats['num_calls']}")
        print(f"  Average Cost per Scenario: ${agent_stats['avg_cost_per_scenario']:.6f}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='统计 AgentClinic 结果目录中的 cost')
    parser.add_argument('--results_dir', default='./async_results_gemini-3-flash-preview', type=str, help='结果目录路径（包含 scenario_X 子目录）')
    parser.add_argument('--verbose', default=False, type=bool, help='显示每个 scenario 的详细信息')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Directory {args.results_dir} does not exist")
        return
    
    # 计算统计信息
    stats = calculate_statistics(args.results_dir)
    
    # 打印统计结果
    print_statistics(stats)
    
    # 如果指定了详细模式，打印每个 scenario 的详情
    if args.verbose:
        print("\n" + "=" * 80)
        print("Scenario Details:")
        print("=" * 80)
        for detail in stats['scenario_details']:
            status = "CORRECT" if detail['is_correct'] else ("INCORRECT" if detail['diagnosis_made'] else "NO DIAGNOSIS")
            print(f"\nScenario {detail['scenario_id']}: {status}")
            print(f"  Total Cost: ${detail['total_cost']:.6f}")
            print(f"    - Doctor: ${detail['doctor_cost']:.6f}")
            print(f"    - Patient: ${detail['patient_cost']:.6f}")
            print(f"    - Measurement: ${detail['measurement_cost']:.6f}")
            print(f"    - Moderator: ${detail['moderator_cost']:.6f}")
    



if __name__ == '__main__':
    main()

