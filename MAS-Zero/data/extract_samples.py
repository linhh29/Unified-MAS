#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本用于从三个jsonl文件中提取前16个样本并保存为新的jsonl文件
"""

import json
import os

def extract_first_n_samples(input_file, output_file, n=16):
    """
    从输入的jsonl文件中提取前n个样本并保存到输出文件
    
    Args:
        input_file: 输入的jsonl文件路径
        output_file: 输出的jsonl文件路径
        n: 要提取的样本数量，默认16
    """
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if line:  # 跳过空行
                f_out.write(line + '\n')
                count += 1
                if count >= n:
                    break
    print(f"从 {input_file} 提取了 {count} 个样本，保存到 {output_file}")

def main():
    # 定义输入和输出文件路径
    base_dir = "/data/qin/lhh/Unified-MAS/MAS-Zero/data/src"
    output_dir = "/data/qin/lhh/Unified-MAS/MAS-Zero/data/src"
    
    files_config = [
        {
            "input": os.path.join(base_dir, "hospitalization_summarization_test.jsonl"),
            "output": os.path.join(output_dir, "hospitalization_summarization_test_16.jsonl")
        },
        {
            "input": os.path.join(base_dir, "j1eval_test.jsonl"),
            "output": os.path.join(output_dir, "j1eval_test_16.jsonl")
        },
        {
            "input": os.path.join(base_dir, "travelplanner_test.jsonl"),
            "output": os.path.join(output_dir, "travelplanner_test_16.jsonl")
        }
    ]
    
    # 处理每个文件
    for config in files_config:
        input_file = config["input"]
        output_file = config["output"]
        
        if os.path.exists(input_file):
            extract_first_n_samples(input_file, output_file, n=16)
        else:
            print(f"警告: 文件 {input_file} 不存在，跳过")

if __name__ == "__main__":
    main()

