#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本用于将 JSON 数组文件转换为 JSONL 格式，并随机划分数据集
每行一个 JSON 对象
"""

import json
import random
import os
from pathlib import Path

def json_to_jsonl_and_split(
    input_file: str,
    output_dir: Path,
    validation_size: int = 32,
    test_size: int = 168,
    seed: int = 42
):
    """
    将 JSON 数组文件转换为 JSONL 格式，并随机划分为 validation 和 test 集
    
    Args:
        input_file: 输入的 JSON 文件路径（包含 JSON 数组）
        output_dir: 输出目录路径
        validation_size: validation set 的大小
        test_size: test set 的大小
        seed: 随机种子
    """
    print(f"正在读取文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)
    
    if not isinstance(data, list):
        raise ValueError(f"输入文件应该包含一个 JSON 数组，但得到的是 {type(data)}")
    
    total = len(data)
    print(f"找到 {total} 个对象")
    
    if total < validation_size + test_size:
        raise ValueError(
            f"数据量不足: 总共有 {total} 个对象，但需要 {validation_size + test_size} 个 "
            f"(validation: {validation_size}, test: {test_size})"
        )
    
    # 设置随机种子以确保可重复性
    random.seed(seed)
    # 打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 划分数据集
    validation_data = shuffled_data[:validation_size]
    test_data = shuffled_data[validation_size:validation_size + test_size]
    discarded_count = total - validation_size - test_size
    
    print(f"\n数据集划分:")
    print(f"  Validation set: {len(validation_data)} 个对象")
    print(f"  Test set: {len(test_data)} 个对象")
    print(f"  丢弃: {discarded_count} 个对象")
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 写入 validation set
    validation_file = output_dir / "hospitalization_summarization_validate.jsonl"
    print(f"\n正在写入 validation set: {validation_file}")
    with open(validation_file, 'w', encoding='utf-8') as f_out:
        for item in validation_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f_out.write(json_line + '\n')
    print(f"成功写入 {len(validation_data)} 个对象到 {validation_file}")
    
    # 写入 test set
    test_file = output_dir / "hospitalization_summarization_test.jsonl"
    print(f"\n正在写入 test set: {test_file}")
    with open(test_file, 'w', encoding='utf-8') as f_out:
        for item in test_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f_out.write(json_line + '\n')
    print(f"成功写入 {len(test_data)} 个对象到 {test_file}")
    
    return validation_file, test_file

def main():
    # 定义输入和输出文件路径
    base_dir = Path("/data/qin/lhh/Unified-MAS/MAS-Zero/data")
    src_dir = base_dir / "src"
    input_file = base_dir / "Hospitalization-Summarization.json"
    
    if not input_file.exists():
        print(f"错误: 输入文件 {input_file} 不存在")
        return
    
    # 执行转换和划分
    validation_file, test_file = json_to_jsonl_and_split(
        input_file=str(input_file),
        output_dir=src_dir,
        validation_size=32,
        test_size=168,
        seed=0
    )
    
    print("\n转换和划分完成！")
    print(f"Validation set: {validation_file}")
    print(f"Test set: {test_file}")

if __name__ == "__main__":
    main()

