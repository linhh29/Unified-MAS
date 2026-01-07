# 输出文件格式说明

## 概述
当使用 `--output_dir` 参数时，AgentClinic 会将所有中间结果、最终结果和成本信息保存到指定目录。

## 目录结构

```
output_dir/
├── summary.json                    # 所有场景的汇总信息
├── scenario_0/                     # 场景 0 的结果
│   ├── dialogue.json               # 对话历史
│   ├── result.json                 # 最终结果（诊断是否正确）
│   └── cost.json                   # 成本信息
├── scenario_1/                     # 场景 1 的结果
│   ├── dialogue.json
│   ├── result.json
│   └── cost.json
└── ...
```

## 文件格式

### 1. summary.json - 汇总信息

包含所有场景的汇总统计信息：

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "config": {
    "dataset": "MedQA",
    "doctor_llm": "gpt4o",
    "patient_llm": "gpt4o",
    "measurement_llm": "gpt4o",
    "moderator_llm": "gpt4o",
    "doctor_bias": "None",
    "patient_bias": "None",
    "total_inferences": 20,
    "num_scenarios": 10,
    "max_concurrent": 50
  },
  "results": {
    "total_scenarios": 10,
    "correct_diagnoses": 8,
    "accuracy": 80.0
  },
  "costs": {
    "doctor": {
      "total_cost": 0.123456,
      "total_tokens": 12345,
      "average_cost_per_scenario": 0.012346
    },
    "patient": {
      "total_cost": 0.234567,
      "total_tokens": 23456,
      "average_cost_per_scenario": 0.023457
    },
    "measurement": {
      "total_cost": 0.045678,
      "total_tokens": 4567,
      "average_cost_per_scenario": 0.004568
    },
    "moderator": {
      "total_cost": 0.078901,
      "total_tokens": 7890,
      "average_cost_per_scenario": 0.007890
    },
    "total": {
      "total_cost": 0.482602,
      "total_tokens": 48258,
      "average_cost_per_scenario": 0.048260
    }
  },
  "scenario_results": [
    {
      "scenario_id": 0,
      "correct": true,
      "cost_summary": { ... }
    },
    ...
  ]
}
```

### 2. scenario_X/dialogue.json - 对话历史

记录每个场景的完整对话过程：

```json
{
  "scenario_id": 0,
  "total_rounds": 5,
  "dialogue_history": [
    {
      "round": 1,
      "type": "doctor",
      "content": "Can you tell me about your symptoms?",
      "progress": 5
    },
    {
      "round": 1,
      "type": "patient",
      "content": "I've been experiencing double vision for about a month...",
      "progress": 5
    },
    {
      "round": 2,
      "type": "doctor",
      "content": "Do these symptoms get worse with activity?",
      "progress": 10
    },
    {
      "round": 2,
      "type": "patient",
      "content": "Yes, they seem to worsen when I'm active...",
      "progress": 10
    },
    {
      "round": 3,
      "type": "doctor",
      "content": "REQUEST TEST: Acetylcholine_Receptor_Antibodies",
      "progress": 15
    },
    {
      "round": 3,
      "type": "measurement",
      "content": "RESULTS: Present (elevated)",
      "progress": 15
    },
    {
      "round": 4,
      "type": "doctor",
      "content": "DIAGNOSIS READY: Myasthenia gravis",
      "progress": 20
    }
  ]
}
```

### 3. scenario_X/result.json - 最终结果

记录诊断结果和场景信息：

```json
{
  "scenario_id": 0,
  "diagnosis_made": true,
  "doctor_diagnosis": "Myasthenia gravis",
  "correct_diagnosis": "Myasthenia gravis",
  "is_correct": true,
  "scenario_data": {
    "OSCE_Examination": {
      "Objective_for_Doctor": "...",
      "Patient_Actor": { ... },
      "Physical_Examination_Findings": { ... },
      "Test_Results": { ... },
      "Correct_Diagnosis": "Myasthenia gravis"
    }
  }
}
```

### 4. scenario_X/cost.json - 成本信息

记录详细的成本统计：

```json
{
  "scenario_id": 0,
  "doctor": {
    "input_tokens": 1234,
    "output_tokens": 567,
    "total_tokens": 1801,
    "input_cost": 0.003085,
    "output_cost": 0.00567,
    "total_cost": 0.008755,
    "num_calls": 5
  },
  "patient": {
    "input_tokens": 2345,
    "output_tokens": 890,
    "total_tokens": 3235,
    "input_cost": 0.005863,
    "output_cost": 0.0089,
    "total_cost": 0.014763,
    "num_calls": 5
  },
  "measurement": {
    "input_tokens": 456,
    "output_tokens": 123,
    "total_tokens": 579,
    "input_cost": 0.00114,
    "output_cost": 0.00123,
    "total_cost": 0.00237,
    "num_calls": 2
  },
  "moderator": {
    "input_tokens": 789,
    "output_tokens": 45,
    "total_tokens": 834,
    "input_cost": 0.001973,
    "output_cost": 0.00045,
    "total_cost": 0.002423,
    "num_calls": 1
  },
  "total": {
    "total_cost": 0.028311,
    "total_tokens": 6449
  },
  "detailed_calls": {
    "doctor": [
      {
        "model": "gpt-4o",
        "input_tokens": 250,
        "output_tokens": 120,
        "cost": 0.001825,
        "inference_num": 1
      },
      ...
    ],
    "patient": [
      {
        "model": "gpt-4o",
        "input_tokens": 300,
        "output_tokens": 150,
        "cost": 0.00225
      },
      ...
    ],
    "measurement": [
      {
        "model": "gpt-4o",
        "input_tokens": 200,
        "output_tokens": 50,
        "cost": 0.001
      },
      ...
    ],
    "moderator": [
      {
        "model": "gpt-4o",
        "input_tokens": 789,
        "output_tokens": 45,
        "cost": 0.002423
      }
    ]
  }
}
```

## 使用方式

### 基本使用
```bash
python3 agentclinic.py \
  --openai_api_key "YOUR_KEY" \
  --output_dir "./results/run_1" \
  --num_scenarios 10
```

### 完整示例
```bash
python3 agentclinic.py \
  --openai_api_key "YOUR_KEY" \
  --doctor_llm gpt4o \
  --patient_llm gpt4o \
  --agent_dataset MedQA_Ext \
  --num_scenarios 32 \
  --max_concurrent 20 \
  --output_dir "./results/medqa_gpt4o_32scenarios"
```

## 文件说明

### dialogue.json
- **用途**: 记录完整的对话历史
- **内容**: 每一轮的医生提问、患者回答、检查结果
- **格式**: JSON 数组，按时间顺序排列
- **字段**:
  - `round`: 回合数
  - `type`: 对话类型（doctor/patient/measurement）
  - `content`: 对话内容
  - `progress`: 进度百分比

### result.json
- **用途**: 记录诊断结果
- **内容**: 医生诊断、正确答案、是否正确、场景原始数据
- **字段**:
  - `diagnosis_made`: 是否做出诊断
  - `doctor_diagnosis`: 医生的诊断
  - `correct_diagnosis`: 正确答案
  - `is_correct`: 诊断是否正确
  - `scenario_data`: 场景的原始数据

### cost.json
- **用途**: 记录详细的成本信息
- **内容**: 每个 agent 的 token 使用量、成本、详细调用记录
- **字段**:
  - 每个 agent（doctor/patient/measurement/moderator）的成本统计
  - `detailed_calls`: 每次 API 调用的详细信息
  - `total`: 总成本和总 token 数

### summary.json
- **用途**: 所有场景的汇总信息
- **内容**: 配置、结果统计、成本汇总、每个场景的结果
- **字段**:
  - `timestamp`: 运行时间戳
  - `config`: 运行配置
  - `results`: 结果统计
  - `costs`: 成本汇总（按 agent 分类）
  - `scenario_results`: 每个场景的简要结果

## 数据分析

### 分析单个场景
```python
import json
from pathlib import Path

# 读取场景 0 的结果
scenario_dir = Path("output_dir/scenario_0")
dialogue = json.load(open(scenario_dir / "dialogue.json"))
result = json.load(open(scenario_dir / "result.json"))
cost = json.load(open(scenario_dir / "cost.json"))

print(f"Scenario {result['scenario_id']}: {result['is_correct']}")
print(f"Total cost: ${cost['total']['total_cost']:.6f}")
print(f"Total rounds: {dialogue['total_rounds']}")
```

### 分析所有场景
```python
import json
from pathlib import Path

# 读取汇总信息
summary = json.load(open("output_dir/summary.json"))

print(f"Accuracy: {summary['results']['accuracy']}%")
print(f"Total cost: ${summary['costs']['total']['total_cost']:.6f}")

# 分析每个 agent 的成本
for agent in ['doctor', 'patient', 'measurement', 'moderator']:
    cost_info = summary['costs'][agent]
    print(f"{agent}: ${cost_info['total_cost']:.6f} ({cost_info['total_tokens']} tokens)")
```

## 注意事项

1. **文件编码**: 所有文件使用 UTF-8 编码
2. **JSON 格式**: 使用缩进格式，便于阅读
3. **并发安全**: 每个场景的文件独立保存，并发环境下安全
4. **文件大小**: 大量场景可能产生较大的汇总文件
5. **目录创建**: 如果输出目录不存在，会自动创建

## 示例输出目录

运行 10 个场景后的目录结构：
```
results/run_1/
├── summary.json
├── scenario_0/
│   ├── dialogue.json
│   ├── result.json
│   └── cost.json
├── scenario_1/
│   ├── dialogue.json
│   ├── result.json
│   └── cost.json
...
└── scenario_9/
    ├── dialogue.json
    ├── result.json
    └── cost.json
```

