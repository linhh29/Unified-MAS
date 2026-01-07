# PMC Cost Tracking 使用说明

## 功能概述

PMC Agent 现在支持完整的 token 成本跟踪功能，包括：
- 输入 token 统计
- 输出 token 统计
- 总 token 统计
- 成本计算（基于模型定价）
- 详细的每次 API 调用记录

## 使用方法

### 1. 从 Plan 对象获取成本

```python
import asyncio
from pmc_agent import PMCAgent

async def main():
    pmc = PMCAgent(model_name="gpt-4", api_key="your_api_key")
    
    task = "Your task description..."
    constraints = {...}
    
    plan = await pmc.plan(task, constraints)
    
    # 从 Plan 对象直接访问成本信息
    print(f"Total Cost: ${plan.total_cost:.4f}")
    print(f"Total Input Tokens: {plan.total_input_tokens:,}")
    print(f"Total Output Tokens: {plan.total_output_tokens:,}")
    print(f"Total Tokens: {plan.total_tokens:,}")

asyncio.run(main())
```

### 2. 使用 get_total_cost() 方法获取详细信息

```python
# 获取完整的成本摘要
cost_summary = pmc.get_total_cost()

print(f"Total Cost: ${cost_summary['total_cost']:.4f}")
print(f"Total Input Tokens: {cost_summary['total_input_tokens']:,}")
print(f"Total Output Tokens: {cost_summary['total_output_tokens']:,}")
print(f"Total Tokens: {cost_summary['total_tokens']:,}")
print(f"Number of API Calls: {cost_summary['num_calls']}")

# 查看每次调用的详细信息
for i, call in enumerate(cost_summary['call_details'], 1):
    print(f"\nCall {i}:")
    print(f"  Agent: {call['agent_role']}")
    print(f"  Model: {call['model']}")
    print(f"  Input Tokens: {call['input_tokens']:,} (${call['input_cost']:.4f})")
    print(f"  Output Tokens: {call['output_tokens']:,} (${call['output_cost']:.4f})")
    print(f"  Total Tokens: {call['total_tokens']:,}")
    print(f"  Cost: ${call['total_cost']:.4f}")
```

### 3. 重置成本跟踪器

如果需要运行多个规划任务并分别跟踪成本：

```python
# 第一个任务
plan1 = await pmc.plan(task1, constraints1)
cost1 = pmc.get_total_cost()

# 重置跟踪器
pmc.reset_cost_tracker()

# 第二个任务
plan2 = await pmc.plan(task2, constraints2)
cost2 = pmc.get_total_cost()
```

## 支持的模型定价

当前实现支持以下 OpenAI 模型的定价：

- **GPT-4 系列**:
  - gpt-4: $0.03/1K input, $0.06/1K output
  - gpt-4-32k: $0.06/1K input, $0.12/1K output
  - gpt-4-turbo: $0.01/1K input, $0.03/1K output
  - gpt-4o: $0.005/1K input, $0.015/1K output

- **GPT-3.5 系列**:
  - gpt-3.5-turbo: $0.0015/1K input, $0.002/1K output
  - gpt-3.5-turbo-16k: $0.003/1K input, $0.004/1K output
  - gpt-3.5-turbo-0125: $0.0005/1K input, $0.0015/1K output

如果使用的模型不在列表中，将默认使用 gpt-4 的定价。

## 成本计算示例

```python
# 假设一次 API 调用使用了 1000 个输入 token 和 500 个输出 token
# 使用 gpt-4 模型：

# 输入成本 = (1000 / 1000) * 0.03 = $0.03
# 输出成本 = (500 / 1000) * 0.06 = $0.03
# 总成本 = $0.06
```

## 输出格式

`get_total_cost()` 返回的字典格式：

```python
{
    "total_input_tokens": 15000,
    "total_output_tokens": 5000,
    "total_tokens": 20000,
    "total_cost": 0.75,
    "num_calls": 10,
    "call_details": [
        {
            "model": "gpt-4",
            "agent_role": "manager",
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "input_cost": 0.03,
            "output_cost": 0.03,
            "total_cost": 0.06
        },
        # ... 更多调用记录
    ]
}
```

## 注意事项

1. 成本计算基于 OpenAI 的官方定价（2024年）
2. 如果模型名称不完全匹配，会尝试模糊匹配
3. 如果无法匹配模型，将使用 gpt-4 的默认定价
4. 所有成本以美元（USD）计算
5. token 数量来自 OpenAI API 响应，是准确的

## 更新定价

如果需要更新模型定价，可以修改 `CostTracker.PRICING` 字典：

```python
from pmc_agent import CostTracker

# 更新定价
CostTracker.PRICING["your-model-name"] = {
    "input": 0.01,  # 每 1K tokens
    "output": 0.02  # 每 1K tokens
}
```

