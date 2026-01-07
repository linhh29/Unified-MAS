
import re

total_cost = []
with open('/data/qin/lhh/Unified-MAS/J1Bench/src/cost_gemini-3-flash-preview.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('---'):
            # 匹配 case_XXX: cost 格式
            match = re.search(r':\s*([\d.]+)', line)
            if match:
                cost = float(match.group(1))
                total_cost.append(cost)

print(len(total_cost))
print(f'Total cost: {sum(total_cost):.6f}')