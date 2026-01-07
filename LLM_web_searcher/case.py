import json

# 读取 JSON 文件
json_file_path = "/data/qin/lhh/Unified-MAS/LLM_web_searcher/intermediate_result/j1eval/optimize/generated_nodes.json"

with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历每个 node 并打印 all_code
nodes = data.get('nodes', [])

for i, node in enumerate(nodes, 1):
    node_name = node.get('node_name', f'Node_{i}')
    all_code = node.get('all_code', '')
    
    print(f"\n{'='*80}")
    print(f"Node {i}: {node_name}")
    print(f"{'='*80}")
    print(all_code)
    print()

