#!/bin/bash
set -e
export OPENAI_API_KEY=sk-BDrpp8zrYLtMWyfY2YZJZZPjIOXwikCyZFfDWL8eUGDqnts2
export OPENAI_API_BASE=https://api.qingyuntop.top/v1

# gpt-5-mini, gemini-3-flash-preview, deepseek-v3.2, qwen3-30b-a3b-instruct-2507
MODEL=gemini-3-flash-preview

# python /root/J1Bench/src/Eval/bench/KQ/KQ.py
# python /root/J1Bench/src/Eval/bench/LC/LC.py
# python /root/J1Bench/src/Eval/bench/CD/CD.py
# python /root/J1Bench/src/Eval/bench/DD/DD.py
python /data/qin/lhh/Unified-MAS/J1Bench/src/Eval/bench/CI/CI.py --model ${MODEL}
# python /root/J1Bench/src/Eval/bench/CR/CR.py