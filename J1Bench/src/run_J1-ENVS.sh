#!/bin/bash
set -e
# export OPENAI_API_KEY=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
export OPENAI_API_KEY=sk-BDrpp8zrYLtMWyfY2YZJZZPjIOXwikCyZFfDWL8eUGDqnts2
export OPENAI_API_BASE=https://api.qingyuntop.top/v1

# gpt-5-mini, gemini-3-flash-preview, deepseek-v3.2, qwen3-30b-a3b-instruct-2507
MODEL=gemini-3-flash-preview

# python run.py --scenario J1Bench.Scenario.KQ --trainee Agent.Trainee.ConsultGPT --save_path "/root/J1Bench/src/data/dialog_history/GPT/KQ_dialog_history.jsonl"
# python run.py --scenario J1Bench.Scenario.LC --trainee Agent.Trainee.LC_GPT --save_path "/root/J1Bench/src/data/dialog_history/GPT/LC_dialog_history.jsonl"
# python run.py --scenario J1Bench.Scenario.CD --lawyer Agent.Lawyer.GPT_CD --save_path "/root/J1Bench/src/data/dialog_history/GPT/CD_dialog_history.jsonl"
# python run.py --scenario J1Bench.Scenario.DD --lawyer Agent.Lawyer.GPT_DD --save_path "/root/J1Bench/src/data/dialog_history/GPT/DD_dialog_history.jsonl"
python run.py --scenario J1Bench.Scenario.CI --judge Agent.Judge.GPT_CI --save_path ./templates/CI_dialog_history_${MODEL}.jsonl --model ${MODEL}
# python run.py --scenario J1Bench.Scenario.CR --judge Agent.Judge.GPT_CR --save_path "/root/J1Bench/src/data/dialog_history/GPT/CR_dialog_history.jsonl"