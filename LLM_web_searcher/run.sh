export OPENAI_API_KEY=sk-BDrpp8zrYLtMWyfY2YZJZZPjIOXwikCyZFfDWL8eUGDqnts2
export OPENAI_API_BASE=https://api.qingyuntop.top/v1

export SERPER_API_KEY=65c5db3fbf945a70559117070f9587f487f3469a
export GITHUB_TOKEN=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

# 使用默认参数运行
META_MODEL=gemini-3-pro-preview

# gemini-3-flash-preview, gpt-5-mini, deepseek-v3.2, qwen3-30b-a3b-instruct-2507
EXECUTOR_MODEL=qwen3-30b-a3b-instruct-2507

# j1eval, hosp_summ, travelplanner
DATANAME=travelplanner
DATASET=/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/${DATANAME}_validate.jsonl

# 使用自定义参数运行（示例）
python run_search.py \
    --model ${META_MODEL} \
    --temperature 1 \
    --max_completion_tokens 8192 \
    --data_path ${DATASET} \
    --max_search_results 10 \
    --max_rounds 10 \
    --max_concurrent 5

python run_optimize.py \
    --nodes_json /data/qin/lhh/Unified-MAS/LLM_web_searcher/intermediate_result/${DATANAME}/search/generated_nodes.json \
    --input_data ${DATASET} \
    --meta_model ${META_MODEL} \
    --executor_model ${EXECUTOR_MODEL} \
    --temperature 1 \
    --max_completion_tokens 8192 \
    --dataset_name ${DATANAME} \
    --max_search_results 10 \
    --max_rounds 3 \
    --max_debug_attempts 3 