export OPENAI_API_KEY=sk-BDrpp8zrYLtMWyfY2YZJZZPjIOXwikCyZFfDWL8eUGDqnts2
export OPENAI_API_BASE=https://api.qingyuntop.top/v1
export SERPER_API_KEY=65c5db3fbf945a70559117070f9587f487f3469a


# travelplanner, hosp_summ, j1eval
DATASET=j1eval
# gpt-5-mini, gemini-3-flash-preview, deepseek-v3.2, qwen3-30b-a3b-instruct-2507
MODEL=gemini-3-flash-preview

META_MODEL=gemini-3-pro-preview

python async_dag_main_question.py \
    --dataset ${DATASET} \
    --nodes_json /data/qin/lhh/Unified-MAS/LLM_web_searcher/intermediate_result/${DATASET}/optimize/generated_nodes.json \
    --executor_model ${MODEL} \
    --meta_model ${META_MODEL} \
    --save_dir ./dag_results_${MODEL}/ \
    --temperature 1 \
    --max_completion_tokens 8192 \
    --max_search_results 10 \
    --max_rounds 3 \
    --max_concurrent 50 \
    --max_debug_attempts 3


