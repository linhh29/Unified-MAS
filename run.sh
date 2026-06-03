export OPENAI_API_KEY=xx
export OPENAI_API_BASE=xx

export SERPER_API_KEY=xx
export GITHUB_TOKEN=xx

# 使用默认参数运行
META_MODEL=gemini-3-pro-preview

# gemini-3-flash-preview, gpt-5-mini, deepseek-v3.2, qwen3-next-80b-a3b-instruct
EXECUTOR_MODEL=qwen3-next-80b-a3b-instruct

# 顺序执行 travelplanner 和 healthbench
for DATANAME in j1eval travelplanner healthbench deepfund; do
    DATASET=xx/${DATANAME}_validate.jsonl
    echo "=========================================="
    echo "Running: ${DATANAME}"
    echo "=========================================="
    使用自定义参数运行（示例）
    python run_search.py \
        --model ${META_MODEL} \
        --temperature 1 \
        --max_completion_tokens 32768 \
        --data_path ${DATASET} \
        --max_search_results 10 \
        --max_rounds 10 \
        --max_concurrent 50

    python run_optimize.py \
        --nodes_json xx/${DATANAME}/search/generated_nodes.json \
        --input_data ${DATASET} \
        --meta_model ${META_MODEL} \
        --executor_model ${EXECUTOR_MODEL} \
        --temperature 1 \
        --max_completion_tokens 32768 \
        --dataset_name ${DATANAME} \
        --max_search_results 10 \
        --max_rounds 1 \
        --max_debug_attempts 3 \
        --num_epochs 10 \
        --max_workers 50
        # --samples_per_epoch 1
done