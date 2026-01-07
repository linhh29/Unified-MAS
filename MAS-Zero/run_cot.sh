export OPENAI_API_KEY=sk-BDrpp8zrYLtMWyfY2YZJZZPjIOXwikCyZFfDWL8eUGDqnts2
export OPENAI_API_BASE=https://api.qingyuntop.top/v1
# travelplanner, hosp_summ, j1eval
DATASET=j1eval
# gpt-5-mini, gemini-3-flash-preview, deepseek-v3.2, qwen3-30b-a3b-instruct-2507
MODEL=gemini-3-flash-preview

python async_cot_main_question.py  --dataset ${DATASET} --model ${MODEL} --save_dir ./cot_results_${MODEL}/ 


