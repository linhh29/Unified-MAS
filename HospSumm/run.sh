KEY=sk-BDrpp8zrYLtMWyfY2YZJZZPjIOXwikCyZFfDWL8eUGDqnts2
URL=https://api.qingyuntop.top/v1

# gpt-5-mini, gemini-3-flash-preview, deepseek-v3.2, qwen3-30b-a3b-instruct-2507
MODEL=gemini-3-flash-preview

python3 HospSumm.py --openai_api_key ${KEY} --doctor_llm ${MODEL} --patient_llm ${MODEL} --measurement_llm ${MODEL} --inf_type llm --base_url ${URL} --output_dir ./async_results_${MODEL} --max_concurrent 50