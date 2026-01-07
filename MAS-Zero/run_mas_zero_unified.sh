export OPENAI_API_KEY=sk-BDrpp8zrYLtMWyfY2YZJZZPjIOXwikCyZFfDWL8eUGDqnts2
export OPENAI_API_BASE=https://api.qingyuntop.top/v1
# travelplanner, hosp_summ, j1eval
DATASET=travelplanner
# gpt-5-mini, gemini-3-flash-preview, deepseek-v3.2, qwen3-30b-a3b-instruct-2507
MODEL=gemini-3-flash-preview

if [ "$DATASET" == "j1eval" ]; then
    BLOCKS="DAMAGES_CALCULATOR FACT_STRUCTURING_AGENT LEGAL_KNOWLEDGE_RETRIEVER LIABILITY_ASSESSMENT_AGENT JUDGMENT_DRAFTER"
elif [ "$DATASET" == "hosp_summ" ]; then
    BLOCKS="SECTION_SEGMENTER_TOOL MEDICAL_ONTOLOGY_RAG RESULTS_PROCESSING_AGENT COURSE_PLAN_AGENT DIAGNOSTIC_SUMMARY_AGENT FINAL_REFINEMENT_AGENT"
elif [ "$DATASET" == "travelplanner" ]; then
    BLOCKS="CONSTRAINT_EXTRACTOR_AGENT ITINERARYPLANNER_AGENT DATA_FILTER_TOOL VALIDATION_TOOL REFINEMENT_AGENT"
fi

# python async_main_question.py  --dataset workflow_search/${DATASET} --option plan --meta_model ${MODEL} --node_model ${MODEL} --verifier_model ${MODEL} --blocks ${BLOCKS} --use_oracle_verifier --defer_verifier --n_generation 5 --save_dir ./async_results_unified_${MODEL}/


python async_main_judge.py  --dataset ${DATASET} --judge_method self --baseline workflow_search --model ${MODEL} --min_sample 0 --max_sample 181 --max_response_per_sample 5 --save_dir ./async_results_unified_${MODEL}/
