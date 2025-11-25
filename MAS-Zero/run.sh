export OPENAI_API_KEY=

# travelplanner, agentclinic
DATASET=j1eval
MODEL=gpt-4o_chatgpt

# python async_main_question.py  --dataset workflow_search/${DATASET} --option plan --meta_model ${MODEL} --node_model ${MODEL} --verifier_model ${MODEL} --blocks COT COT_SC Reflexion LLM_debate --use_oracle_verifier --defer_verifier --n_generation 5 


# python main_judge.py  --dataset ${DATASET} --judge_method self --baseline workflow_search --model ${MODEL} --min_sample 0 --max_sample 181 --max_response_per_sample 5 
