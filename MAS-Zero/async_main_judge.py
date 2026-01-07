import argparse
import asyncio
import json
import os
import re

from common import EQUALITY_TEMPLATE, MCQ_EQUALITY_TEMPLATE, ANSWER_PATTERN
from llm_judge import self_verifier_list_wise
from sampler.chat_completion_sampler import ChatCompletionSampler, AsyncChatCompletionSampler
from sampler.o_chat_completion_sampler import OChatCompletionSampler
from sampler.together_completion_sampler import ChatCompletionSampler as ToChatCompletionSampler
from sampler.vllm_completion_sampler import ChatCompletionSampler as VllmChatCompletionSampler
from score import travelplanner_extraction
from travel_eval_utils.travelplanner_eval import eval_score as eval_travelplanner_score
from hosp_summ_eval_utils.hosp_summ_eval import eval_score_async as eval_hosp_summ_score_async
from j1eval_eval_utils.j1eval_eval import eval_score_async as eval_j1eval_score_async



parser = argparse.ArgumentParser()
parser.add_argument('--judge_method', type=str)
parser.add_argument('--baseline', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--max_sample', type=int)
parser.add_argument('--min_sample', type=int, default=0)
parser.add_argument('--max_response_per_sample', type=int)
parser.add_argument('--model', type=str, default="gpt-5-mini")
parser.add_argument('--majority_vote', action='store_true')
parser.add_argument("--save_dir", type=str, default="async_results")
parser.add_argument('--max_concurrent', type=int, default=50, help='Maximum number of concurrent tasks')
args = parser.parse_args()


async def process_single_example(
    example_id: int,
    dataset: str,
    judge_method: str,
    max_response_per_sample: int,
    model: str,
    majority_vote: bool,
    root_dir: str,
    result_path: str,
    post_processer,
    equality_checker,
    sampler,
    result_lock: asyncio.Lock,
    special_ids_lock: asyncio.Lock,
    special_ids: list,
    correct_example_lock: asyncio.Lock,
    correct_example: list
):
    """Process a single example asynchronously"""
    print(f'-------- example_id {example_id} --------')

    response_path = f'{root_dir}/{dataset}/{example_id}/{model}_{model}_{model}_0_plan_response'
    # reponse_path = f'{root_dir}/{dataset}/{example_id}/{model}_{model}_{model}_0__reponse' #sometimes miss "plan"

    try:
        with open(response_path, 'r') as json_file:
            responses = json.load(json_file)
    except Exception as e:
        print(f'example_id {example_id} response file {response_path} does not exisit')
        async with special_ids_lock:
            special_ids.append(f'example_id {example_id} response file {response_path} does not exisit')
        async with correct_example_lock:
            correct_example.append(0)
        return

    if len(responses) < max_response_per_sample:
        print(f'responses length {len(responses)} is lower than {max_response_per_sample}')
        async with special_ids_lock:
            special_ids.append(f'example_id {example_id}: responses length {len(responses)} is lower than {max_response_per_sample}')
        # just a warning is fine
        # continue

    question = responses[0]['problem']  # all responses have the same answer

    # accumulate
    extracted_answers = []
    correct_answers = []

    for response in responses:
        filter_response = response['response']

        # TODO: for gpqa, in some cases, it gives the final answer instead of final selection
        if '<TOO_HARD>' in filter_response:
            filter_response = filter_response[:filter_response.index('<TOO_HARD>')]
            # print(f'<TOO_HARD> detected: response: {response['response']}; filter_response: {filter_response}')

        if dataset == 'travelplanner':
            extraction_prompt = travelplanner_extraction + "Text:\n" + filter_response + "\nJSON:\n"
            extracted_answer, _ = await equality_checker([dict(content=extraction_prompt, role="user")], response_format='normal')
            extracted_answer = extracted_answer.replace('```', '').replace('json', '')
        elif dataset == 'j1eval':
            extracted_answer = filter_response.split('\nAnswer:', 1)[-1].strip()
        elif dataset == 'hosp_summ' or dataset == 'hospsumm':
            extracted_answer = filter_response.split('\nAnswer:', 1)[-1].strip()
        else:
            raise NotImplementedError

        extracted_answers.append(
            extracted_answer.strip() if extracted_answer is not None else extracted_answer)  # for exact match, "strip()" can make a significant difference

        correct_answer = response['correct_answer']
        correct_answers.append(correct_answer)

    print('extracted_answers: ', extracted_answers)
    print('correct_answers: ', correct_answers)

    is_correct = False

    if judge_method == 'self':
        # TODO: consider a list-wise judge

        post_process_path = f'{root_dir}/{dataset}/{example_id}/{model}_{model}_{model}_0_plan_sub_task_post_process.json'
        log_path = f'{root_dir}/{dataset}/{example_id}/{model}_{model}_{model}_0_plan_sub_self_verifier_log'
        score_path = f'{root_dir}/{dataset}/{example_id}/{model}_{model}_{model}_0_plan_score.json'

        chosen_id = await self_verifier_list_wise.run_self_verifier(post_process_path, log_path, score_path, responses, sampler, post_processer,
                                                              extracted_answers, dataset, max_response_per_sample, majority_vote)

        print('chosen_id: ', chosen_id)
        correct_answer = correct_answers[chosen_id]
        extracted_answer = extracted_answers[chosen_id]

        if dataset == 'travelplanner':
            res, concrete_dict = eval_travelplanner_score(extracted_answer, example_id)
            print(111111111,res)
            print(concrete_dict)
            score = int(res)
        elif dataset == 'hosp_summ' or dataset == 'hospsumm':
            res = await eval_hosp_summ_score_async(extracted_answer, example_id, set_type='test')
            print(f'HospSumm LLM Judge Score for instance {example_id}: {res}')
            score = float(res)  # LLM Judge Score is a float between 0 and 1
        elif dataset == 'j1eval':
            res = await eval_j1eval_score_async(extracted_answer, example_id)
            print(f'J1Eval score for instance {example_id}: {res}')
            score = float(res)
        else:
            raise NotImplementedError
            

        # For hosp_summ, LLM Judge Score is a float between 0 and 1, so we always record the score
        if dataset == 'hosp_summ' or dataset == 'j1eval':
            print(f'summary: correct_answer: {correct_answer} vs. extracted_answer: {extracted_answer}')
            async with result_lock:
                with open(result_path, "a+") as fh:
                    fh.write(
                        f'experiemnt {example_id}: LLM_Judge_Score={score:.4f} ({responses[chosen_id]["n"]}); correct_answer: {correct_answer} vs. extracted_answer: {extracted_answer}; SCORE: {score}\n')
            is_correct = True  # Always record for hosp_summ to track LLM Judge scores
        elif dataset == 'travelplanner':
            print(f'correct: correct_answer: {correct_answer} vs. extracted_answer: {extracted_answer}')
            async with result_lock:
                with open(result_path, "a+") as fh:
                    fh.write(
                        f'experiemnt {example_id}: 1 ({responses[chosen_id]["n"]}); correct_answer: {correct_answer} vs. extracted_answer: {extracted_answer}; SCORE: {score}\n')
            is_correct = True

    if is_correct:
        async with correct_example_lock:
            correct_example.append(1)
    else:
        print(f'Cannot Find Correct Answer acorss reponses for example_id: {example_id}')
        async with correct_example_lock:
            correct_example.append(0)


async def main():
    dataset = args.dataset
    judge_method = args.judge_method
    max_sample = args.max_sample
    min_sample = args.min_sample
    max_response_per_sample = args.max_response_per_sample
    model = args.model
    majority_vote = args.majority_vote
    max_concurrent = args.max_concurrent

    special_ids = []
    root_dir = f'{args.save_dir}/question/meta_agent/{args.baseline}'

    # all results
    result_path = f'{root_dir}/{dataset}/{model}_{model}_{judge_method}.results_{max_response_per_sample}'
    if os.path.exists(result_path):
        os.remove(result_path)  # remove the file, do not repeat

    print('result_path: ', result_path)

    # we always use gpt-4o for post-process and equilty check
    post_processer = AsyncChatCompletionSampler(model='gpt-4o')
    equality_checker = AsyncChatCompletionSampler(model='gpt-4o')
    sampler = AsyncChatCompletionSampler(model=model)

    correct_example = []

    # Create locks for thread-safe operations
    result_lock = asyncio.Lock()
    special_ids_lock = asyncio.Lock()
    correct_example_lock = asyncio.Lock()

    # Control concurrency with semaphore
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(example_id):
        async with semaphore:
            return await process_single_example(
                example_id, dataset, judge_method, max_response_per_sample, model, majority_vote,
                root_dir, result_path, post_processer, equality_checker, sampler,
                result_lock, special_ids_lock, special_ids, correct_example_lock, correct_example
            )

    # Create tasks for all examples
    tasks = []
    for example_id in range(min_sample, max_sample + 1):
        tasks.append(process_with_semaphore(example_id))

    # Execute all tasks concurrently
    await asyncio.gather(*tasks)

    for special_id in special_ids:
        print('special_id: ', special_id)

    acc = sum(correct_example) / len(correct_example) if len(correct_example) > 0 else 0
    print(f'coorect {sum(correct_example)}; Total: {len(correct_example)}; Acc: {acc}')

    with open(result_path, "a+") as fh:
        fh.write(f'coorect {sum(correct_example)}; Total: {len(correct_example)}; Acc: {acc}\n')


if __name__ == "__main__":
    asyncio.run(main())
