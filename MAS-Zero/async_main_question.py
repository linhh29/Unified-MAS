import argparse
import asyncio
import copy
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

import async_search as search
from prompts.swe.patch_oracle import AGENTLESS_REPAIR
from sampler import init_model
from utils import extract_xml
from utils import load_questions

PLANNER_INSTRUCTION = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B).

***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
Travel Plan:
Day 1:
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 2:
Current City: Charlotte
Transportation: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte.
Lunch: Olive Tree Cafe, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -

***** Example Ends *****

Given information: {text}
Query: {query}
Travel Plan:"""

J1_INSTRUCTION = """你是一名严谨、公正的审判长，你的任务是：根据原告和被告的陈述以及一些额外的信息，生成相应的说理内容，给出最终判决。请你始终保持中立、专业、公正的审判风格，不得偏袒任何一方。请用中文输出你的内容。
你可以获得到的信息：{text}
"""

HospSumm_INSTRUCTION = """
Patient Records: {patient_info}

Summarize the key diagnostic information and significant results based on the patients’ multiple health (long) records during hospitalization.
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--valid_size', type=int, default=128)
    # parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repeats', type=int, default=1)
    # parser.add_argument('--multiprocessing', action='store_true', default=True)
    # parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='./async_results/')
    parser.add_argument('--expr_name', type=str)
    parser.add_argument('--n_generation', type=int, default=10)
    parser.add_argument('--max_round', type=int, default=5)
    parser.add_argument('--max_sc', type=int, default=5)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--option', type=str, default='')
    parser.add_argument('--meta_model',
                        type=str)
    parser.add_argument('--node_model',
                        type=str)
    parser.add_argument('--verifier_model',
                        type=str)
    
    parser.add_argument('--shorten_context', action='store_true')
    parser.add_argument('--merge_context', action='store_true')

    parser.add_argument(
        "--blocks", type=str, nargs="*", help="Number of examples to use (overrides default)"
    )
    parser.add_argument('--dataset', type=str)
    parser.add_argument(
        "--given_examples", type=int, nargs="*", help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--use_oracle_verifier", action='store_true', default=False
    )
    parser.add_argument(
        "--defer_verifier", action='store_true'
    )
    parser.add_argument(
        "--no_decompose", action='store_true'
    )
    parser.add_argument(
        "--no_meta_reward", action='store_true'
    )
    args = parser.parse_args()

    return args


async def run_sync_search(example, example_id, meta_model, node_model, verifier_model, n, dataset, extra_info,
                          blocks, n_generation, save_dir, option, defer_verifier, debug_max):
    expr_name = f'question/meta_agent/{dataset}/{example_id}/{meta_model}_{node_model}_{verifier_model}_{n}'
    print('args.expr_name: ', expr_name)
    print(example.keys())

    if 'hosp_summ' in dataset or 'hospsumm' in dataset:
        patient_info = example['instruct']
        reference_summary = example['answer']
        questions = [HospSumm_INSTRUCTION.format(patient_info=patient_info)]
        answers = [reference_summary]
    elif 'travelplanner' in dataset:
        questions = [PLANNER_INSTRUCTION.format(text=example['reference_information'], query=example['query'])]
        answers = ['']
    elif 'j1eval' in dataset:
        drop_keys = {"id", "court_information"}
        obj = copy.deepcopy(example)
        parts = []
        for k in drop_keys:
            obj.pop(k, None)
        parts.append(json.dumps(obj, ensure_ascii=False))
        result = "".join(parts)
        questions = [J1_INSTRUCTION.format(text=result)]
        answers = [example['court_information']['ground_truth']["court_judgment"]]


    task_queue = []
    for q in questions:
        taskInfo = ('task', 'User', q, None, None, None, -1)
        task_queue.append(taskInfo)

    extra_info["answers"] = answers
    extra_info["questions"] = questions
    extra_info["example_id"] = example_id
    extra_info["instance_id"] = example_id
    extra_info["response_dict"] = []

    # search
    await search.search(extra_info, task_queue, meta_model, blocks, verifier_model, n_generation,
                        save_dir, expr_name, option, dataset, defer_verifier, debug_max)


async def main(args):
    blocks = args.blocks
    meta_model = args.meta_model
    node_model = args.node_model
    verifier_model = args.verifier_model
    use_oracle_verifier = args.use_oracle_verifier
    max_round = args.max_round
    max_sc = args.max_sc

    print('meta_model: ', meta_model)
    print('verifier_model: ', verifier_model)
    print('node_model: ', node_model)

    json_model = ['gpt', 'gemini', 'qwen', 'deepseek']
    # xml_model = ['qwen', 'llama-3.3', 'deepseek']
    xml_model = ['llama-3.3']

    extra_info = {}
    if any(kw in node_model for kw in json_model):
        format_inst_template = ("Reply EXACTLY with the following JSON format.\n{request_keys}\n"
                                "DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n\n"
                                "If there are any double quotation marks or other special characters in the response, you should escape them with a backslash!\n\n")
        extra_info["format_choice"] = "json"

    elif any(kw in node_model for kw in xml_model):
        format_inst_template = ("Reply EXACTLY with the following XML format.\n{request_keys}\n"
                                "DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!\n\n")
        extra_info["format_choice"] = 'xml'

    else:
        raise NotImplementedError

    init_model(verifier_model)
    init_model(node_model)
    init_model(meta_model)

    extra_info["FORMAT_INST"] = format_inst_template
    extra_info["shorten_context"] = args.shorten_context
    extra_info["merge_context"] = args.merge_context
    extra_info["COST_TOTAL"] = 0.0
    extra_info["no_decompose"] = args.no_decompose
    extra_info["no_meta_reward"] = args.no_meta_reward

    print('shorten_context: ', args.shorten_context)
    print('merge_context: ', args.merge_context)
    print('global_no_meta_reward: ', args.no_meta_reward)
    print('global_no_decompose: ', args.no_decompose)

    code_snippet = None
    for n in range(args.n_repeats):
        if 'travelplanner' in args.dataset:

            cot_instruction = "Please think step by step and then solve the task."
            # output_description = "Return ONLY an integer. DO NOT return anything other than the integer answer."
            output_description = (
                "If the question is asked for a numeric result, Return ONLY an integer and DO NOT return anything other than the integer answer; "
                "If the question is asked for more than numeric results, Return what the question asked and make sure the answer is complete.")

            debate_role = ['Math Professor', 'Travel Planner']

            test_path = Path("/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/travelplanner_test_16.jsonl")
            if not test_path.exists():
                raise FileNotFoundError(f"TravelPlanner test file not found: {test_path}")

            with test_path.open("r", encoding="utf-8") as fh:
                examples = [json.loads(line) for line in fh if line.strip()]


            extra_info["node_model"] = node_model
            extra_info["verifier_model"] = verifier_model
            extra_info["output_description"] = output_description
            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["dataset"] = args.dataset
            extra_info["code_snippet"] = code_snippet

            # 控制并发数量的信号量，最多同时运行5个任务
            semaphore = asyncio.Semaphore(50)

            async def run_task_with_semaphore(*a, **kw):
                async with semaphore:
                    return await run_sync_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):
                # print(example['query'])

                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

            await tqdm_asyncio.gather(*tasks)
        
        elif 'hosp_summ' in args.dataset or 'hospsumm' in args.dataset:

            cot_instruction = "Please think step by step and then solve the task."
            # output_description = "Return ONLY an integer. DO NOT return anything other than the integer answer."
            output_description = (
                "Summarize the key diagnostic information and significant results based on the patients' multiple health (long) records during hospitalization. "
                "Provide a comprehensive discharge summary.")

            debate_role = ['Medical Expert', 'Doctor', 'Nurse']

            test_path = Path("/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/hosp_summ_test_16.jsonl")
            if not test_path.exists():
                raise FileNotFoundError(f"HospSumm test file not found: {test_path}")

            with test_path.open("r", encoding="utf-8") as fh:
                examples = [json.loads(line) for line in fh if line.strip()]


            extra_info["node_model"] = node_model
            extra_info["verifier_model"] = verifier_model
            extra_info["output_description"] = output_description
            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["dataset"] = args.dataset
            extra_info["code_snippet"] = code_snippet

            # 控制并发数量的信号量，最多同时运行5个任务
            semaphore = asyncio.Semaphore(50)

            async def run_task_with_semaphore(*a, **kw):
                async with semaphore:
                    return await run_sync_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):
                # print(example['query'])

                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

            await tqdm_asyncio.gather(*tasks)
        elif 'j1eval' in args.dataset:

            cot_instruction = "Please think step by step and then solve the task."
            # output_description = "Return ONLY an integer. DO NOT return anything other than the integer answer."
            output_description = (
                "If the question is asked for a numeric result, Return ONLY an integer and DO NOT return anything other than the integer answer; "
                "If the question is asked for more than numeric results, Return what the question asked and make sure the answer is complete."
                "如果问题是中文信息，请用中文输出你的内容！")

            debate_role = ['Jugder', 'Lawyer']

            test_path = Path("/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/j1eval_test_16.jsonl")
            if not test_path.exists():
                raise FileNotFoundError(f"TravelPlanner test file not found: {test_path}")

            with test_path.open("r", encoding="utf-8") as fh:
                examples = [json.loads(line) for line in fh if line.strip()]


            extra_info["node_model"] = node_model
            extra_info["verifier_model"] = verifier_model
            extra_info["output_description"] = output_description
            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["dataset"] = args.dataset
            extra_info["code_snippet"] = code_snippet

            # 控制并发数量的信号量，最多同时运行5个任务
            semaphore = asyncio.Semaphore(50)

            async def run_task_with_semaphore(*a, **kw):
                async with semaphore:
                    return await run_sync_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):
                # print(example['query'])

                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

            await tqdm_asyncio.gather(*tasks)

        else:
            raise NotImplementedError


if __name__ == '__main__':
    args = parse_arguments()

    asyncio.run(main(args))
