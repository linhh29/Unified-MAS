from common import ANSWER_PATTERN, async_check_equality
from sampler import AsyncChatCompletionSampler

from utils import extract_xml
from utils import load_questions
import common
import json
from common import HTML_JINJA, SingleEvalResult
import re
from travel_eval_utils.travelplanner_eval import eval_score as eval_travelplanner_score
# MedSentry eval is imported inline in run_score to use async version

travelplanner_extraction = """Please assist me in extracting valid information from a given natural language text and reconstructing it in JSON format, as demonstrated in the following example. If transportation details indicate a journey from one city to another (e.g., from A to B), the 'current_city' should be updated to the destination city (in this case, B). Use a ';' to separate different attractions, with each attraction formatted as 'Name, City'. If there's information about transportation, ensure that the 'current_city' aligns with the destination mentioned in the transportation details (i.e., the current city should follow the format 'from A to B'). Also, ensure that all flight numbers and costs are followed by a colon (i.e., 'Flight Number:' and 'Cost:'), consistent with the provided example. Each item should include ['day', 'current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation']. Replace non-specific information like 'eat at home/on the road' with '-'. Additionally, delete any '$' symbols.
-----EXAMPLE-----
 [{{
        "days": 1,
        "current_city": "from Dallas to Peoria",
        "transportation": "Flight Number: 4044830, from Dallas to Peoria, Departure Time: 13:10, Arrival Time: 15:01",
        "breakfast": "-",
        "attraction": "Peoria Historical Society, Peoria;Peoria Holocaust Memorial, Peoria;",
        "lunch": "-",
        "dinner": "Tandoor Ka Zaika, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    }},
    {{
        "days": 2,
        "current_city": "Peoria",
        "transportation": "-",
        "breakfast": "Tandoor Ka Zaika, Peoria",
        "attraction": "Peoria Riverfront Park, Peoria;The Peoria PlayHouse, Peoria;Glen Oak Park, Peoria;",
        "lunch": "Cafe Hashtag LoL, Peoria",
        "dinner": "The Curzon Room - Maidens Hotel, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    }},
    {{
        "days": 3,
        "current_city": "from Peoria to Dallas",
        "transportation": "Flight Number: 4045904, from Peoria to Dallas, Departure Time: 07:09, Arrival Time: 09:20",
        "breakfast": "-",
        "attraction": "-",
        "lunch": "-",
        "dinner": "-",
        "accommodation": "-"
    }}]
-----EXAMPLE END-----
"""

class DataScorer:

    def __init__(self, dataset, technique, mode_verifier):
        self.dataset = dataset
        self.technique = technique
        self.equality_checker = AsyncChatCompletionSampler(model="gpt-4-turbo-preview")
        self.mode_verifier = mode_verifier
        self.LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    async def run_score(self, answer, extracted_answer, use_oracle_verifier, judge_path, instance_id, n, code_snippet):

        print(11111111111, instance_id)

        if 'swe_bench' in self.dataset:
            raise NotImplementedError("Should use multi")

            score, percentage, passed_tests, total_tests = run_swebench_evaluation(judge_path, instance_id, extracted_answer, self.technique, n, code_snippet)

            with open(judge_path, 'a+') as judge_file:
                judge_file.write(
                    f'{instance_id} → {passed_tests} passed test | {total_tests} total_tests | '
                    f'{passed_tests}/{total_tests} passed → {percentage:.1f}% | Score: {score}\n')

            return score

        elif 'aime24' in self.dataset:
            res = await async_check_equality(self.equality_checker, answer, extracted_answer, use_oracle_verifier=True, judge_path=judge_path)
            return float(res)
        elif 'gpqa_diamond' in self.dataset:
            res = extracted_answer
            is_early_stop = False
            try:
                if isinstance(res, str) and res in self.LETTER_TO_INDEX:
                    predicted_idx = self.LETTER_TO_INDEX[res]
                elif 'A)' in res:
                    predicted_idx = 0
                elif 'B)' in res:
                    predicted_idx = 1
                elif 'C)' in res:
                    predicted_idx = 2
                elif 'D)' in res:
                    predicted_idx = 3
                elif isinstance(res, list):
                    try_res = res[1]
                    predicted_idx = self.LETTER_TO_INDEX[try_res.content]
                elif res.content in self.LETTER_TO_INDEX:
                    predicted_idx = self.LETTER_TO_INDEX[res.content]
                elif 'A)' in res.content:
                    predicted_idx = 0
                elif 'B)' in res.content:
                    predicted_idx = 1
                elif 'C)' in res.content:
                    predicted_idx = 2
                elif 'D)' in res.content:
                    predicted_idx = 3
                else:
                    print(f"error in q {instance_id}")
                    score = 0
                    is_early_stop = True
            except Exception as e:
                score = 0
                is_early_stop = True

            if not is_early_stop:  # if cannot find predicted_idx, then done
                if predicted_idx == answer:
                    score = 1
                else:
                    score = 0

            print(f'extracted_answer: {extracted_answer}; answer: {answer}; score: {score}')

            return score

        elif 'travelplanner' in self.dataset:
            res, concrete_dict = eval_travelplanner_score(extracted_answer, instance_id)
            print(111111111,res)
            print(concrete_dict)
            return float(res)
        elif 'agentclinic' in self.dataset:
            # Use async version for AgentClinic evaluation
            from agentclinic_eval_utils.agentclinic_eval import eval_score_async as eval_agentclinic_score_async
            res = await eval_agentclinic_score_async(extracted_answer, instance_id, set_type='test')
            print(f'AgentClinic score for instance {instance_id}: {res}')
            return float(res)
        elif 'j1eval' in self.dataset:
            # Use async version for AgentClinic evaluation
            from j1eval_eval_utils.j1eval_eval import eval_score_async as eval_j1eval_score_async
            res = await eval_j1eval_score_async(extracted_answer, instance_id)
            print(f'J1Eval score for instance {instance_id}: {res}')
            return float(res)
        else:
            raise NotImplementedError

    async def score(self, example_id, n, prompt_message, question, response_text, answer, sub_tasks_text, use_oracle_verifier, judge_path, response_path,
                    response_dict, instance_id, code_snippet):

        if 'swe_bench' in self.dataset:
            extracted_answer = response_text.split('\n\nAnswer:', 1)[-1].strip()
            if '<patch>' in extracted_answer:
                extracted_answer = extract_xml(extracted_answer, 'patch').strip()
        elif 'travelplanner'in self.dataset:
            extraction_prompt = travelplanner_extraction + "Text:\n" + response_text + "\nJSON:\n"
            extracted_answer, _ = await self.equality_checker([dict(content=extraction_prompt, role="user")], response_format='normal')
            extracted_answer = extracted_answer.replace('```', '').replace('json', '')
        else:
            match = re.search(ANSWER_PATTERN, response_text)
            extracted_answer = match.group(1) if match else None
            extracted_answer = extracted_answer.strip()

        print('extracted_answer: ', extracted_answer)

        with open(judge_path, 'a+') as judge_file:
            judge_file.write(f'Question: {question}\nproposed answer: {response_text}\nExtracted answer: {extracted_answer}\nCorrect answer: {answer}\n')

        with open(response_path, 'w') as json_file:
            response_dict.append({
                'example_id': example_id,
                'problem': question,
                'correct_answer': answer,
                'n': n,
                'response': response_text,
                'sub_tasks_text': sub_tasks_text})

            json.dump(response_dict, json_file, indent=4)

        if use_oracle_verifier:
            score_oracle_verifier = await self.run_score(answer, extracted_answer, use_oracle_verifier=True, judge_path=judge_path, instance_id=instance_id,
                                                         n=n,
                                                         code_snippet=code_snippet)
            score = score_oracle_verifier
            score_model_verifier = None
        else:
            if sub_tasks_text is None:
                score_model_verifier = await self.run_score(self.mode_verifier, question, response_text, use_oracle_verifier=False, judge_path=judge_path,
                                                            instance_id=instance_id, n=n, code_snippet=code_snippet)
            else:
                score_model_verifier = await self.run_score(self.mode_verifier, question, sub_tasks_text, use_oracle_verifier=False, judge_path=judge_path,
                                                            instance_id=instance_id, n=n, code_snippet=code_snippet)
            score = score_model_verifier

        html = common.jinja_env.from_string(HTML_JINJA).render(
            prompt_messages=prompt_message,
            next_message=dict(content=response_text, role="assistant"),
            score=score,
            correct_answer=answer,
            extracted_answer=extracted_answer,
        )
        convo = prompt_message + [dict(content=response_text, role="assistant")]
        results = SingleEvalResult(html=html, score=score, convo=convo)
        return score_oracle_verifier, score_model_verifier, results
