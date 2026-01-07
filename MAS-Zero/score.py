from common import ANSWER_PATTERN, async_check_equality
from sampler import AsyncChatCompletionSampler

from utils import extract_xml
from utils import load_questions
import common
import json
from common import HTML_JINJA, SingleEvalResult
import re
from travel_eval_utils.travelplanner_eval import eval_score as eval_travelplanner_score
from hosp_summ_eval_utils.hosp_summ_eval import eval_score_async as eval_hosp_summ_score_async
from j1eval_eval_utils.j1eval_eval import eval_score_async as eval_j1eval_score_async

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

j1eval_extraction = """你是一名熟悉中国法院裁判文书结构的法律助理，擅长从冗长文本中精准提取“最终裁判结果/主文”。现在我会给你一段中国法院的裁判文书内容（可能包含案情、理由、量化思路、补证要求等），你的任务是只基于文本本身，提取并整理出该文书的“最终审判结果”。
例子：
Text:
裁判说理与主文                                                                                                                            
一、案件基本事实与证据采信                                                                                                                                   
1. 经审理查明并采信的事实：原告主张其于2021年5月20日将一尊财神铜像及一件三足鼎交由被告存放（原告提交2021年5月20日《存放协议》），双方于2021年8月18日就返还事 
宜另行签订合约，约定被告应于2021年8月28日中午12点前返还涉案物品，若不能返还则被告需赔偿580,000元。被告在诉辩中承认其签署了2021年8月18日的合约并对物品曾占有。
基于上述书面证据及被告的承认，可以认定双方之间存在关于涉案物品的交付关系以及被告对返还的书面承诺，故被告对原告负有返还义务。上述属于法院在现有材料下可采信的 
事实。                                                                                                                                                       
2. 未被采信或证据不足之处：被告主张涉案物品被九龙水上俱乐部扣押且其已向派出所、俱乐部及12345投诉但未能取回，属关键抗辩事实。但被告未能提交俱乐部的书面收存/扣
押凭证、派出所的报警回执或不予立案书面说明、监控录像或第三方书面证言等能证明第三方扣押或控制事实的独立书面证据，故该项抗辩在现有证据下不能成立。另就《存放协 
议》与2021年8月18日合约的具体条款内容、涉案物品的所有权证据及市场价值，案卷亦未提交权威鉴定或完整证明材料，导致对赔偿金额580,000元的性质（违约金或预定损害赔 
偿）与合理性无法作出确定性采信。                                                                                                                             
二、法律适用与责任认定理由 
1. 法律适用：本案适用《中华人民共和国民法典》关于合同、物权保护和侵权的相关规定。合同约定受法律保护，当事人约定返还义务明确的，受约束。若一方不履行[168/1897]
另一方有权请求返还原物或请求损害赔偿。对约定违约金或约定赔偿数额，法院可根据实际损失、公平原则及当事过错对其进行调整。                                       
2. 被告返还义务与违约：基于被告承认签署的2021年8月18日合约可认定其对原告负有在既定期限内返还涉案物品的义务。被告未按约定返还，构成违约（合同责任）。除非被告 
能提供充分证据证明其免责事由（如经原告同意处置、不可抗力或第三方独立扣押且其已尽到合理救济义务），否则应承担违约责任。                                       
3. 第三人抗辩的举证与影响：被告主张第三方扣押为其不能返还的事实载明责任归属的关键，依法应由被告承担举证责任。若被告能举证证明第三方非法扣押且其本人已尽合理保
管义务并积极采取了救济措施，法院可据此酌情减轻或免除其对原告的责任；但此情形不取消原告对第三方直接请求返还或赔偿的权利。本案中被告未提交足以证明第三方扣押及 
其尽责的独立证据，故其抗辩不能成立，不能免除其对原告的违约责任。                                                                                             
4. 赔偿与返还的替代关系：返还原物为首要救济；若物品确因客观原因不能返还且该事实成立，应以赔偿实际损失为替代救济。双方书面约定的580,000元为事前约定的赔偿/违约
金额，但法院在确认约定数额时需审查其与实际损失的相称性及是否显失公平。本案因缺乏权威鉴定价或其他证明实际损失的证据，不能直接采纳或支持580,000元的全额请求，法
院应在保障权利人与防止证据不足情形下对赔偿数额作区间化、分情形裁量或责令补证后再行确定。                                                                     
三、对赔偿数额与费用承担的裁量原则（量化思路）                                                                                                               
1. 裁量原则：赔偿以实际损失为基准，兼顾合同约定与当事过错；对当事人约定的高额赔偿，若证据能证明其合理且不显失公平，法院可维持；证据不足或约定过高且与实际损失
明显不符，法院应依法调整。诉讼费用、保全、鉴定及担保费用原则上由败诉方承担，部分胜诉或责任分配时按比例分担。                                                 
2. 鉴于本案证据状况，法院在终局判决前作出以下处理：                                                                                                          
- 责令当事人在判决送达之日起15日内补齐下列关键证据（详见下文“补证清单”）。                                                                                   
- 若当事人在限期内补齐并证实涉案物品的权威鉴定价V及/或证实第三方扣押事实，经法院审查可据以作出最终量化判决；若当事人在限期内未能补正全部关键证据，法院将基于 
现有证据作出初步判决并在可允许范围内采用证据强度分层后的费率与赔偿区间进行裁量（具体如下）：                                                                 
情形A（补证充分并经鉴定或第三方书证支持）：以鉴定评估价V为赔偿基准；如合约约定580,000元与V相当且不显失公平，可维持；若580,000元显著高于V且被告无重大过错，按V
或在V基础上酌情上浮（上浮比例一般不超过30%，视被告过错情节决定）。被告在此情形下承担全部诉讼费、鉴定费及原告为保全垫付的费用，原告可对第三方另行追偿。       
情形B（部分证据补正、但对价值或扣押事实仍有争议）：法院可在100,000—300,000元区间内判定赔偿，首选基点150,000元；被告尽责且积极救济的向区间下端调整，存在过错或
隐匿行为的向区间上端调整；费用按胜诉比例或法院酌定分配。                                                                                                     
情形C（关键证据未补正、证据薄弱）：法院可在保守区间50,000—150,000元内作出初步判决，倾向于区间下部，并保留原告在补证后申请追加或变更执行的权利；费用分配亦可酌
定以保障程序公正。                                                                                                                                           
四、主文（裁判结果）                                                                                                                                         
1. 确认：被告陈某淐对原告原某光存有返还一尊财神铜像及一件三足鼎的合同义务（根据双方2021年5月20日存放协议及2021年8月18日合约），被告未按约定返还，构成违约。  
2. 责令补证：鉴于本案涉及赔偿数额及第三方扣押事实存有关键证据缺失，判决如下：被告在本判决书生效之日起15日内向本院提交下列证据原件或经法院核验的复印件并在送达
之日起同时抄送原告：                                                                                                                                         
   （1）2021年8月18日合约与2021年5月20日《存放协议》的完整原件及签字页；                                                                                     
   （2）涉案财神铜像与三足鼎的高清照片、尺寸、重量及完好状况说明；
   （3）若有，请提交权威文物或市场评估机构出具的材质、年代及市场评估报告；
   （4）九龙水上俱乐部（或现场管理方）关于涉案物品收存、扣押或控制的书面凭证或说明；
   （5）向潍州路派出所的报警回执、受理或不予立案的书面说明、12345投诉受理及处理回执、与俱乐部的书面沟通记录或调解记录；
   （6）当时在场人员（如刘桂芹、张师傅、会所工作人员等）的书面证言或可供核验的联系方式。
3. 暂定赔偿与费用负担：
   （1）在当事人提交并经法院审查、必要时委托鉴定或核验证据后，法院将根据补证情况适用上文所述的情形A、B或C的量化方法确定最终赔偿数额并作出终局判决；
   （2）在补证期间，若原告申请财产保全并经法院裁定采取保全措施，保全所需的担保由申请人先行提供；保全经终局判决确认支持的，相关保全费及担保费由败诉方承担；若
保全被认定无理由或滥用，申请人承担相应费用及对被保全方的损失负赔偿责任；
   （3）本案的诉讼费、鉴定费及保全/担保费用的最终承担将根据法院对事实与责任的最终认定确定，原则上由败诉方或按责任比例分担。
五、送达与救济
1. 被告应在规定期限内提交上述证据，逾期未提交或提交不充分的，法院将根据现有证据依法作出判决并可在本案的保守量化区间内裁定赔偿；原告保留在补证后申请追加执行或
变更判决的权利。2. 如不服本判决，当事人可在法律规定的期限内向上一级人民法院提起上诉。
补证清单（为便于当事人和法院执行，重复列明）：2021年8月18日合约与2021年5月20日存放协议原件、涉案物品高清照片与实物说明、权威鉴定/评估报告（如有）、九龙水上俱
乐部对收存/扣押的书面凭证、潍州路派出所的报警回执或处理文书、12345投诉受理回执、在场证人书面证言或可供核验联系方式。

Extracted Answer:
1. 被告陈某淐对原告原某光存有返还一尊财神铜像及一件三足鼎的合同义务（根据双方2021年5月20日存放协议及2021年8月18日合约），被告未按约定返还，构成违约。
2. 在当事人提交并经法院审查、必要时委托鉴定或核验证据后，法院将根据补证情况适用上文所述的情形A、B或C的量化方法确定最终赔偿数额并作出终局判决。         
3. 在补证期间，若原告申请财产保全并经法院裁定采取保全措施，保全所需的担保由申请人先行提供；保全经终局判决确认支持的，相关保全费及担保费由败诉方承担；若保全被认定无理由或滥用，申请人承担相应费用及对被保全方的损失负赔偿责任。                                                                                    
4.本案的诉讼费、鉴定费及保全/担保费用的最终承担将根据法院对事实与责任的最终认定确定，原则上由败诉方或按责任比例分担。

"""

class DataScorer:

    def __init__(self, dataset, technique, mode_verifier):
        self.dataset = dataset
        self.technique = technique
        self.equality_checker = AsyncChatCompletionSampler(model="gpt-4o")
        self.mode_verifier = mode_verifier
        self.LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    async def run_score(self, answer, extracted_answer, use_oracle_verifier, judge_path, instance_id, n, code_snippet):

        print(11111111111, instance_id)
        if 'travelplanner' in self.dataset:
            res, concrete_dict = eval_travelplanner_score(extracted_answer, instance_id)
            print(111111111,res)
            print(concrete_dict)
            return float(res)
        elif 'hosp_summ' in self.dataset or 'hospsumm' in self.dataset:
            res = await eval_hosp_summ_score_async(extracted_answer, instance_id, set_type='test')
            print(f'HospSumm LLM Judge Score for instance {instance_id}: {res}')
            return float(res)
        elif 'j1eval' in self.dataset:
            res = await eval_j1eval_score_async(extracted_answer, instance_id)
            print(f'J1Eval score for instance {instance_id}: {res}')
            return float(res)
        else:
            raise NotImplementedError

    async def score(self, example_id, n, prompt_message, question, response_text, answer, sub_tasks_text, use_oracle_verifier, judge_path, response_path,
                    response_dict, instance_id, code_snippet):
        if 'travelplanner'in self.dataset:
            extraction_prompt = travelplanner_extraction + "Text:\n" + response_text + "\nJSON:\n"
            extracted_answer, _ = await self.equality_checker([dict(content=extraction_prompt, role="user")], response_format='normal')
            extracted_answer = extracted_answer.replace('```', '').replace('json', '')
        elif 'j1eval'in self.dataset:
            # extraction_prompt = j1eval_extraction + "Text:\n" + response_text + "\nExtracted Answer:\n"
            # extracted_answer, _ = await self.equality_checker([dict(content=extraction_prompt, role="user")])
            extracted_answer = response_text.split('\nAnswer:', 1)[-1].strip()
        elif 'hosp_summ' in self.dataset or 'hospsumm' in self.dataset:
            extracted_answer = response_text.split('\nAnswer:', 1)[-1].strip()
        else:
            raise NotImplementedError
        if '[TOO_HARD]' in extracted_answer:
            extracted_answer = extracted_answer[:extracted_answer.index('[TOO_HARD]')]

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
