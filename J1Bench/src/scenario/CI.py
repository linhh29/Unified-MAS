import argparse
import asyncio
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import jsonlines
from tqdm import tqdm
import time
import random
from utils.register import register_class, registry
from utils.utils_func import load_jsonl
from utils.cost_tracker import CostTracker

@register_class(alias='J1Bench.Scenario.CI')
class CI:
    def __init__(self, args):
        case_database = load_jsonl(args.case_database)
        self.args = args
        # 不再在初始化时重置全局成本，每个case会独立管理自己的成本
        self.case_triplet = []
        for case in case_database:
            plaintiff = registry.get_class(args.plaintiff)(
                args,
                plaintiff_info = case
                )
            defendant = registry.get_class(args.defendant)(
                args,
                defendant_info = case
                )
            judge = registry.get_class(args.judge)(
                args,
                judge_info = case
                )
            plaintiff.id = case['id']
            defendant.id = case['id']
            judge.id = case['id']
            self.case_triplet.append((plaintiff, defendant, judge))
        self.max_conversation_turn = args.max_conversation_turn
        self.save_path = args.save_path
        self.max_workers = args.max_workers
        self.case_concurrency = args.case_concurrency
        self.write_lock = None
        self.model = args.model
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument("--case_database", default = "/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/j1eval_test_16.jsonl", type=str)
        parser.add_argument("--plaintiff", default="Agent.Plaintiff.GPT_CI", help="registry name of plaintiff agent") 
        parser.add_argument("--defendant", default="Agent.Defendant.GPT_CI", help="registry name of defendant agent")
        parser.add_argument("--judge", default="Agent.Judge.GPT_CI", help="registry name of judge agent")
        parser.add_argument("--max_conversation_turn", default=30, type=int, help="max conversation turn")
        parser.add_argument("--save_path", default="./CI_dialog_history.jsonl", help="save path for dialog history")
        parser.add_argument("--max_workers", default=50, type=int, help="max workers for parallel CI")
        parser.add_argument("--case_concurrency", default=50, type=int, help="max concurrent cases when using run()")
        parser.add_argument("--async_request_concurrency", default=50, type=int, help="max concurrent async OpenAI calls per agent")
        parser.add_argument("--model", default="gpt-5-mini", type=str, help="LLM model name")



    def remove_processed_cases(self):
        processed_case_ids = {}
        if os.path.exists(self.save_path):
            with jsonlines.open(self.save_path, "r") as f:
                for obj in f:
                    processed_case_ids[obj["case_id"]] = 1
            f.close()
        client_num = len(self.case_triplet)
        for i, client in enumerate(self.case_triplet[::-1]):
            print(client[0].id)
            if processed_case_ids.get(client[0].id) is not None:
                self.case_triplet.pop((client_num-(i+1)))
            
        random.shuffle(self.case_triplet)
        print("To-be-consulted case Number: ", len(self.case_triplet))
        
    def run(self):
        asyncio.run(self._run_async())
    
    async def _run_async(self):
        await self._process_cases_concurrently(self.case_concurrency)
    
    def parallel_run(self):
        asyncio.run(self._parallel_run_async())

    async def _parallel_run_async(self):
        await self._process_cases_concurrently(self.max_workers, label="Parallel Consult Start")

    async def _process_cases_concurrently(self, concurrency, label=None):
        self.remove_processed_cases()
        if not self.case_triplet:
            print("No cases to run.")
            return

        limit = max(1, concurrency)
        if label:
            print(label)
        start = time.time()
        semaphore = asyncio.Semaphore(limit)
        tasks = [asyncio.create_task(self._run_with_semaphore(semaphore, triplet)) for triplet in self.case_triplet]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await task
        if label:
            print("duration: ", time.time() - start)
        # 所有case完成后写入总成本摘要
        await self.write_total_cost()

    async def _run_with_semaphore(self, semaphore, triplet):
        async with semaphore:
            plaintiff, defendant, judge = triplet
            await self._civil_prediction(plaintiff, defendant, judge)
        
        
    async def save_dialog_info(self, dialog_info):
        if self.write_lock is None:
            self.write_lock = asyncio.Lock()
        async with self.write_lock:
            await asyncio.to_thread(self._write_dialog_info, dialog_info)

    def _write_dialog_info(self, dialog_info):
        with jsonlines.open(self.save_path, "a") as f:
            f.write(dialog_info)
    
    async def _civil_prediction(self, plaintiff, defendant, judge):
        # 为当前case创建独立的CostTracker实例，避免异步环境下的数据覆盖
        case_id = plaintiff.id
        cost_tracker = CostTracker()
        
        # 将cost_tracker设置到所有agent的engine中
        plaintiff.engine.cost_tracker = cost_tracker
        defendant.engine.cost_tracker = cost_tracker
        judge.engine.cost_tracker = cost_tracker
        
        dialog_history = [{"turn": 0, "role": "Judge", "content": judge.judge_greetings}]
        print("############### Dialog ###############")
        print("--------------------------------------")
        print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
        print(dialog_history[-1]["content"])
        
        for turn in range(self.max_conversation_turn):
            judge_response = await self._call_agent(judge, dialog_history[-1]["content"])
            
            if judge_response is None:
                break
            else:
                judge_response = judge_response.replace('审判长：','')
            dialog_history.append({"turn": turn+1, "role": "Judge", "content": judge_response.replace('对原告说：', '').replace('对被告说：', '')})
            print("--------------------------------------")
            print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
            print(dialog_history[-1]["content"])
            
            dialogue = ''
            for d in dialog_history:
                dialogue += d["role"] + ": " + d["content"] + "\n"
            
            if '对原告说' in judge_response:
                plaintiff_response = await self._call_agent(plaintiff, dialogue)
                dialog_history.append({"turn": turn+1, "role": "Plaintiff's Lawyer", "content": plaintiff_response})
            elif '对被告说' in judge_response:
                defendant_response = await self._call_agent(defendant, dialogue)
                dialog_history.append({"turn": turn+1, "role": "Defendant's Lawyer", "content": defendant_response})
            
            print("--------------------------------------")
            print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
            print(dialog_history[-1]["content"])
            
            if '结束庭审' in judge_response:
                break
        
        # 如果循环因为达到max_conversation_turn而结束，且最后一个发言者不是法官，让法官进行总结
        if dialog_history[-1]["role"] != "Judge":
            dialogue = ''
            for d in dialog_history:
                dialogue += d["role"] + ": " + d["content"] + "\n"
            summary_prompt = "庭审已达到最大轮次，请作为审判长进行总结并结束庭审。"
            judge_summary = await self._call_agent(judge, summary_prompt)
            if judge_summary is not None:
                judge_summary = judge_summary.replace('审判长：', '')
                dialog_history.append({"turn": len(dialog_history), "role": "Judge", "content": judge_summary})
                print("--------------------------------------")
                print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
                print(dialog_history[-1]["content"])
            
        # 获取当前case的成本
        case_cost = cost_tracker.get_total_cost()
        
        dialog_info = {
            "case_id": plaintiff.id,
            'save_path': self.args.save_path,
            "judge": self.args.judge,
            "judge_engine_name": judge.engine.model_path,
            "plaintiff": self.args.plaintiff,
            "plaintiff_engine_name": plaintiff.engine.model_path,
            "defendant": self.args.defendant,
            "defendant_engine_name": defendant.engine.model_path,
            "dialog_history": dialog_history,
            "cost": case_cost  # 记录该case的成本
        }
        print(dialog_info)
        await self.save_dialog_info(dialog_info)
        # 每完成一个case就写入一次该case的成本
        await self.write_case_cost(case_id, case_cost)

    async def _call_agent(self, agent, content):
        if hasattr(agent, "async_speak"):
            return await agent.async_speak(content)
        # loop = asyncio.get_running_loop()
        # speak_fn = getattr(agent, "speak")
        # # 注意：如果使用 run_in_executor，会在不同线程执行
        # # 建议所有agent都实现 async_speak 方法
        # return await loop.run_in_executor(None, speak_fn, content)

    async def write_case_cost(self, case_id: str, cost: float):
        """异步写入单个case的成本，使用锁保证线程安全"""
        if self.write_lock is None:
            self.write_lock = asyncio.Lock()
        async with self.write_lock:
            await asyncio.to_thread(self._write_case_cost, case_id, cost)
    
    def _write_case_cost(self, case_id: str, cost: float):
        cost_path = os.path.join(os.getcwd(), "cost_" + self.model + ".txt")
        with open(cost_path, "a", encoding="utf-8") as f:
            f.write(f"case_{case_id}: {cost:.6f}\n")
        print(f"Case {case_id} cost written to {cost_path}: {cost:.6f}")

    async def write_total_cost(self):
        """异步写入total_cost（所有case的总和），使用锁保证线程安全"""
        if self.write_lock is None:
            self.write_lock = asyncio.Lock()
        async with self.write_lock:
            await asyncio.to_thread(self._write_total_cost)
    
    def _write_total_cost(self):
        # 计算所有case的总成本（从已保存的cost文件中读取，或者从dialog_info中累加）
        # 这里简化处理，只写入一个标记
        cost_path = os.path.join(os.getcwd(), "cost_" + self.model + ".txt")
        with open(cost_path, "a", encoding="utf-8") as f:
            f.write("--- Total cost summary ---\n")
        print(f"Total cost summary written to {cost_path}")
            