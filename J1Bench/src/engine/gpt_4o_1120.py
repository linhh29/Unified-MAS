import asyncio
import os
import time

import openai
from openai import AsyncOpenAI, OpenAI

from utils.register import register_class
from utils.cost_tracker import CostTracker
from .base_engine import Engine
from typing import Optional


@register_class(alias="Engine.GPT4o_1120")
class GPT_1120Engine(Engine):
   
    OPENAI_PRICING = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-5-mini": {
            'input': 0.00025,
            'output': 0.002
        },
        "gemini-3-flash-preview": {
            'input': 0.0005,
            'output': 0.003
        },
        "deepseek-v3.2": {
            'input': 0.000284,
            'output': 0.000426
        },
        "qwen3-30b-a3b-instruct-2507": {
            'input': 0.0001065,
            'output': 0.000426
        },
    }

    def __init__(
        self, 
        openai_api_key, openai_api_base, openai_model_name="gpt-5-mini", 
        temperature=0.0, max_tokens=4096, max_async_requests=50,
        cost_tracker: Optional[CostTracker] = None,
        ):
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        assert openai_api_key is not None, "openai_api_key is required" 
        openai_api_base = os.environ.get("OPENAI_API_BASE")
        
        self.model_path = openai_model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.INPUT_COST_PER_MTOKEN = self.OPENAI_PRICING[openai_model_name]["input"]
        self.OUTPUT_COST_PER_MTOKEN = self.OPENAI_PRICING[openai_model_name]["output"]
        
        if openai_api_base is not None:
            print(f"Using OpenAI API base: {openai_api_base}")
            print(f"Using OpenAI API key: {openai_api_key}")
            print(f"Using OpenAI model name: {openai_model_name}")
        #     self.client = OpenAI(
        #         api_key=openai_api_key,
        #         base_url=openai_api_base
        #     )
        # else:
        # print(f"Using OpenAI API key: {openai_api_key}")
        # self.client = OpenAI(
        #     api_key=openai_api_key
        # )
        self.async_client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base
        )
        # print('111111')
        self.max_async_requests = max_async_requests
        self._async_semaphore = None
        self.cost_tracker = cost_tracker  # Each case has its own cost tracker
            
    # def get_response(self, messages, flag=1):
    #     while True:
    #         try:
    #             model_path = self.model_path
    #             i = 0
    #             while i < 10:
    #                 try:
    #                     response = self.client.chat.completions.create(
    #                         model=model_path,
    #                         messages=messages,
    #                         temperature=self.temperature,
    #                         max_tokens=self.max_tokens,
    #                     )
    #                     break
    #                 except openai.BadRequestError:
    #                     time.sleep(10)
    #                     i += 1
    #                 except openai.RateLimitError:
    #                     time.sleep(60)
    #                     i += 1
    #                 except Exception as e:
    #                     print(e)
    #                     i += 1
    #                     time.sleep(10)
    #                     continue
    #             self._record_usage_cost(response)
    #             return response.choices[0].message.content

    #         except:
    #             time.sleep(600)

    async def async_get_response(self, messages, flag=1):
        """Asynchronous counterpart with concurrency control."""
        if self._async_semaphore is None:
            self._async_semaphore = asyncio.Semaphore(self.max_async_requests)

        model_path = self.model_path

        async with self._async_semaphore:
            if model_path == 'gpt-5-mini':
                response = await self.async_client.chat.completions.create(
                    model=model_path,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=32768,
                    reasoning_effort='low',
                )
            elif model_path == 'gemini-3-flash-preview':
                response = await self.async_client.chat.completions.create(
                    model=model_path,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=32768
                )
            elif model_path == 'deepseek-v3.2':
                response = await self.async_client.chat.completions.create(
                    model=model_path,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=32768
                )
            elif model_path == 'qwen3-30b-a3b-instruct-2507':
                response = await self.async_client.chat.completions.create(
                    model=model_path,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=32768
                )

        self._record_usage_cost(response)
        return response.choices[0].message.content

    def _record_usage_cost(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = self._extract_usage_value(usage, "prompt_tokens")
        completion_tokens = self._extract_usage_value(usage, "completion_tokens")
        cost = (prompt_tokens / 1000) * self.INPUT_COST_PER_MTOKEN
        cost += (completion_tokens / 1000) * self.OUTPUT_COST_PER_MTOKEN
        
        # Use the case-specific cost tracker if provided
        if self.cost_tracker is not None:
            self.cost_tracker.add_cost(cost)

    @staticmethod
    def _extract_usage_value(usage, key: str) -> int:
        if usage is None:
            return 0
        if hasattr(usage, key):
            return getattr(usage, key) or 0
        if isinstance(usage, dict):
            return usage.get(key, 0) or 0
        return 0