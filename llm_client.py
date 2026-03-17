"""
LLM 客户端模块 - 使用同步 OpenAI 接口
"""
import os
from typing import List, Dict, Optional, Any
from openai import OpenAI


class LLMClient:
    """同步 OpenAI LLM 客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gemini-3-pro-preview",
        temperature: float = 0.7,
        max_completion_tokens: int = 2000
    ):
        """
        Args:
            api_key: OpenAI API Key，如果不提供则从环境变量 OPENAI_API_KEY 读取
            base_url: API 基础 URL，用于兼容其他 OpenAI 兼容的 API
            model: 模型名称
            temperature: 温度参数
            max_completion_tokens: 最大生成 token 数
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE")        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        
        # 简单的成本统计（单位：美元），按模型的每 1K token 单价估算
        # 价格可根据实际情况自行调整
        self._pricing: Dict[str, Dict[str, float]] = {
            # 参考 OpenAI 公布的 gpt-4o-mini 价格（示例），单位：$/1K tokens
            "gemini-3-flash-preview": {
                "prompt": 0.0005,      # 输入 token 单价
                "completion": 0.003,  # 输出 token 单价
            },
            "gemini-3-pro-preview": {
                "prompt": 0.002,      # 输入 token 单价
                "completion": 0.012,  # 输出 token 单价
            },
            "gpt-5-mini": {
                'prompt': 0.00025,
                'completion': 0.002
            },
            "deepseek-v3.2": {
                'prompt': 0.000284,
                'completion': 0.000426
            },
            "deepseek-chat": {
                'prompt': 0.000284,
                'completion': 0.000426
            },
            "qwen3-next-80b-a3b-instruct": {
                "prompt": 0.000143,
                "completion": 0.000572,
            },
        }
        self.total_cost: float = 0.0
    
    def _record_cost(self, usage: Any) -> None:
        """
        根据 OpenAI 返回的 usage 记录并打印本次调用成本和累计成本。
        """
        if not usage:
            return
        
        pricing = self._pricing.get(self.model)
        if not pricing:
            # 未配置该模型的价格时不做统计
            raise ValueError(f"未配置该模型的价格: {self.model}")
            return
        
        # usage 在新 SDK 中通常是一个对象，也可能是 dict，兼容两种访问方式
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        if prompt_tokens is None and isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
        
        completion_tokens = getattr(usage, "completion_tokens", None)
        if completion_tokens is None and isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens", 0)
        
        prompt_tokens = prompt_tokens 
        completion_tokens = completion_tokens 
        total_tokens = prompt_tokens + completion_tokens
        
        # 根据单价估算本次调用成本
        cost = (
            prompt_tokens * pricing["prompt"]
            + completion_tokens * pricing["completion"]
        ) / 1000.0
        
        self.total_cost += cost
        
        print(
            f"[LLM Cost] model={self.model}, "
            f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, "
            f"total_tokens={total_tokens}, "
            f"call_cost=${cost:.6f}, accumulated_cost=${self.total_cost:.6f}"
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = 'normal',
    ) -> str:
        """
        发送聊天消息
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
            response_format: response_format
            
        Returns:
            模型回复内容
        """
        try:
            if response_format == 'normal':
                if self.model == "gemini-3-pro-preview":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_completion_tokens,
                        reasoning_effort="high",
                    )
                elif self.model == "gemini-3-flash-preview":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_completion_tokens,
                    )
                elif self.model == "gpt-5-mini":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_completion_tokens,
                        reasoning_effort='low',
                    )
                elif self.model == "deepseek-v3.2":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_completion_tokens,
                    )
                elif self.model == "deepseek-chat":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_completion_tokens,
                    )
                elif self.model == "qwen3-next-80b-a3b-instruct":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_completion_tokens,
                    )
            elif response_format == 'json_object':
                if self.model == "gemini-3-pro-preview":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_completion_tokens,
                        reasoning_effort="high",
                        response_format={"type": "json_object"},
                    )
                elif self.model == "gemini-3-flash-preview":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_completion_tokens,
                        response_format={"type": "json_object"},
                    )
                elif self.model == "gpt-5-mini":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_completion_tokens,
                        reasoning_effort='low',
                        response_format={"type": "json_object"},
                    )
                elif self.model == "deepseek-v3.2":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_completion_tokens,
                        response_format={"type": "json_object"},
                    )
                elif self.model == "deepseek-chat":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_completion_tokens,
                        response_format={"type": "json_object"},
                    )
                elif self.model == "qwen3-next-80b-a3b-instruct":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_completion_tokens,
                        response_format={"type": "json_object"},
                    )
            else:
                raise ValueError(f"不支持的 response_format: {response_format}")
            usage = getattr(response, "usage", None)
            self._record_cost(usage)
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM chat error: {e}")
            raise
    



