from functools import partial

from sampler.chat_completion_sampler import ChatCompletionSampler, AsyncChatCompletionSampler
from sampler.o_chat_completion_sampler import OChatCompletionSampler
from sampler.together_completion_sampler import ChatCompletionSampler as ToChatCompletionSampler
from sampler.vllm_completion_sampler import AsyncChatCompletionSampler as VllmChatCompletionSampler

model_init_map = {
    "o3-mini": partial(OChatCompletionSampler, model="o3-mini"),
    "gpt-4o": partial(AsyncChatCompletionSampler, model="gpt-4o"),
    "gpt-5-mini": partial(AsyncChatCompletionSampler, model="gpt-5-mini"),
    "gpt-5": partial(AsyncChatCompletionSampler, model="gpt-5"),
    "gpt-5-nano": partial(AsyncChatCompletionSampler, model="gpt-5-nano"),
    "gemini-3-flash-preview": partial(AsyncChatCompletionSampler, model="gemini-3-flash-preview"),
    "deepseek-v3.2": partial(AsyncChatCompletionSampler, model="deepseek-v3.2"),
    "qwen3-30b-a3b-instruct-2507": partial(AsyncChatCompletionSampler, model="qwen3-30b-a3b-instruct-2507"),
    "qwen-2.5-32b-instr": partial(VllmChatCompletionSampler, model="qwen-2.5-32b-instr"),
    "qwen3-30b-a3b": partial(VllmChatCompletionSampler, model="qwen3-30b-a3b"),
    "qwq-32b": partial(ToChatCompletionSampler, model="Qwen/Qwen2.5-32B-Instruct"),
    "llama-3.3-70b-instr": partial(ToChatCompletionSampler, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "qwen3-235b": partial(ToChatCompletionSampler, model="Qwen/Qwen3-235B-A22B-fp8-tput"),
    "deepseek-v3": partial(ToChatCompletionSampler, model="deepseek-ai/DeepSeek-V3")
}

AVAILABLE_MODELS = {}


def init_model(name: str):
    global AVAILABLE_MODELS
    if name in AVAILABLE_MODELS:
        return
    AVAILABLE_MODELS[name] = model_init_map[name](max_tokens=4096)


def get_model(name):
    global AVAILABLE_MODELS
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {name} is not initialized. Please call init_model('{name}') first.")
    return AVAILABLE_MODELS[name]
