"""
工具函数模块
"""
import re
from pathlib import Path
from typing import List, Dict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from typing import Dict, Any
import os
from typing import Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from Unified_MAS.prompts import get_debug_prompt, get_node_optimization_prompt
import json
import copy
import math
import threading

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

DeepFund_INSTRUCTION = """You are a portfolio manager making trading decisions based on the provided reference information.

Reference Information:
{text}

Query: {query}

You must provide your decision as a structured output with only the following field:
- action: One of ["Buy", "Sell", "Hold"]

Return ONLY a valid JSON object with the key "action". Your response should be well-reasoned and consider all aspects of the reference information (e.g., company news, insider trades, policy news, technical data, current positions, and account state).
"""


def _format_deepfund_reference(example: dict) -> tuple:
    """Build reference text and query from a deepfund example. Exclude answer/trade_price/final_price so model does not see them. Returns (reference_text, query)."""
    obj = {k: v for k, v in example.items() if k not in ("answer", "trade_price", "final_price", "id")}
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    query = f"Based on the above reference information, what is your trading decision for {example.get('ticker', '')} on {example.get('trading_date', '')}? Provide action (Buy/Sell/Hold) only."
    return text, query



def normalize_arxiv_url(url: str) -> str:
    """
    将 arxiv.org/html 替换为 arxiv.org/abs，统一处理 arxiv 链接。
    
    Args:
        url: 原始URL
        
    Returns:
        规范化后的URL
    """
    if 'arxiv.org/html' in url:
        url = url.replace('arxiv.org/html', 'arxiv.org/abs')
    return url


def sanitize_filename(title: str, max_length: int = 200) -> str:
    """
    将title转换为安全的文件名。
    
    Args:
        title: 原始标题
        max_length: 最大长度
        
    Returns:
        清理后的文件名
    """
    if not title:
        return "untitled"
    
    # 移除或替换不安全的字符
    filename = "".join(c if c.isalnum() or c in "._- " else "_" for c in title)
    # 移除多余的空格和下划线
    filename = "_".join(filename.split())
    # 限制长度
    filename = filename[:max_length]
    # 移除开头和结尾的下划线
    filename = filename.strip("_")
    
    if not filename:
        filename = "untitled"
    
    return filename


def find_pdf_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    """
    在HTML页面中查找PDF下载链接。
    
    Args:
        soup: BeautifulSoup解析的HTML对象
        base_url: 基础URL，用于构建绝对URL
        
    Returns:
        PDF链接列表
    """
    pdf_links = []
    
    # 查找所有可能的PDF链接
    # 1. 直接链接到PDF文件的<a>标签
    for link in soup.find_all('a', href=True):
        href = link.get('href', '').lower()
        if href.endswith('.pdf') or 'pdf' in href or 'download' in href.lower():
            # 构建绝对URL
            if href.startswith('http'):
                pdf_links.append(href)
            elif href.startswith('/'):
                pdf_links.append(urljoin(base_url, href))
            else:
                pdf_links.append(urljoin(base_url, href))
    
    # 2. 查找包含"pdf"、"download"等关键词的链接
    for link in soup.find_all('a', href=True):
        text = link.get_text().lower()
        href = link.get('href', '')
        if ('pdf' in text or 'download' in text) and href:
            if href.startswith('http'):
                pdf_links.append(href)
            elif href.startswith('/'):
                pdf_links.append(urljoin(base_url, href))
    
    # 3. 查找meta标签中的PDF链接
    for meta in soup.find_all('meta', property=True):
        if 'pdf' in meta.get('property', '').lower() or 'pdf' in meta.get('content', '').lower():
            content = meta.get('content', '')
            if content and content.endswith('.pdf'):
                pdf_links.append(content)
    
    # 去重并返回
    return list(set(pdf_links))


def download_pdf(pdf_url: str, save_path: Path, headers: Dict[str, str], timeout: int = 30) -> bool:
    """
    下载PDF文件。
    
    Args:
        pdf_url: PDF文件的URL
        save_path: 保存路径
        headers: HTTP请求头
        timeout: 超时时间
        
    Returns:
        是否下载成功
    """
    try:
        response = requests.get(pdf_url, headers=headers, timeout=timeout, stream=True)
        if response.status_code == 200:
            # 检查Content-Type是否为PDF
            content_type = response.headers.get('Content-Type', '').lower()
            if 'pdf' in content_type or pdf_url.lower().endswith('.pdf'):
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
    except Exception as e:
        print(f"    [PDF Download Error] {e}")
    return False


def is_github_url(url: str) -> bool:
    """
    检查URL是否是GitHub链接。
    
    Args:
        url: 要检查的URL
        
    Returns:
        是否是GitHub URL
    """
    return 'github.com' in url.lower()


def parse_github_url(url: str) -> Dict[str, str]:
    """
    解析GitHub URL，提取仓库信息。
    
    Args:
        url: GitHub URL
        
    Returns:
        包含 owner, repo, path, file_type 的字典
    """
    # 移除可能的查询参数和锚点
    url = url.split('?')[0].split('#')[0]
    
    # 匹配 GitHub URL 模式
    # https://github.com/owner/repo
    # https://github.com/owner/repo/tree/branch/path
    # https://github.com/owner/repo/blob/branch/path/to/file
    pattern = r'github\.com/([^/]+)/([^/]+)(?:/(?:tree|blob)/([^/]+)(?:/(.+))?)?'
    match = re.search(pattern, url)
    
    if match:
        owner = match.group(1)
        repo = match.group(2)
        branch = match.group(3) or 'main'  # 默认为 main
        path = match.group(4) or ''
        
        # 判断是文件还是目录
        file_type = 'file' if '/blob/' in url else 'directory' if '/tree/' in url else 'repo'
        
        return {
            'owner': owner,
            'repo': repo,
            'branch': branch,
            'path': path,
            'file_type': file_type,
            'raw_url': f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}" if path else None
        }
    
    return {}

def build_question_and_answer(dataset, example) -> tuple[str, str]:
    """Build the single CoT question prompt and ground-truth answer."""
    if "hosp_summ" in dataset:
        patient_info = example["instruct"]

        base_prompt = HospSumm_INSTRUCTION.format(
            patient_info=patient_info
        )
        # Explicit CoT instruction + final "Answer:" line for easier extraction
        cot_suffix = (
            # "\n\nPlease think step by step and explain your reasoning. "
            "At the end, provide your final answer in a new line starting with "
            "'Answer:' followed by a summary."
        )
        question = base_prompt + cot_suffix
        answer = example["answer"]
    elif "travelplanner" in dataset:
        query = example["query"]
        ref_info = example["reference_information"]
        base_prompt = PLANNER_INSTRUCTION.format(text=ref_info, query=query)
        cot_suffix = (
            # "\n\nPlease think step by step when constructing the travel plan. "
            "Ensure all details come strictly from the given information."
        )
        question = base_prompt + cot_suffix
        # TravelPlanner uses metric-based scoring, no direct string answer here
        answer = str(example["answer"]) # placeholder, scoring reads ground truth by instance_id
    elif "j1eval" in dataset:
        drop_keys = {"id", "court_information"}
        obj = copy.deepcopy(example)
        for k in drop_keys:
            obj.pop(k, None)
        text = json.dumps(obj, ensure_ascii=False)
        base_prompt = J1_INSTRUCTION.format(text=text)
        cot_suffix = (
            # "\n\n请逐步推理案件事实和法律适用过程，"
            "最后在新的一行以 'Answer:' 开头给出简洁的最终裁判主文。"
        )
        question = base_prompt + cot_suffix
        answer = '\n'.join(example["court_information"]["ground_truth"]["court_judgment"])
    elif "deepfund" in dataset:
        ref_text, query = _format_deepfund_reference(example)
        base_prompt = DeepFund_INSTRUCTION.format(text=ref_text, query=query)
        cot_suffix = (
            "Provide your trading decision as a structured output: action (Buy/Sell/Hold) only. "
            "Return ONLY a valid JSON object with the key 'action'."
        )
        question = base_prompt + cot_suffix
        answer = str(example["answer"]) # placeholder, scoring reads ground truth by instance_id
    elif "healthbench" in dataset:
        prompt_messages = example.get("prompt") or []
        conversation = "\n\n".join(f"{m['role']}: {m['content']}" for m in prompt_messages)
        question = (
            "The following is a multi-turn conversation. Provide the assistant's next response only.\n\n"
            + conversation
        )
        ideal = (example.get("ideal_completions_data") or {}).get("ideal_completion") or ""
        answer = ideal
    elif "aime" in dataset:
        # AIME: 数学题，直接在末尾要求以 Answer: 给出最终数值
        base_prompt = example["question"]
        cot_suffix = (
            "\n\nPlease think step by step. "
            "At the end, provide your final numeric answer in a new line starting with 'Answer:'."
        )
        question = base_prompt + cot_suffix
        answer = str(example["answer"])
    else:
        raise NotImplementedError

    return question, answer

# # 设置 CUDA 可见设备
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# model_path = "xx/Qwen3-Next-80B-A3B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

# PyTorch/CUDA 模型非线程安全，多线程并行时需串行化 get_j_tilde 调用，否则会死锁
_j_tilde_lock = threading.Lock()

def get_j_tilde(
    context: str,
    answer: str,
) -> float:
    """
    获取上下文文本和答案文本的 J_tilde
    
    Args:
        context: 输入上下文文本
        answer: 输入答案文本
        
    Returns:
        J_tilde 值（float）
    """
    with _j_tilde_lock:
        model.eval()  # 设置为评估模式
        # 编码输入
        device = next(model.parameters()).device
        context_inputs = tokenizer(
            context,
            return_tensors="pt",
        ).input_ids.to(device)

        answer_inputs = tokenizer(
            answer,
            return_tensors="pt",
        ).input_ids.to(device)

        input_ids = torch.cat([context_inputs, answer_inputs], dim=1)
        # 前向传播获取 logits
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        context_len = context_inputs.shape[1]
        target_logits = logits[:, context_len-1 : -1, :]
        target_labels = answer_inputs
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_nlls = loss_fct(
            target_logits.reshape(-1, target_logits.size(-1)),
            target_labels.reshape(-1)
        )
        j_tilde = -token_nlls.mean().item()

    return j_tilde

def get_node_score(j_history, w):
    """
    获取节点得分
    """
    baseline_j = j_history[0]
    current_j = j_history[-1]

    # s magnitude
    delta = (current_j - baseline_j) / (baseline_j + 1e-9)
    s_magnitude = (math.tanh(delta + 1) + 1) / 2

    # s stability
    n = len(j_history)
    score_sum = 0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            val_diff = j_history[j] - j_history[i]
            if val_diff > 0:
                sign = 1
            elif val_diff < 0:
                sign = -1
            else:
                sign = 0
            score_sum += sign
            pair_count += 1
    if pair_count == 0:
        s_stability = 0
    else:
        s_stability = (score_sum / (2*pair_count)) + 0.5

    node_score = (1 - w) * s_magnitude + w * s_stability
    return node_score



def create_pipeline_executor(nodes_data, executor_llm_client, search_engine, meta_llm_client, dataset_name=None, num_epochs: int = 1) -> Any:
    """
    创建 PipelineExecutor 类并动态添加所有节点方法和 execute_pipeline 方法
    
    Args:
        nodes_data: 从 generated_nodes.json 加载的数据
        executor_llm_client: Executor LLM 客户端实例（用于节点执行）
        search_engine: 搜索引擎实例
        meta_llm_client: Meta LLM 客户端实例（用于 debug）
        dataset_name: 数据集名称，用于保存优化结果
        num_epochs: 运行的 epoch 数目；每 epoch 跑完 validation set 后按平均 reward 选节点优化
    
    Returns:
        PipelineExecutor 实例
    """
    # 创建 PipelineExecutor 类
    class PipelineExecutor:
        def __init__(self, executor_llm_client, search_engine, meta_llm_client, nodes_data, dataset_name=None, num_epochs=1):
            self.llm_client = executor_llm_client  # executor-model，用于节点执行
            self.search_engine = search_engine
            self.meta_llm_client = meta_llm_client
            self.nodes_data = nodes_data  # 存储节点数据，用于优化
            self.dataset_name = dataset_name  # 存储数据集名称，用于保存优化结果
            self.intermediate_outputs = []  # 存储每个节点的中间输出
            self.num_epochs = num_epochs
            self._optimization_buffer = []  # 缓冲样本，epoch 结束后由 perform_epoch_optimization 处理
    
    # 在类的命名空间中执行所有节点代码
    class_namespace = {}
    
    # 添加所有节点方法
    for node in nodes_data.get('nodes', []):
        node_name = node.get('node_name')
        all_code = node.get('all_code', '')
        
        if node_name and all_code:
            try:
                # 在类的命名空间中执行节点代码
                exec(all_code, {}, class_namespace)
                # 将方法绑定到类
                if node_name in class_namespace:
                    original_method = class_namespace[node_name]
                    
                    # 包装方法以打印输入和输出，并存储中间结果
                    def create_wrapped_node_method(orig_method, name):
                        def wrapped_method(self, *args, **kwargs):
                            print(f"\n{'='*80}")
                            print(f"[Node Execution] {name}")
                            print(f"{'='*80}")
                            
                            # 执行原始方法
                            result = orig_method(self, *args, **kwargs)
                            
                            # 存储中间输出（将内容展开为可读文本）
                            def format_output(obj, indent=0):
                                """将输出对象格式化为可读文本"""
                                indent_str = "  " * indent
                                if isinstance(obj, dict):
                                    lines = []
                                    for key, value in obj.items():
                                        if isinstance(value, (dict, list)):
                                            lines.append(f"{indent_str}{key}:")
                                            lines.append(format_output(value, indent + 1))
                                        else:
                                            lines.append(f"{indent_str}{key}: {value}")
                                    return "\n".join(lines)
                                elif isinstance(obj, list):
                                    lines = []
                                    for i, item in enumerate(obj):
                                        if isinstance(item, (dict, list)):
                                            lines.append(f"{indent_str}[{i}]:")
                                            lines.append(format_output(item, indent + 1))
                                        else:
                                            lines.append(f"{indent_str}[{i}]: {item}")
                                    return "\n".join(lines)
                                else:
                                    return f"{indent_str}{obj}"
                            
                            # 格式化输出内容
                            if isinstance(result, (dict, list)):
                                # result_str = format_output(result)
                                result_str = json.dumps(result, ensure_ascii=False, indent=2)
                            else:
                                result_str = str(result)
                            
                            # 存储节点名称和输出
                            self.intermediate_outputs.append({
                                'node_name': name,
                                'output': result_str
                            })
                            
                            # print(f"\n[Node Output] {name}")
                            # if isinstance(result, (dict, list)):
                            #     print(f"Output (formatted): {json.dumps(result, ensure_ascii=False, indent=2)[:1000]}...")
                            # else:
                            #     print(f"Output: {result}")
                            # print(f"{'='*80}\n")
                            
                            return result
                        return wrapped_method
                    
                    wrapped_method = create_wrapped_node_method(original_method, node_name)
                    setattr(PipelineExecutor, node_name, wrapped_method)
                    print(f"  ✓ Loaded node: {node_name} (with logging)")
                else:
                    print(f"  ✗ Warning: Node {node_name} not found after execution")
            except Exception as e:
                print(f"  ✗ Warning: Failed to load node {node_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 添加 execute_pipeline 方法
    connections_code = nodes_data.get('Connections', '')
    if connections_code:
        try:
            # 在类的命名空间中执行 Connections 代码
            exec(connections_code, {}, class_namespace)
            if 'execute_pipeline' in class_namespace:
                original_execute_pipeline = class_namespace['execute_pipeline']
                
                # 包装 execute_pipeline 方法以打印输入和输出，并计算 j_tilde
                def wrapped_execute_pipeline(self, initial_input_data, initial_answer_data, sample_index=None):
                    print(f"\n{'#'*80}")
                    print(f"[Pipeline Execution] Starting pipeline")
                    print(f"{'#'*80}")
                    print(f"Input question:")
                    print(initial_input_data)
                    print(f"Input answer:")
                    print(initial_answer_data)
                    print(f"{'#'*80}\n")
                    
                    # 清空中间输出列表
                    self.intermediate_outputs = [{'node_name': 'Initial Input', 'output': 'Question: ' + initial_input_data}]
                    self.j_tilde_values = []
                    self.node_scores = []
                    self.node_rewards = []
                    self.alpha = 0.1
                    self.w = 0.6
                    
                    # 执行原始的 execute_pipeline
                    result = original_execute_pipeline(self, initial_input_data)
 
                    # 计算 j_tilde
                    context = ''
                    for intermediate in self.intermediate_outputs:
                        context = context + '\n' + intermediate['output']
                        j_tilde = get_j_tilde(context, initial_answer_data)

                        if len(self.j_tilde_values) > 0:
                            last_j_tilde = self.j_tilde_values[-1]
                            j_tilde = self.alpha * last_j_tilde + (1 - self.alpha) * j_tilde

                        self.j_tilde_values.append(j_tilde)
                        node_score = get_node_score(self.j_tilde_values, self.w)
                        self.node_scores.append(node_score)

                        if len(self.node_rewards) == 0:
                            node_reward = 0
                        else:
                            node_reward = self.node_scores[-1] - self.node_scores[-2]
                        self.node_rewards.append(node_reward)
                        print(f"Node: {intermediate['node_name']}, J_tilde: {j_tilde}, Node score: {node_score}, Node reward: {node_reward}")

                    # 构建 buffer_entry，由调用方收集后传给 perform_epoch_optimization（支持并行时主进程汇总）
                    buffer_entry = {
                        'sample_index': sample_index,
                        'question': initial_input_data,
                        'answer': initial_answer_data,
                        'intermediate_outputs': copy.deepcopy(self.intermediate_outputs),
                        'node_rewards': copy.deepcopy(self.node_rewards),
                    }
                    return result, buffer_entry
                
                setattr(PipelineExecutor, 'execute_pipeline', wrapped_execute_pipeline)
                
                def perform_epoch_optimization(self, buffer=None, epoch=None):
                    """
                    Epoch 模式：跑完整个 validation set 后调用。
                    按 node 求平均 reward，选平均 reward 最小的 node 进行更新；
                    更新时只使用「该 node 在该样本上 reward 最低」的样本（即对应要更新该 node 的样本）。
                    Args:
                        buffer: 可选，收集的 buffer_entry 列表；不传则用 self._optimization_buffer（兼容旧逻辑）
                        epoch: 当前 epoch 编号（从 1 开始），用于命名保存文件，与 validate_results_epoch_X.jsonl 一致
                    Returns:
                        bool: 是否执行了优化
                    """
                    buf_list = buffer if buffer is not None else self._optimization_buffer
                    if len(buf_list) == 0:
                        return False
                    
                    # 1. 按 node 求平均 reward（每个样本都有所有 node 的 reward）
                    node_reward_sum = {}  # node_name -> sum
                    node_reward_count = {}  # node_name -> count
                    for buf in buf_list:
                        outs = buf['intermediate_outputs']
                        rewards = buf['node_rewards']
                        for i, (out, r) in enumerate(zip(outs, rewards)):
                            if i >= 1 and out.get('node_name') != 'Initial Input':
                                key = out['node_name']
                                node_reward_sum[key] = node_reward_sum.get(key, 0) + r
                                node_reward_count[key] = node_reward_count.get(key, 0) + 1
                    
                    # 求平均
                    avg_rewards = {}
                    for k in node_reward_sum:
                        cnt = node_reward_count.get(k, 1)
                        avg_rewards[k] = node_reward_sum[k] / cnt if cnt > 0 else 0
                    
                    candidates = [(k, v) for k, v in avg_rewards.items() if k != 'Initial Input']
                    if not candidates:
                        return False
                    
                    target_node_name = min(candidates, key=lambda x: x[1])[0]
                    min_avg_reward = min(candidates, key=lambda x: x[1])[1]
                    
                    # 2. 筛选：只保留「该样本上 target_node 的 reward 是该样本所有 node 中最低」的样本
                    filtered_buffer = []
                    for buf in buf_list:
                        outs = buf['intermediate_outputs']
                        rewards = buf['node_rewards']
                        # 找到该样本上 reward 最低的 node
                        node_to_reward = {}
                        for i, (out, r) in enumerate(zip(outs, rewards)):
                            if i >= 1 and out.get('node_name') != 'Initial Input':
                                node_to_reward[out['node_name']] = r
                        if not node_to_reward:
                            continue
                        sample_min_node = min(node_to_reward.items(), key=lambda x: x[1])[0]
                        if sample_min_node == target_node_name:
                            filtered_buffer.append(buf)
                    
                    # 若没有样本「对应」该 node，则退化为使用全部样本
                    if len(filtered_buffer) == 0:
                        filtered_buffer = list(buf_list)
                        print(f"[Epoch Optimization] No sample with {target_node_name} as worst node; using all {len(filtered_buffer)} samples")
                    else:
                        print(f"[Epoch Optimization] Filtered to {len(filtered_buffer)} samples that correspond to node {target_node_name} (worst for those samples)")
                    
                    # 3. 使用 filtered_buffer 进行优化（复用原有逻辑）
                    target_node = None
                    for node in self.nodes_data.get('nodes', []):
                        if node.get('node_name') == target_node_name:
                            target_node = node
                            break
                    
                    if not target_node:
                        print(f"[Epoch Optimization] Warning: Could not find node {target_node_name}")
                        if buffer is None:
                            self._optimization_buffer = []
                        return False
                    
                    node_type = target_node.get('node_type', '')
                    node_description = target_node.get('description', '')
                    node_implementation = target_node.get('implementation', {})
                    node_all_code = target_node.get('all_code', '')
                    
                    combined_qa_parts = []
                    for buf in filtered_buffer:
                        sidx = buf['sample_index']
                        q, a = buf['question'], buf['answer']
                        combined_qa_parts.append(
                            f"[Sample {sidx}]\nQuestion: {q}\nExpected Answer: {a}"
                        )
                    combined_question = "\n\n".join(combined_qa_parts)
                    combined_answer = "(Expected Answer is included per sample above)"
                    
                    intermediate_for_prompt = []
                    for buf in filtered_buffer:
                        sidx = buf['sample_index']
                        outs = buf['intermediate_outputs']
                        intermediate_for_prompt.append({
                            'node_name': f'[Sample {sidx}]',
                            'output': f'--- Intermediate outputs for the sample with Question/Answer shown in Task Context under [Sample {sidx}] ---'
                        })
                        for out in outs:
                            intermediate_for_prompt.append({
                                'node_name': out['node_name'],
                                'output': out.get('output', ''),
                            })
                    
                    max_optimization_attempts = 3
                    optimization_success = False
                    
                    for attempt in range(1, max_optimization_attempts + 1):
                        try:
                            print(f"[Epoch Optimization] Attempt {attempt}/{max_optimization_attempts}: Optimizing node {target_node_name} (avg reward: {min_avg_reward:.4f})")
                            opt_system, opt_user = get_node_optimization_prompt(
                                question=combined_question,
                                answer=combined_answer,
                                node_name=target_node_name,
                                node_type=node_type,
                                node_description=node_description,
                                node_implementation=node_implementation,
                                node_all_code=node_all_code,
                                intermediate_outputs=intermediate_for_prompt,
                                node_reward=min_avg_reward,
                                node_index=1
                            )
                            opt_messages = [
                                {"role": "system", "content": opt_system},
                                {"role": "user", "content": opt_user}
                            ]
                            opt_response_str = self.meta_llm_client.chat(opt_messages, response_format='json_object')
                            opt_response = json.loads(opt_response_str)
                            
                            optimized_impl = opt_response.get('optimized_implementation', {})
                            if optimized_impl:
                                if 'prompt_template' in optimized_impl and optimized_impl['prompt_template']:
                                    node_implementation['prompt_template'] = optimized_impl['prompt_template']
                                if 'code_snippet' in optimized_impl and optimized_impl['code_snippet']:
                                    node_implementation['code_snippet'] = optimized_impl['code_snippet']
                                if 'tools_needed' in optimized_impl and optimized_impl['tools_needed']:
                                    node_implementation['tools_needed'] = optimized_impl['tools_needed']
                                if 'logic_description' in optimized_impl and optimized_impl['logic_description']:
                                    node_implementation['logic_description'] = optimized_impl['logic_description']
                            
                            optimized_code = opt_response.get('optimized_all_code', '')
                            if optimized_code:
                                target_node['all_code'] = optimized_code
                            target_node['implementation'] = node_implementation
                            
                            base_dir = Path(__file__).parent / "intermediate_result" / self.dataset_name / "optimize" / "rounds"
                            base_dir.mkdir(parents=True, exist_ok=True)
                            sample_ids_str = "_".join(str(b["sample_index"]) for b in filtered_buffer)
                            if epoch is not None:
                                output_file = base_dir / f"epoch_{epoch}_generated_nodes.json"
                            else:
                                output_file = base_dir / f"epoch_round_{sample_ids_str}_generated_nodes.json"
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(self.nodes_data, f, ensure_ascii=False, indent=2)
                            
                            all_node_rewards = [{"node_name": k, "avg_reward": v} for k, v in avg_rewards.items()]
                            round_info = {
                                "round": sample_ids_str,
                                "samples": [b["sample_index"] for b in filtered_buffer],
                                "optimized_node": target_node_name,
                                "all_node_avg_rewards": all_node_rewards
                            }
                            info_file = base_dir / "optimization_info.json"
                            all_optimization_info = {"rounds": []}
                            if info_file.exists():
                                with open(info_file, 'r', encoding='utf-8') as f:
                                    all_optimization_info = json.load(f)
                            all_optimization_info["rounds"].append(round_info)
                            with open(info_file, 'w', encoding='utf-8') as f:
                                json.dump(all_optimization_info, f, ensure_ascii=False, indent=2)
                            
                            optimization_success = True
                            break
                        except Exception as e:
                            print(f"[Epoch Optimization] Attempt {attempt}: Error: {e}")
                            if attempt < max_optimization_attempts:
                                print(f"[Epoch Optimization] Retrying...")
                    
                    if buffer is None:
                        self._optimization_buffer = []
                    return optimization_success
                
                setattr(PipelineExecutor, 'perform_epoch_optimization', perform_epoch_optimization)
                print(f"  ✓ Loaded execute_pipeline method (with logging)")
            else:
                print(f"  ✗ Warning: execute_pipeline not found after execution")
        except Exception as e:
            print(f"  ✗ Warning: Failed to load execute_pipeline: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ✗ Warning: No Connections code found in nodes_data")
    
    # 添加 debug 方法
    def debug_pipeline(self, sample_input: Dict, nodes_data: Dict, max_iterations: int = 5) -> tuple[bool, Dict[str, Any], bool]:
        """
        调试 pipeline，自动修复错误
        
        Args:
            sample_input: 第一个样本输入
            nodes_data: 节点数据（用于更新）
            max_iterations: 最大调试次数
            
        Returns:
            (success: bool, fixed_nodes_data: Dict, was_fixed: bool)
            was_fixed: 是否实际调用了 LLM 进行修复（True 表示有 bug 并进行了修复）
        """
        import traceback
        was_fixed = False
        
        for iteration in range(max_iterations):
            print(f"\n{'='*80}")
            print(f"Debug iteration {iteration + 1}/{max_iterations}")
            print(f"{'='*80}")
            
            try:
                # 尝试执行 pipeline
                # 在 debug 过程中，不需要真实的答案，传递空字符串即可
                # sample_index=None 表示不进行节点优化（debug 时不需要优化）
                result, _ = self.execute_pipeline(sample_input, 'debug', sample_index=0)
                print(f"\n✓ Pipeline executed successfully!")
                print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)[:500]}...")
                
                # 成功，返回（未调用 LLM 修复时 was_fixed=False）
                return True, nodes_data, was_fixed
                
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                error_traceback = traceback.format_exc()
                
                print(f"\n✗ Error occurred: {error_type}: {error_msg}")
                print(f"Traceback:\n{error_traceback}")
                
                # 收集错误信息
                error_info = f"""
Error Type: {error_type}
Error Message: {error_msg}
Traceback:
{error_traceback}
"""
                
                # 找出出错的节点（从堆栈跟踪中）
                error_node = None
                error_node_code = None
                for node in nodes_data.get('nodes', []):
                    node_name = node.get('node_name', '')
                    if node_name in error_traceback:
                        error_node = node
                        error_node_code = node.get('all_code', '')
                        break
                
                # 如果找不到具体节点，尝试从错误信息推断
                if not error_node:
                    # 检查是否是 Connections 的问题
                    if 'execute_pipeline' in error_traceback:
                        error_node_code = nodes_data.get('Connections', '')
                    else:
                        # 默认修复第一个节点
                        error_node = nodes_data.get('nodes', [{}])[0] if nodes_data.get('nodes') else None
                        if error_node:
                            error_node_code = error_node.get('all_code', '')
                
                connections_code = nodes_data.get('Connections', '')
                
                # 调用 LLM 修复代码（使用 meta-model）
                print(f"\n[Debug] Calling LLM (meta-model) to fix the error...")
                debug_system, debug_user = get_debug_prompt(
                    error_info, error_node_code or '', connections_code, sample_input
                )
                
                debug_messages = [
                    {"role": "system", "content": debug_system},
                    {"role": "user", "content": debug_user}
                ]
                
                fix_response_str = self.meta_llm_client.chat(debug_messages, response_format='json_object')
                fix_response = json.loads(fix_response_str)
                
                print(f"[Debug] LLM provided fix explanation: {fix_response.get('explanation', 'N/A')}")
                
                # 更新代码
                fixed_node_code = fix_response.get('fixed_node_code', '')
                fixed_connections_code = fix_response.get('fixed_connections_code', '')
                
                if fixed_node_code and error_node:
                    error_node['all_code'] = fixed_node_code
                    print(f"[Debug] Updated node code: {error_node.get('node_name', 'Unknown')}")
                    was_fixed = True
                
                if fixed_connections_code:
                    nodes_data['Connections'] = fixed_connections_code
                    print(f"[Debug] Updated Connections code")
                    was_fixed = True
                
                # 重新创建 executor 并更新方法（使用 executor-model）
                print(f"[Debug] Recreating pipeline executor with fixed code...")
                # 传递 dataset_name 以保持一致性（虽然 debug 时不会用到优化功能）
                new_executor = create_pipeline_executor(
                    nodes_data, self.llm_client, self.search_engine, self.meta_llm_client, self.dataset_name,
                    getattr(self, 'num_epochs', 1)
                )
                
                # 更新当前 executor 的所有方法（包括节点方法和 execute_pipeline）
                for attr_name in dir(new_executor):
                    if not attr_name.startswith('_') and attr_name not in ['llm_client', 'search_engine']:
                        try:
                            attr_value = getattr(new_executor, attr_name)
                            if callable(attr_value):
                                setattr(self, attr_name, attr_value)
                        except:
                            pass
        
        # 达到最大迭代次数，返回失败
        print(f"\n✗ Debug failed after {max_iterations} iterations")
        return False, nodes_data, was_fixed
    
    # 将 debug 方法添加到类
    setattr(PipelineExecutor, 'debug_pipeline', debug_pipeline)
    
    # 创建实例
    executor = PipelineExecutor(
        executor_llm_client, search_engine, meta_llm_client, nodes_data, dataset_name, num_epochs
    )
    return executor





