"""
PMC: Planning with Multi-Constraints via Collaborative Language Agents
Implementation based on the paper "Planning with Multi-Constraints via Collaborative Language Agents" at COLING 2025

This implementation includes:
1. Manager Agent: Responsible for task decomposition
2. Executor Agents: Handle specific constraints and subtasks
3. Collaborative planning system for multi-constraint problems
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import openai
from openai import AsyncOpenAI
import argparse
from pathlib import Path
import re
import time
try:
    from tqdm.asyncio import tqdm_asyncio
except ImportError:
    # Fallback if tqdm is not available
    tqdm_asyncio = asyncio
from travel_eval_utils.travelplanner_eval import eval_score

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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./async_results/')
    parser.add_argument('--model',type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    return args

class ConstraintType(Enum):
    """Types of constraints in planning tasks"""
    BUDGET = "budget"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    PREFERENCE = "preference"
    HARD = "hard"
    COMMONSENSE = "commonsense"


@dataclass
class Subtask:
    """Represents a subtask in the planning process"""
    id: int
    description: str
    constraint_types: List[ConstraintType]
    dependencies: List[int]  # IDs of dependent subtasks
    result: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    local_constraints: Optional[List[str]] = None  # Local constraints for this subtask


@dataclass
class Plan:
    """Represents a complete plan"""
    subtasks: List[Subtask]
    final_plan: Optional[str] = None
    constraints_satisfied: bool = False
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CostInfo:
    """Token usage and cost information for a single API call"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0


class CostTracker:
    """Tracks token usage and costs across all API calls"""
    
    # OpenAI pricing per 1K tokens (as of 2024)
    PRICING = {
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
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.call_details: List[Dict[str, Any]] = []
    
    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> CostInfo:
        """Calculate cost for a given model and token usage"""
        # Normalize model name (handle variations)
        model_key = model_name
        for key in self.PRICING.keys():
            if key in model_name or model_name in key:
                model_key = key
                break
        
        # Get pricing (default to gpt-4 if model not found)
        pricing = self.PRICING.get(model_key)
        
        input_cost = (input_tokens / 1000.0) * pricing["input"]
        output_cost = (output_tokens / 1000.0) * pricing["output"]
        total_cost = input_cost + output_cost
        total_tokens = input_tokens + output_tokens
        
        return CostInfo(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
    
    def add_call(self, model_name: str, cost_info: CostInfo, agent_role: str = "unknown"):
        """Add a call to the tracker"""
        self.total_input_tokens += cost_info.input_tokens
        self.total_output_tokens += cost_info.output_tokens
        self.total_tokens += cost_info.total_tokens
        self.total_cost += cost_info.total_cost
        
        self.call_details.append({
            "model": model_name,
            "agent_role": agent_role,
            "input_tokens": cost_info.input_tokens,
            "output_tokens": cost_info.output_tokens,
            "total_tokens": cost_info.total_tokens,
            "input_cost": cost_info.input_cost,
            "output_cost": cost_info.output_cost,
            "total_cost": cost_info.total_cost
        })
    
    def get_summary(self, start_index: int = 0) -> Dict[str, Any]:
        """Get a summary of all costs
        
        Args:
            start_index: Only include call_details from this index onwards (for per-task cost tracking)
        """
        if start_index == 0:
            call_details = self.call_details
            total_input_tokens = self.total_input_tokens
            total_output_tokens = self.total_output_tokens
            total_tokens = self.total_tokens
            total_cost = self.total_cost
        else:
            # Calculate incremental cost from start_index
            call_details = self.call_details[start_index:]
            total_input_tokens = sum(call['input_tokens'] for call in call_details)
            total_output_tokens = sum(call['output_tokens'] for call in call_details)
            total_tokens = sum(call['total_tokens'] for call in call_details)
            total_cost = sum(call['total_cost'] for call in call_details)
        
        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "num_calls": len(call_details),
            "call_details": call_details
        }
    
    def reset(self):
        """Reset the tracker"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.call_details = []


class LLMAgent:
    """Base LLM Agent class for PMC"""
    
    def __init__(self, model_name, api_key: Optional[str] = None, base_url: Optional[str] = None, 
                 role: str = "assistant", temperature: float = 1,
                 cost_tracker: Optional[CostTracker] = None):
        self.model_name = model_name
        self.role = role
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)
        self.cost_tracker = cost_tracker
    
    async def query(self, prompt: str, system_message: Optional[str] = None) -> Tuple[str, CostInfo]:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            if self.model_name == 'gpt-5-mini':
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=32768,
                    reasoning_effort='low',
                )
            elif self.model_name == 'gpt-4o':
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=1,
                )
            elif self.model_name == 'gemini-3-flash-preview':
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=32768
                )
            elif self.model_name == 'deepseek-v3.2':
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=32768
                )
            elif self.model_name == 'qwen3-30b-a3b-instruct-2507':
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=32768
                )
            
            # Extract token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            
            # Calculate cost
            cost_tracker = self.cost_tracker or CostTracker()
            cost_info = cost_tracker.calculate_cost(self.model_name, input_tokens, output_tokens)
            
            # Add to tracker if provided
            if self.cost_tracker:
                self.cost_tracker.add_call(self.model_name, cost_info, self.role)
            
            return response.choices[0].message.content, cost_info
        except Exception as e:
            print(f"Error in LLM query: {e}")
            # Return error with zero cost
            error_cost = CostInfo()
            return f"Error: {str(e)}", error_cost


class ManagerAgent(LLMAgent):
    """Manager Agent responsible for task decomposition"""
    
    def __init__(self, model_name, api_key: Optional[str] = None, base_url: Optional[str] = None, cost_tracker: Optional[CostTracker] = None):
        super().__init__(model_name, api_key, base_url, role="manager", temperature=1, cost_tracker=cost_tracker)
    
    async def decompose_task(self, task_description: str, constraints: Dict[str, Any]) -> List[Subtask]:
        """
        Decompose a complex task into subtasks based on PMC paper Appendix G
        
        Args:
            task_description: The main task description
            constraints: Dictionary of constraints (budget, temporal, spatial, etc.)
        
        Returns:
            List of Subtask objects
        """
        # Manager Agent prompt from PMC paper Appendix G (adapted without tool references)
        system_msg = """You are a task management assistant designed to break down tasks and manage task progress.

The main job in task breakdown is populating the JSON template. Based on user's query, your main task is to gather valid information related to transportation, dining, attractions, and accommodation.

You must first output the Chain of Thoughts (COT). In the COT, you need to explain how you break down the main task into sub-tasks. The sub-tasks need to be broken down to a very low granularity, hence it's possible that some sub-tasks will depend on the execution results of previous tasks. You also need to specify which sub-tasks require the execution results of previous tasks. When writing about each sub-task, you must also write out its respective local constraints. Finally, you write the global constraint of the main task.

Before filling in the template, you must first understand the user's request, carefully analyzing the tasks contained within it. Once you have a clear understanding of the tasks, you determine the sequence in which each task should be executed. Following this sequence, you rewrite the tasks into complete descriptions, taking into account the dependencies between them.

After determining your subtasks, you must first identify the local constraints for each sub-task, then the global constraints. Local constraints are constraints that needed to be considered in only in each specific sub-task. Meanwhile, global constraints are the constraints mentioned in the query that needed to be jointly considered across all the sub-tasks. You must not write global constraints that are only related to only some of the sub-tasks.

Important Reminder: Global constraint is constraint that are jointly considered across all the sub-tasks. You must not write global constraint that is only related to only some of the sub-tasks.

Important Rule: You must only output one global constraint that you think is the most important one based on the user query.

You must output the JSON at the end."""
        
        prompt = f"""User Query:
{task_description}

Constraints Analysis:
{json.dumps(constraints, indent=2)}

Please break down this task into subtasks following these requirements:

1. First, provide your Chain of Thoughts (COT) explaining:
   - How you break down the main task into sub-tasks
   - Why each subtask is necessary
   - Which sub-tasks depend on previous tasks
   - The local constraints for each sub-task
   - The global constraint (only one, the most important one)

2. Then output a JSON structure following this format:
{{
    "main_task": "Gather valid information related to transportation, dining, attractions, and accommodation based on user's query",
    "global_constraints": ["<the single most important global constraint>"],
    "sub_tasks": {{
        "task_1": {{
            "content": "<description of subtask. If depends on previous task, start with 'Based on <item> from task_X,'>",
            "local_constraints": ["<local constraint 1>", "<local constraint 2>", ...],
            "require_data": ["<task_id if depends on previous task>"]
        }},
        "task_2": {{
            "content": "<description of subtask>",
            "local_constraints": ["<local constraint 1>", ...],
            "require_data": ["<task_id if depends>"]
        }}
    }}
}}

Important notes:
- The breakdown process of the sub-tasks must be simple with low granularity
- There is no limit to the number of subtasks
- "require_data" lists previous sub-tasks which their information is required by the current sub-task
- "content" should indicate dependent subtasks if applicable (e.g., "Based on the item A searched in task_1,...")
- When writing "local_constraints", write it as specific as possible
- Local constraints filter items individually (e.g., "Indian, Chinese or Mediterranean cuisine" not "Indian, Chinese and Mediterranean cuisine")
- You must only output one global constraint that is the most important one based on the user query
- Global constraint must be jointly considered across ALL sub-tasks

Output your COT first, then the JSON."""
        
        response, cost_info = await self.query(prompt, system_msg)
        
        # Parse the response to extract subtasks
        try:
            # Extract JSON from response (handle markdown code blocks and COT)
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                # Try to find JSON block
                parts = response.split("```")
                json_str = None
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        json_str = part[4:].strip()
                    elif part.startswith("{"):
                        json_str = part
                    if json_str and ("sub_tasks" in json_str or "subtasks" in json_str.lower()):
                        break
                if not json_str:
                    # Try to find JSON in the response
                    import re
                    json_match = re.search(r'\{[^{}]*"sub_tasks"[^{}]*\{[^{}]*\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        json_str = response.strip()
            else:
                # Try to extract JSON from response (may have COT before it)
                import re
                json_match = re.search(r'\{[^{}]*"(?:main_task|sub_tasks)"[^{}]*\{.*?\}\s*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response.strip()
            
            decomposition = json.loads(json_str)
            
            # Parse the new format: {main_task, global_constraints, sub_tasks: {task_1: {...}, task_2: {...}}}
            subtasks = []
            subtask_id = 1
            
            if "sub_tasks" in decomposition:
                sub_tasks_dict = decomposition["sub_tasks"]
                for task_key, task_data in sub_tasks_dict.items():
                    # Extract constraint types from local_constraints
                    local_constraints = task_data.get("local_constraints", [])
                    constraint_types = []
                    
                    # Map local constraints to ConstraintType enum
                    constraint_text = " ".join(local_constraints).lower()
                    if any(keyword in constraint_text for keyword in ["budget", "cost", "price"]):
                        constraint_types.append(ConstraintType.BUDGET)
                    if any(keyword in constraint_text for keyword in ["date", "time", "day", "duration"]):
                        constraint_types.append(ConstraintType.TEMPORAL)
                    if any(keyword in constraint_text for keyword in ["city", "location", "origin", "destination", "transport"]):
                        constraint_types.append(ConstraintType.SPATIAL)
                    if any(keyword in constraint_text for keyword in ["preference", "cuisine", "accommodation", "pet"]):
                        constraint_types.append(ConstraintType.PREFERENCE)
                    if any(keyword in constraint_text for keyword in ["must", "require", "no", "not"]):
                        constraint_types.append(ConstraintType.HARD)
                    if not constraint_types:
                        constraint_types.append(ConstraintType.COMMONSENSE)
                    
                    # Extract dependencies from require_data
                    require_data = task_data.get("require_data", [])
                    dependencies = []
                    if require_data:
                        # Map task names to IDs
                        task_name_to_id = {}
                        for idx, (key, _) in enumerate(sub_tasks_dict.items(), 1):
                            task_name_to_id[key] = idx
                        
                        for req_task in require_data:
                            if req_task in task_name_to_id:
                                dependencies.append(task_name_to_id[req_task])
                    
                    subtask = Subtask(
                        id=subtask_id,
                        description=task_data.get("content", ""),
                        constraint_types=constraint_types if constraint_types else [ConstraintType.COMMONSENSE],
                        dependencies=dependencies,
                        local_constraints=local_constraints if local_constraints else []
                    )
                    subtasks.append(subtask)
                    subtask_id += 1
            elif isinstance(decomposition, list):
                # Fallback: handle old format (list of subtasks)
                for data in decomposition:
                    constraint_types = [ConstraintType(ct) for ct in data.get("constraint_types", [])]
                    subtask = Subtask(
                        id=data.get("id", subtask_id),
                        description=data.get("description", ""),
                        constraint_types=constraint_types if constraint_types else [ConstraintType.COMMONSENSE],
                        dependencies=data.get("dependencies", []),
                        local_constraints=data.get("local_constraints", [])
                    )
                    subtasks.append(subtask)
                    subtask_id += 1
            
            return subtasks
        
        except Exception as e:
            print(f"Error parsing subtasks: {e}")
            print(f"Response was: {response}")
            # Return a default decomposition
            return [
                Subtask(
                    id=1,
                    description="Extract and understand all constraints from the task",
                    constraint_types=[ConstraintType.HARD, ConstraintType.COMMONSENSE],
                    dependencies=[],
                    local_constraints=[]
                ),
                Subtask(
                    id=2,
                    description="Plan transportation considering temporal and spatial constraints",
                    constraint_types=[ConstraintType.TEMPORAL, ConstraintType.SPATIAL, ConstraintType.BUDGET],
                    dependencies=[1],
                    local_constraints=[]
                ),
                Subtask(
                    id=3,
                    description="Plan accommodations considering preference and budget constraints",
                    constraint_types=[ConstraintType.PREFERENCE, ConstraintType.BUDGET, ConstraintType.HARD],
                    dependencies=[2],
                    local_constraints=[]
                ),
                Subtask(
                    id=4,
                    description="Plan daily activities (meals, attractions) considering all constraints",
                    constraint_types=[ConstraintType.PREFERENCE, ConstraintType.BUDGET, ConstraintType.TEMPORAL, ConstraintType.COMMONSENSE],
                    dependencies=[2, 3],
                    local_constraints=[]
                ),
                Subtask(
                    id=5,
                    description="Integrate all plans and verify constraint satisfaction",
                    constraint_types=[ConstraintType.HARD, ConstraintType.COMMONSENSE],
                    dependencies=[2, 3, 4],
                    local_constraints=[]
                )
            ]


class ExecutorAgent(LLMAgent):
    """Executor Agent responsible for executing specific subtasks"""
    
    def __init__(self, agent_id: str, constraint_type: ConstraintType, 
                 model_name, api_key: Optional[str] = None, base_url: Optional[str] = None, cost_tracker: Optional[CostTracker] = None):
        super().__init__(model_name, api_key, base_url, role=f"executor_{agent_id}", temperature=1, cost_tracker=cost_tracker)
        self.agent_id = agent_id
        self.constraint_type = constraint_type
    
    def _get_executor_prompt_by_type(self, constraint_type: ConstraintType) -> str:
        """Get the executor prompt based on constraint type from PMC paper Appendix G"""
        
        # Map constraint types to executor agent types from the paper
        executor_prompts = {
            ConstraintType.SPATIAL: """You are a search agent. You can search for information related to transportation, cities, accommodations, restaurants, and attractions based on user's query.

These are the rules you should follow:
1. Before you search, you must output your reasoning. You must mention what information you have obtained from previous results and what information you are looking to obtain.
2. If you cannot provide an informative response based on user query, please consider alternative approaches.
3. Please do not make any assumptions using your internal knowledge. All information should be derived from the provided data.
4. After you gather all the information you need, please output the information based on user's query. Your information must be as detailed as possible.
5. You should only provide informative response based on user query. Don't provide any other advice.

For each item in your search result, you need to ensure you write out all the features. Do not miss any detail of every feature.

Your output format is as below:
Search Result of <Type of Items>:
1. Name: <Name of Item 1>
   <Feature 1>: <Detail of Feature 1 of Item 1>
   <Feature 2>: <Detail of Feature 2 of Item 1>
   ...
   <Feature n>: <Detail of Feature n of Item 1>

2. Name: <Name of Item 2>
   <Feature 1>: <Detail of Feature 1 of Item 2>
   <Feature 2>: <Detail of Feature 2 of Item 2>
   ...
   <Feature n>: <Detail of Feature n of Item 2>

...

N. Name: <Name of Item N>
   <Feature 1>: <Detail of Feature 1 of Item N>
   <Feature 2>: <Detail of Feature 2 of Item N>
   ...
   <Feature n>: <Detail of Feature n of Item N>""",
            
            ConstraintType.BUDGET: """You are a search and analysis agent specialized in budget and cost considerations. You can search for and analyze information related to transportation, accommodations, restaurants, and attractions with focus on pricing and budget constraints.

These are the rules you should follow:
1. Before you analyze, you must output your reasoning. You must mention what information you have obtained from previous results and what budget constraints you need to consider.
2. Please do not make any assumptions using your internal knowledge.
3. You must provide detailed cost information for each item you recommend.
4. Your analysis must ensure the total cost stays within the budget constraint.
5. You should only provide informative response based on user query and budget requirements.

For each item in your result, ensure you include all cost-related features and details.""",
            
            ConstraintType.TEMPORAL: """You are a search and planning agent specialized in temporal constraints. You can search for and plan information related to transportation schedules, dates, durations, and time-based activities.

These are the rules you should follow:
1. Before you plan, you must output your reasoning about temporal constraints.
2. Please do not make any assumptions using your internal knowledge.
3. You must provide detailed temporal information (dates, times, durations) for each item.
4. Your planning must ensure all temporal constraints are satisfied.
5. You should only provide informative response based on user query and temporal requirements.""",
            
            ConstraintType.PREFERENCE: """You are a search agent specialized in user preferences. You can search for information related to accommodations, restaurants, and attractions based on user preferences (cuisine, accommodation type, pet-friendly, etc.).

These are the rules you should follow:
1. Before you search, you must output your reasoning about user preferences.
2. Please do not make any assumptions using your internal knowledge.
3. You must provide detailed information that matches user preferences.
4. Your search results must satisfy all preference constraints.
5. You should only provide informative response based on user query and preferences.

For each item, ensure all preference-related features are clearly stated.""",
            
            ConstraintType.HARD: """You are a constraint verification agent. You verify that search results and plans satisfy hard constraints (must-satisfy requirements).

These are the rules you should follow:
1. Before you verify, you must output your reasoning about hard constraints.
2. Please do not make any assumptions using your internal knowledge.
3. You must verify that all hard constraints are satisfied.
4. Your verification must be thorough and specific.
5. You should only provide verification results based on constraints.""",
            
            ConstraintType.COMMONSENSE: """You are a commonsense planning agent. You ensure that plans are realistic, sensible, and align with commonsense expectations.

These are the rules you should follow:
1. Before you analyze, you must output your reasoning about commonsense considerations.
2. Please ensure plans are realistic and practical.
3. You must verify that plans align with commonsense.
4. Your analysis should ensure diverse and sensible planning.
5. You should provide commonsense-based recommendations."""
        }
        
        return executor_prompts.get(constraint_type, executor_prompts[ConstraintType.SPATIAL])
    
    async def execute_subtask(self, subtask: Subtask, task_description: str, 
                            constraints: Dict[str, Any], 
                            previous_results: Dict[int, str]) -> str:
        """
        Execute a subtask given its description and dependencies
        Based on PMC paper Appendix G executor agent prompts
        
        Args:
            subtask: The subtask to execute
            task_description: Original task description
            constraints: All constraints
            previous_results: Results from dependent subtasks (subtask_id -> result)
        
        Returns:
            Result of the subtask execution
        """
        # Get executor prompt based on constraint type
        system_msg = self._get_executor_prompt_by_type(self.constraint_type)
        
        # Build context from dependent subtasks
        context = ""
        if subtask.dependencies:
            context = "\n\nResults from previous subtasks (use these to inform your solution):\n"
            for dep_id in subtask.dependencies:
                if dep_id in previous_results:
                    context += f"\nSubtask {dep_id} Result:\n{previous_results[dep_id]}\n"
        
        # Get local constraints for this subtask
        local_constraints_text = ""
        local_constraints = getattr(subtask, 'local_constraints', None) or []
        if local_constraints:
            local_constraints_text = "\n\nLocal Constraints for this subtask:\n" + "\n".join(f"- {lc}" for lc in local_constraints)
        
        prompt = f"""User Query:
{task_description}

Current Subtask to Execute:
- ID: {subtask.id}
- Description: {subtask.description}
- Constraint Type Focus: {self.constraint_type.value}
{local_constraints_text}

All Constraints:
{json.dumps(constraints, indent=2)}
{context}

Please execute this subtask and provide detailed results. Your response should:
1. Fully address the subtask description
2. Satisfy all relevant constraints, especially {self.constraint_type.value} constraints
3. Be compatible and consistent with results from dependent subtasks
4. Include specific, detailed information that can be used in the final plan
5. Follow the output format specified in the instructions

Provide your reasoning first, then the detailed results."""
        
        result, cost_info = await self.query(prompt, system_msg)
        return result


class PMCAgent:
    """Main PMC Agent that coordinates manager and executor agents"""
    
    def __init__(self, model_name, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 num_executors_per_constraint: int = 1):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.cost_tracker = CostTracker()
        
        # Store intermediate outputs for logging
        self.intermediate_outputs: List[Dict[str, Any]] = []
        
        # Initialize manager and executors with cost tracker
        self.manager = ManagerAgent(model_name, api_key, base_url, cost_tracker=self.cost_tracker)
        self.executors: Dict[ConstraintType, List[ExecutorAgent]] = {}
        self.num_executors_per_constraint = num_executors_per_constraint
        
        # Initialize executor agents for each constraint type
        for constraint_type in ConstraintType:
            self.executors[constraint_type] = [
                ExecutorAgent(f"{constraint_type.value}_{i}", constraint_type, model_name, api_key, base_url, cost_tracker=self.cost_tracker)
                for i in range(num_executors_per_constraint)
            ]
    
    def _log_intermediate_output(self, step: str, agent_role: str, agent_id: Optional[str],
                                 prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        """Log intermediate output from agents"""
        self.intermediate_outputs.append({
            "step": step,
            "agent_role": agent_role,
            "agent_id": agent_id,
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
    
    def get_intermediate_outputs(self) -> List[Dict[str, Any]]:
        """Get all intermediate outputs"""
        return self.intermediate_outputs.copy()
    
    def clear_intermediate_outputs(self):
        """Clear intermediate outputs"""
        self.intermediate_outputs = []
    
    async def plan(self, task_description: str, constraints: Dict[str, Any], 
                  data: Optional[Dict[str, Any]] = None) -> Plan:
        """
        Main planning method that coordinates the collaborative agent system
        
        Args:
            task_description: Description of the planning task
            constraints: Dictionary of constraints
            data: Optional data to use for planning (e.g., TravelPlanner data)
        
        Returns:
            Plan object with subtasks and final plan
        """
        # Clear intermediate outputs for new planning task
        self.clear_intermediate_outputs()
        
        # Step 1: Manager decomposes the task
        print("Step 1: Manager agent decomposing task...")
        
        # Log manager's task decomposition
        manager_prompt = f"Task: {task_description}\n\nConstraints:\n{json.dumps(constraints, indent=2)}"
        subtasks = await self.manager.decompose_task(task_description, constraints)
        
        # Store manager's output (we'll capture the actual response in decompose_task)
        self._log_intermediate_output(
            step="task_decomposition",
            agent_role="manager",
            agent_id="manager",
            prompt=manager_prompt,
            response=json.dumps([{
                "id": st.id,
                "description": st.description,
                "constraint_types": [ct.value for ct in st.constraint_types],
                "dependencies": st.dependencies
            } for st in subtasks], indent=2),
            metadata={"num_subtasks": len(subtasks)}
        )
        
        print(f"Task decomposed into {len(subtasks)} subtasks")
        
        # Step 2: Execute subtasks in dependency order
        completed_subtasks: Dict[int, str] = {}
        subtask_objects: Dict[int, Subtask] = {st.id: st for st in subtasks}
        
        # Topological sort to execute subtasks in correct order
        execution_order = self._topological_sort(subtasks)
        
        for subtask_id in execution_order:
            subtask = subtask_objects[subtask_id]
            print(f"\nStep 2.{subtask_id}: Executing subtask {subtask_id}: {subtask.description}")
            subtask.status = "in_progress"
            
            # Select executor agents based on constraint types
            # Use multiple executors for complex subtasks with multiple constraint types
            executor_results = []
            
            for constraint_type in subtask.constraint_types:
                if constraint_type in self.executors:
                    executors = self.executors[constraint_type]
                    # Use the first executor for this constraint type
                    executor = executors[0]
                    print(f"  Using executor agent: {executor.agent_id}")
                    result = await executor.execute_subtask(
                        subtask, task_description, constraints, completed_subtasks
                    )
                    executor_results.append(result)
            
            # If multiple executors worked on this subtask, combine their results
            if len(executor_results) > 1:
                # Use a simple combination: concatenate with separator
                combined_result = "\n\n".join(executor_results)
            else:
                combined_result = executor_results[0] if executor_results else "No result"
            
            subtask.result = combined_result
            subtask.status = "completed"
            completed_subtasks[subtask_id] = combined_result
            print(f"  Subtask {subtask_id} completed")
        
        # Step 3: Integrate all results into final plan
        print("\nStep 3: Integrating results into final plan...")
        
        # Collect subtask results for logging
        subtask_summary = "\n\n".join([
            f"Subtask {st.id}: {st.description}\nResult: {st.result}\n"
            for st in subtasks if st.result
        ])
        
        data_context = ""
        if data:
            data_context = f"\n\nAvailable Data:\n{json.dumps(data, indent=2)}"
        
        integrator_prompt = f"""Original Task: {task_description}

Constraints:
{json.dumps(constraints, indent=2)}
{data_context}

Subtask Results:
{subtask_summary}"""
        
        final_plan = await self._integrate_plans(subtasks, task_description, constraints, data)
        
        # Log integrator's output
        self._log_intermediate_output(
            step="plan_integration",
            agent_role="integrator",
            agent_id="integrator",
            prompt=integrator_prompt,
            response=final_plan,
            metadata={"num_subtasks": len(subtasks)}
        )
        
        # Step 4: Constraint verification is done in eval_score after JSON extraction
        # (removed _verify_constraints as it's handled by eval_score)
        
        # Get cost summary (will be updated in run_sync_search with task-specific cost)
        cost_summary = self.cost_tracker.get_summary()
        
        plan = Plan(
            subtasks=subtasks,
            final_plan=final_plan,
            constraints_satisfied=False,  # Will be set by eval_score later
            total_cost=cost_summary["total_cost"],
            total_input_tokens=cost_summary["total_input_tokens"],
            total_output_tokens=cost_summary["total_output_tokens"],
            total_tokens=cost_summary["total_tokens"]
        )
        
        return plan
    
    def _topological_sort(self, subtasks: List[Subtask]) -> List[int]:
        """Topological sort of subtasks based on dependencies"""
        # Build dependency graph
        graph = {st.id: st.dependencies for st in subtasks}
        in_degree = {st.id: 0 for st in subtasks}
        
        for st in subtasks:
            for dep in st.dependencies:
                if dep in in_degree:
                    in_degree[st.id] += 1
        
        # Kahn's algorithm
        queue = [st.id for st in subtasks if in_degree[st.id] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for st in subtasks:
                if node in st.dependencies:
                    in_degree[st.id] -= 1
                    if in_degree[st.id] == 0:
                        queue.append(st.id)
        
        return result
    
    async def _integrate_plans(self, subtasks: List[Subtask], task_description: str,
                              constraints: Dict[str, Any], data: Optional[Dict[str, Any]]) -> str:
        """Integrate all subtask results into a final cohesive plan
        Based on PMC paper Appendix G integrator/deliverer agent prompt
        """
        integrator = LLMAgent(self.model_name, self.api_key, self.base_url, role="integrator", temperature=1, cost_tracker=self.cost_tracker)
        
        # Integrator/Deliverer Agent prompt from PMC paper Appendix G
        system_msg = """You are a proficient planner. Based on the provided items and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense.

The provided items for each task are ranked in preferences order, from highest to lowest. Please prioritise the higher ranking options in your plan but also make sure meet all the constraints from the query.

The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B).

Before you write your detailed plan, please analyse the hard constraints based on the query. In addition to that, we will also give you the hard constraints that we have analysed so far from the query. You also need to analyse the commonsense constraints for a diverse and sensible trip plan. Your commonsense constraints must also include not repeating restaurant choices throughout the trip.

Later, you write the detailed plan and adhere to the format given in the example. Please remember that the travel plan that you give must adhere to all of the constraints. Your plan has to be as complete as possible, without requiring decisions to be made upon arrival.

Finally, you write the reasons of why this plan will adhere all the constraints. Don't output anything else after that.

Remember, your output format for "Travel Plan" must fully adhere to the format in the example. For example, the Breakfast section only requires the name of restaurant, followed by city location. Don't write anything extra that is not required, for example the cost.

Important rule, please do not make any assumption that a non-restaurant place has meal. You don't need to plan any meal before heading to your travel destination. You don't need to plan any lunch or dinner after heading back from trip. Please make sure you never have repeating restaurant choices throughout the trip.

***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
Hard constraints: <All the hard constraints given to you and based on the query>
Commonsense constraints: <All the commonsense constraints for a diverse and sensible trip plan>
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

Reasons: <Reason why the plan adheres to constraints>
***** Example Ends *****"""
        
        # Collect all subtask results
        subtask_summary = "\n\n".join([
            f"Subtask {st.id}: {st.description}\nResult: {st.result}\n"
            for st in subtasks if st.result
        ])
        
        # Extract hard constraints from constraints dict
        hard_constraints = []
        if "hard" in constraints:
            hard_dict = constraints["hard"]
            for key, value in hard_dict.items():
                if value:
                    hard_constraints.append(f"{key}: {value}")
        
        # Build commonsense constraints analysis
        commonsense_constraints = [
            "Ensure diverse restaurant choices throughout the trip (no repetition)",
            "Plan activities that are realistic and practical",
            "Consider travel time and logistics between locations",
            "Ensure accommodation is available for required nights",
            "Plan attractions and activities that are accessible and reasonable"
        ]
        
        data_context = ""
        if data:
            data_context = f"\n\nAvailable Data:\n{json.dumps(data, indent=2)}"
        
        prompt = f"""Query: {task_description}

Hard constraints: {', '.join(hard_constraints) if hard_constraints else 'None specified'}

Commonsense constraints: {', '.join(commonsense_constraints)}

Subtask Results (ranked by preference, highest to lowest):
{subtask_summary}
{data_context}

Please provide a detailed travel plan following the format in the example above. Your output should include:
1. Hard constraints analysis
2. Commonsense constraints analysis
3. Travel Plan (following the exact format from the example)
4. Reasons (explain why this plan adheres to all constraints)

Remember:
- All information must be derived from the provided subtask results and data
- Do not repeat restaurant choices throughout the trip
- Follow the exact format shown in the example
- Do not include cost information in the plan sections
- Use '-' for unnecessary information
- When traveling between cities in one day, use "from A to B" format for Current City
"""
        
        final_plan, cost_info = await integrator.query(prompt, system_msg)
        return final_plan
    
    # Removed _verify_constraints method - constraint verification is now handled by eval_score
    
    def get_total_cost(self, start_index: int = 0) -> Dict[str, Any]:
        """
        Get total cost information
        
        Args:
            start_index: Only include call_details from this index onwards (for per-task cost tracking)
        
        Returns:
            Dictionary with total_cost, total_input_tokens, total_output_tokens, total_tokens, and call_details
        """
        return self.cost_tracker.get_summary(start_index)
    
    def reset_cost_tracker(self):
        """Reset the cost tracker"""
        self.cost_tracker.reset()


def extract_constraints_from_query(query: str, model_name: str = "gpt-4o", 
                                   api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract constraints from a TravelPlanner query using LLM
    
    Args:
        query: The query string from TravelPlanner dataset
        model_name: Model to use for extraction
        api_key: OpenAI API key
    
    Returns:
        Dictionary of extracted constraints
    """
    # First try rule-based extraction for basic constraints
    constraints = _extract_constraints_rule_based(query)
    
    # Then use LLM to refine and extract more complex constraints
    try:
        client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        system_msg = """You are a constraint extraction agent. Extract all constraints from a travel planning query.
        Return a JSON object with the following structure:
        {
            "budget": <number or null>,
            "temporal": {
                "start_date": <string or null>,
                "end_date": <string or null>,
                "duration_days": <number or null>
            },
            "spatial": {
                "origin": <string or null>,
                "destination": <string or null>,
                "destination_count": <number or null>,
                "destination_state": <string or null>,
                "num_people": <number or null>
            },
            "preference": {
                "transportation": <string or null>,
                "accommodation": {
                    "room_type": <string or null>,
                    "pet_friendly": <boolean or null>
                },
                "cuisine": <array or null>
            },
            "hard": {
                "no_flights": <boolean or null>,
                "pet_friendly_accommodation": <boolean or null>
            }
        }
        
        Only include fields that are explicitly mentioned or clearly implied in the query.
        """
        
        prompt = f"""Extract all constraints from the following travel planning query:

Query: {query}

Return only valid JSON, no additional text."""
        
        # Use asyncio to make the call synchronous in this context
        # For async version, this would need to be refactored
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we need to handle this differently
            # For now, skip LLM extraction and use rule-based only
            return constraints
        else:
            response = loop.run_until_complete(
                client.chat.completions.create(
                    model='gpt-4o',
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1
                )
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON from response
            if "```json" in result_text:
                json_str = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                json_str = result_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = result_text.strip()
            
            llm_constraints = json.loads(json_str)
            
            # Merge LLM constraints with rule-based constraints (LLM takes priority)
            constraints = _merge_constraints(constraints, llm_constraints)
            
    except Exception as e:
        print(f"Warning: LLM constraint extraction failed: {e}, using rule-based extraction only")
    
    return constraints


def _extract_constraints_rule_based(query: str) -> Dict[str, Any]:
    """Rule-based constraint extraction as fallback"""
    constraints = {
        "temporal": {},
        "spatial": {},
        "preference": {},
        "hard": {}
    }
    
    query_lower = query.lower()
    
    # Extract budget
    budget_match = re.search(r'\$?([\d,]+)', query)
    if budget_match and "budget" in query_lower:
        try:
            constraints["budget"] = float(budget_match.group(1).replace(',', ''))
        except:
            pass
    
    # Extract dates
    date_pattern = r'([A-Z][a-z]+ \d{1,2}(?:th|st|nd|rd)?,? \d{4})'
    dates = re.findall(date_pattern, query)
    if len(dates) >= 2:
        constraints["temporal"]["start_date"] = dates[0]
        constraints["temporal"]["end_date"] = dates[-1]
    elif len(dates) == 1:
        constraints["temporal"]["date"] = dates[0]
    
    # Extract number of days
    days_match = re.search(r'(\d+)[- ]day', query_lower)
    if days_match:
        constraints["temporal"]["duration_days"] = int(days_match.group(1))
    
    # Extract transportation preferences
    if "no flight" in query_lower or "without flight" in query_lower or "not flight" in query_lower:
        constraints["hard"]["no_flights"] = True
        constraints["preference"]["transportation"] = "no_flights"
    
    # Extract pet-friendly
    if "pet" in query_lower:
        constraints["preference"]["accommodation"] = constraints["preference"].get("accommodation", {})
        constraints["preference"]["accommodation"]["pet_friendly"] = True
        constraints["hard"]["pet_friendly_accommodation"] = True
    
    # Extract room type
    if "not shared" in query_lower or "non-shared" in query_lower:
        constraints["preference"]["accommodation"] = constraints["preference"].get("accommodation", {})
        constraints["preference"]["accommodation"]["room_type"] = "not_shared"
    elif "shared room" in query_lower:
        constraints["preference"]["accommodation"] = constraints["preference"].get("accommodation", {})
        constraints["preference"]["accommodation"]["room_type"] = "shared"
    
    # Extract number of people
    people_match = re.search(r'(\d+)\s+(?:people|person)', query_lower)
    if people_match:
        constraints["spatial"]["num_people"] = int(people_match.group(1))
    elif "couple" in query_lower:
        constraints["spatial"]["num_people"] = 2
    elif "single" in query_lower or "individual" in query_lower:
        constraints["spatial"]["num_people"] = 1
    
    # Extract city count
    cities_match = re.search(r'(\d+)\s+cit(?:y|ies)', query_lower)
    if cities_match:
        constraints["spatial"]["destination_count"] = int(cities_match.group(1))
    
    # Extract origin and destination
    from_match = re.search(r'from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', query, re.IGNORECASE)
    if from_match:
        constraints["spatial"]["origin"] = from_match.group(1).strip()
    
    to_match = re.search(r'to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', query, re.IGNORECASE)
    if to_match:
        constraints["spatial"]["destination"] = to_match.group(1).strip()
    
    # Extract state
    state_match = re.search(r'in\s+([A-Z][a-z]+)', query)
    if state_match:
        constraints["spatial"]["destination_state"] = state_match.group(1)
    
    return constraints


def _merge_constraints(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two constraint dictionaries, with update taking priority"""
    result = base.copy()
    
    for key, value in update.items():
        if value is None:
            continue
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_constraints(result[key], value)
        else:
            result[key] = value
    
    return result


async def extract_constraints_from_query_async(query: str, model_name: str = "gpt-4o",
                                              api_key: Optional[str] = None,
                                              base_url: Optional[str] = None,
                                              cost_tracker: Optional[CostTracker] = None) -> Dict[str, Any]:
    """
    Async version of constraint extraction using LLM
    
    Args:
        query: The query string from TravelPlanner dataset
        model_name: Model to use for extraction
        api_key: OpenAI API key
        cost_tracker: Optional cost tracker
    
    Returns:
        Dictionary of extracted constraints
    """
    # First try rule-based extraction
    constraints = _extract_constraints_rule_based(query)
    
    # Then use LLM to refine
    try:
        extractor = LLMAgent('gpt-4o', api_key, base_url, role="constraint_extractor", 
                            temperature=1, cost_tracker=cost_tracker)
        
        system_msg = """You are a constraint extraction agent. Extract all constraints from a travel planning query.
        Return a JSON object with the following structure:
        {
            "budget": <number or null>,
            "temporal": {
                "start_date": <string or null>,
                "end_date": <string or null>,
                "duration_days": <number or null>
            },
            "spatial": {
                "origin": <string or null>,
                "destination": <string or null>,
                "destination_count": <number or null>,
                "destination_state": <string or null>,
                "num_people": <number or null>
            },
            "preference": {
                "transportation": <string or null>,
                "accommodation": {
                    "room_type": <string or null>,
                    "pet_friendly": <boolean or null>
                },
                "cuisine": <array or null>
            },
            "hard": {
                "no_flights": <boolean or null>,
                "pet_friendly_accommodation": <boolean or null>
            }
        }
        
        Only include fields that are explicitly mentioned or clearly implied in the query.
        Return only valid JSON, no additional text."""
        
        prompt = f"""Extract all constraints from the following travel planning query:

Query: {query}"""
        
        response, cost_info = await extractor.query(prompt, system_msg)
        
        # Extract JSON from response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        
        llm_constraints = json.loads(json_str)
        
        # Merge LLM constraints with rule-based constraints (LLM takes priority)
        constraints = _merge_constraints(constraints, llm_constraints)
        
    except Exception as e:
        print(f"Warning: LLM constraint extraction failed: {e}, using rule-based extraction only")
    
    return constraints


async def run_sync_search(example, example_id, model_name, api_key, base_url, save_dir):
    """
    Run planning for a single example and save results
    
    Args:
        example: Example data from dataset
        example_id: ID of the example
        model_name: Model name for PMC agent
        api_key: API key for PMC agent
        base_url: Base URL for API calls
        save_dir: Directory to save results
    
    Returns:
        Plan object
    """
    # Save results to files
    task_dir = Path(save_dir) / f"task_{example_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize PMC agent for this task - each example gets its own agent instance
    # This allows per-example cost tracking
    pmc = PMCAgent(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )

    query = example['query']
    task = PLANNER_INSTRUCTION.format(text=example['reference_information'], query=query)
    

    # Extract constraints from query
    print(f"\n[Example {example_id}] Extracting constraints from query...")
    constraints = await extract_constraints_from_query_async(
        query, 
        model_name='gpt-4o', 
        api_key=api_key,
        base_url=base_url,
        cost_tracker=pmc.cost_tracker  # Use PMC agent's cost tracker
    )
    
    # Execute planning
    plan = await pmc.plan(task, constraints)

    # Convert final_plan to JSON format
    print(f"\n[Example {example_id}] Converting final plan to JSON format...")
    try:
        extractor = LLMAgent(
            'gpt-4o', 
            api_key, 
            base_url,
            role="plan_extractor", 
            temperature=1, 
            cost_tracker=pmc.cost_tracker  # Use PMC agent's cost tracker
        )
        
        extraction_prompt = travelplanner_extraction + "\nText:\n" + plan.final_plan + "\nJSON:\n"
        
        extracted_json, cost_info = await extractor.query(extraction_prompt)
        
        # Clean up the extracted JSON (remove markdown code blocks)
        extracted_json = extracted_json.strip()
        if "```json" in extracted_json:
            extracted_json = extracted_json.split("```json")[1].split("```")[0].strip()
        elif "```" in extracted_json:
            extracted_json = extracted_json.split("```")[1].split("```")[0].strip()
        
        # Validate JSON format and evaluate score
        try:
            json.loads(extracted_json)  # Validate JSON
            plan_json_str = extracted_json
            
            # Evaluate the plan using eval_score
            print(f"\n[Example {example_id}] Evaluating plan score...")
            final_score, concrete_dict = eval_score(extracted_json, example_id)
            print(f"[Example {example_id}] Final Score: {final_score}")
            print(f"[Example {example_id}] Concrete Dict: {concrete_dict}")
            
        except json.JSONDecodeError as e:
            print(f"[Example {example_id}] Warning: Extracted JSON is invalid: {e}")
            plan_json_str = None
            final_score = None
            concrete_dict = None
        except Exception as e:
            print(f"[Example {example_id}] Error in eval_score: {e}")
            plan_json_str = extracted_json  # Keep the JSON string even if eval fails
            final_score = None
            concrete_dict = None

    except Exception as e:
        print(f"[Example {example_id}] Error converting plan to JSON: {e}")
        plan_json_str = None
        final_score = None
        concrete_dict = None
    
    # Get cost summary from the PMC agent's cost tracker
    # This ensures we get the cost only for this specific task/example
    cost_summary = pmc.cost_tracker.get_summary()
    
    # Print results
    print("\n" + "="*80)
    print(f"[Example {example_id}] FINAL PLAN")
    print("="*80)
    print(plan.final_plan)
    print("\n" + "="*80)
    if final_score is not None:
        print(f"Final Score: {final_score}")
    print(f"Total Cost for this task: ${cost_summary['total_cost']:.4f}")
    print("="*80)
    
    
    
    # 1. Save intermediate outputs (all agent outputs during planning)
    intermediate_file = task_dir / "intermediate_outputs.json"
    intermediate_data = {
        "example_id": example_id,
        "query": query,
        "task_description": task,
        "constraints": constraints,
        "intermediate_outputs": pmc.get_intermediate_outputs(),  # Use task_pmc instead of pmc
        "subtasks": [
            {
                "id": st.id,
                "description": st.description,
                "constraint_types": [ct.value for ct in st.constraint_types],
                "dependencies": st.dependencies,
                "status": st.status,
                "result": st.result
            }
            for st in plan.subtasks
        ],
        "plan_metadata": {
            "final_score": final_score,
            "concrete_dict": concrete_dict,
            "total_cost": cost_summary["total_cost"],  # Use cost_summary from task_cost_tracker
            "total_tokens": cost_summary["total_tokens"],
            "total_input_tokens": cost_summary["total_input_tokens"],
            "total_output_tokens": cost_summary["total_output_tokens"]
        }
    }
    with intermediate_file.open("w", encoding="utf-8") as f:
        json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
    print(f"\n[Example {example_id}] Saved intermediate outputs to {intermediate_file}")
    
    # 2. Save final plan
    final_plan_file = task_dir / "final_plan.txt"
    with final_plan_file.open("w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\n")
        if final_score is not None:
            f.write(f"Final Score: {final_score}\n\n")
        f.write("="*80 + "\n")
        f.write("FINAL PLAN\n")
        f.write("="*80 + "\n\n")
        f.write(plan.final_plan)
    print(f"[Example {example_id}] Saved final plan to {final_plan_file}")
    
    # Also save as JSON for structured access
    final_plan_json_file = task_dir / "final_plan.json"
    final_plan_data = {
        "example_id": example_id,
        "query": query,
        "constraints": constraints,
        "final_plan": plan.final_plan,
        "final_plan_json": plan_json_str,  # Add extracted JSON format
        "final_score": final_score,
        "concrete_dict": concrete_dict,
        "subtasks": [
            {
                "id": st.id,
                "description": st.description,
                "constraint_types": [ct.value for ct in st.constraint_types],
                "dependencies": st.dependencies,
                "status": st.status,
                "result": st.result
            }
            for st in plan.subtasks
        ]
    }
    with final_plan_json_file.open("w", encoding="utf-8") as f:
        json.dump(final_plan_data, f, indent=2, ensure_ascii=False)
    
    # Save extracted JSON as a separate file if conversion was successful
    if plan_json_str:
        final_plan_extracted_json_file = task_dir / "final_plan_extracted.json"
        with final_plan_extracted_json_file.open("w", encoding="utf-8") as f:
            f.write(plan_json_str)
        print(f"[Example {example_id}] Saved extracted JSON plan to {final_plan_extracted_json_file}")
    
    # Save evaluation score to a separate file
    if final_score is not None:
        score_file = task_dir / "score.txt"
        with score_file.open("w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("EVALUATION SCORE\n")
            f.write("="*80 + "\n\n")
            f.write(f"Final Score: {final_score}\n\n")
            if concrete_dict:
                f.write("Concrete Dict:\n")
                f.write(json.dumps(concrete_dict, indent=2, ensure_ascii=False))
        print(f"[Example {example_id}] Saved evaluation score to {score_file}")
    
    # 3. Save cost information (only for this task)
    # cost_summary was already obtained above
    cost_file = task_dir / "cost.txt"
    with cost_file.open("w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("COST SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Input Tokens: {cost_summary['total_input_tokens']:,}\n")
        f.write(f"Total Output Tokens: {cost_summary['total_output_tokens']:,}\n")
        f.write(f"Total Tokens: {cost_summary['total_tokens']:,}\n")
        f.write(f"Total Cost: ${cost_summary['total_cost']:.4f}\n")
        f.write(f"Number of API Calls: {cost_summary['num_calls']}\n\n")
        f.write("="*80 + "\n")
        f.write("DETAILED COST BREAKDOWN\n")
        f.write("="*80 + "\n\n")
        for i, call in enumerate(cost_summary['call_details'], 1):
            f.write(f"Call {i}:\n")
            f.write(f"  Agent: {call['agent_role']}\n")
            f.write(f"  Model: {call['model']}\n")
            f.write(f"  Input Tokens: {call['input_tokens']:,} (${call['input_cost']:.4f})\n")
            f.write(f"  Output Tokens: {call['output_tokens']:,} (${call['output_cost']:.4f})\n")
            f.write(f"  Total Tokens: {call['total_tokens']:,}\n")
            f.write(f"  Cost: ${call['total_cost']:.4f}\n\n")
    print(f"[Example {example_id}] Saved cost information to {cost_file}")
    
    return plan

# Example usage and integration with TravelPlanner
async def main(args):
    """Example usage of PMC Agent"""
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get API credentials from environment
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")

    test_path = Path("/data/qin/lhh/Unified-MAS/MAS-Zero/data/src/travelplanner_test_16.jsonl")
    if not test_path.exists():
        raise FileNotFoundError(f"TravelPlanner test file not found: {test_path}")

    with test_path.open("r", encoding="utf-8") as fh:
        examples = [json.loads(line) for line in fh if line.strip()]
    
        
    
    # 控制并发数量的信号量，最多同时运行指定数量的任务
    semaphore = asyncio.Semaphore(50)

    async def run_task_with_semaphore(example_id, example):
        async with semaphore:
            # PMC agent is initialized inside run_sync_search for each example
            return await run_sync_search(example, example_id, args.model, api_key, base_url, save_dir=args.save_dir)
    
    tasks = []
    for example_id, example in enumerate(examples):
        tasks.append(run_task_with_semaphore(example_id, example))
    
    # Execute all tasks with progress bar
    if hasattr(tqdm_asyncio, 'gather'):
        results = await tqdm_asyncio.gather(*tasks)
    else:
        results = await asyncio.gather(*tasks)
    
   
    
    return results


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(args))

