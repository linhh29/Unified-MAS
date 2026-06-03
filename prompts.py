"""
Prompt模板模块
包含所有LLM交互使用的prompt模板
"""
from typing import List, Dict, Optional
import json

def get_task_keywords_prompt(samples_text: str) -> tuple[str, str]:
    """
    生成task_keywords提取的prompt。
    
    Args:
        samples_text: 样本文本
        
    Returns:
        (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are an expert dataset and task analyst. "
        "You are given multiple JSONL samples from a benchmark dataset. "
        "Your task is to carefully read all samples, analyze this Specific Domain Task, "
        "and extract keywords across six specific dimensions required to solve this task. "
        "The extracted keywords should be concise and representative, and should not focus on the specific data samples, but the general domain and task."
    )
    
    user_prompt = f"""
        # Input Data
        User Task samples:
        {samples_text}

        # Extraction Rules (Strictly Follow)
        Analyze the description above and reason to extract keywords for the following six dimensions. For each dimension, provide 5-10 most representative terms:

        1.  **Domain**: The macro industry background (e.g., Fintech, Supply Chain, Bioinformatics, etc.).
        2.  **Task**: The core technical problem to solve (e.g., Anomaly Detection, Named Entity Recognition, Summarization, etc.).
        3.  **Entities**: The specific data objects or physical entities involved (e.g., Transaction Logs, PDF Contracts, Protein Sequences, Sensor Data, etc.).
        4.  **Actions**: The specific operations performed on the data (e.g., Classify, Extract, Reason, Optimize, Verify, etc.).
        5.  **Constraints**: Performance metrics or limitations (e.g., Low Latency, Privacy Preserving, Explainability, Offline Inference, etc.).
        6.  **Desired Outcomes**: The expected results or metrics (e.g., Accuracy, Precision, Recall, F1 Score, AUC, MAP, NDCG, etc.).
        7.  **Implicit Knowledge**: *CRITICAL* - Based on your expert knowledge, infer specific jargon, SOTA techniques, common challenges, or potential risks that are not explicitly mentioned but are essential for solving this problem (e.g., "Imbalanced Data" for fraud, "Hallucination" for GenAI, "Bullwhip Effect" for supply chain, etc.).

        # Output Format
        Please output both your thinking and answer in the JSON format.
        "thinking" entry: [Your thinking process, how you arrive your answer]
        "answer" entry: [your answer in the JSON format]

        For "thinking" entry, you need to first carefully read the samples_text, summarize the task description, and then reason step by step to arrive your answer.

        For "answer" entry, please output a valid JSON object. Do not include any conversational filler or markdown formatting outside the JSON code block. Format as follows:

        {{
            "Domain": ["..."],
            "Task": ["..."],
            "Entities": ["..."],
            "Actions": ["..."],
            "Constraints": ["..."],
            "Desired_Outcomes": ["..."],
            "Implicit_Knowledge": ["..."]
        }}
    """
    
    return system_prompt, user_prompt


def get_search_queries_prompt(keywords_json_str: str) -> tuple[str, str]:
    """
    生成search_queries生成的prompt。
    
    Args:
        keywords_json_str: 关键词JSON字符串
        
    Returns:
        (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are an expert in Information Retrieval (IR) and Multi-Agent System Design. "
        "You know how to construct precise search queries to retrieve background knowledge, high-quality academic papers, "
        "code implementations, and industry Standard Operating Procedures (SOPs)."
    )
    
    user_prompt = f"""
    # Goal
    Based on the provided [Structured Keywords (Domain, Task, Entities, Actions, Constraints, Desired_Outcomes, Implicit_Knowledge)], apply four specific search strategies to generate a list of search queries for Google Scholar, GitHub, and General Web Search.

    # Input Data
    Structured Keywords JSON:
    {keywords_json_str}

    # Search Strategy Definitions
    Apply the following four strategies to construct your queries for each dimension:

    1.  **Strategy A: Background Knowledge**
        *   *Logic:* Domain + Implicit_Knowledge 
        *   *Aim:* Use domain jargon to find background knowledge, cutting-edge solutions, theoretical frameworks, and surveys.

    2.  **Strategy B: High-quality Academic Papers about System Architecture (Workflow & Design)**
        *   *Logic:* Task + Constraints 
        *   *Aim:* Find architectural designs (e.g., Router, Pipeline, Map-Reduce) that satisfy specific constraints (e.g., Privacy, Real-time).

    3.  **Strategy C: Technical Code Implementation**
        *   *Logic:* Entities + Actions 
        *   *Aim:* Find code repositories, libraries, or preprocessing tools for specific data types.

    4.  **Strategy D: Evaluation & Metrics**
        *   *Logic:* Task + Desired_Outcomes 
        *   *Aim:* Find standard datasets and quantitative metrics to evaluate the Agent's performance.

    # Output Instructions
    Generate 5-10 search queries for EACH strategy. Use Boolean operators (AND, OR) where appropriate to optimize results.

    Please output ONLY a valid JSON object with the following structure:

    {{
      "strategy_A": [
        {{"query": "...", "reasoning": "Using [Implicit Term] to find advanced patterns"}}
      ],
      "strategy_B": [
        {{"query": "...", "reasoning": "To find architectures satisfying [Constraint]"}}
      ],
      "strategy_C": [
        {{"query": "...", "reasoning": "To find tools for processing [Object] via [Action]"}}
      ],
      "strategy_D": [
        {{"query": "...", "reasoning": "To find benchmarks for [Outcome]"}}
      ]
    }}
    """
    
    return system_prompt, user_prompt


def get_strategy_analysis_prompt(strategy_name: str, task_thinking: str, files_summary: List[str]) -> tuple[str, str]:
    """
    根据Strategy生成相应的分析prompt。
    
    Args:
        strategy_name: Strategy名称
        task_thinking: 任务思考描述
        files_summary: 文件摘要列表
        
    Returns:
        (system_prompt, user_prompt)
    """
    files_text = chr(10).join(files_summary)
    
    if "Strategy A" in strategy_name or "Background Knowledge" in strategy_name:
        system_prompt = (
            "You are an expert legal and technical analyst. "
            "Your task is to analyze multiple documents (PDFs and TXTs) that were retrieved through web search "
            "for background knowledge related to a specific task, and provide a comprehensive summary."
        )
        user_prompt = f"""
Task Description (from task_keywords thinking):
{task_thinking}

Strategy: {strategy_name}

Documents Retrieved:
{files_text}

**IMPORTANT: Please provide EXTREMELY DETAILED and COMPREHENSIVE analysis. The more detailed, the better. Include specific examples, step-by-step explanations, concrete details, and thorough descriptions.**

Your task:
1. Analyze all the documents above and identify from which aspects/aspects they discuss the background knowledge related to the task described above. **Be very specific and detailed about each aspect.**

2. Summarize the key background information that is needed to solve this task. **Provide EXTREMELY DETAILED descriptions**, including but not limited to:
   - **Overall task workflow and processes**: Provide a DETAILED, step-by-step workflow with specific stages, decision points, inputs/outputs at each stage, and the complete process flow. Include concrete examples and detailed explanations of each step.
   - **Key points and important considerations**: List ALL important points with detailed explanations, why they matter, and how they impact the task. Be thorough and comprehensive.
   - **Domain-specific knowledge and terminology**: Provide detailed definitions, explanations, and context for each term. Include how these concepts relate to each other and their significance in the domain.
   - **Relevant frameworks, methodologies, or approaches**: Describe each framework/methodology in DETAIL, including their components, how they work, when to use them, and their advantages/disadvantages. Provide specific examples.
   - **Common challenges and solutions**: Detail each challenge with specific scenarios, root causes, and provide detailed solutions with step-by-step approaches. Include real-world examples.
   - **Best practices and standards**: Provide detailed best practices with specific guidelines, checklists, and detailed explanations of why each practice is important.

3. Provide a structured summary that clearly explains:
   - What background knowledge aspects are covered in these documents (with detailed descriptions)
   - What specific background information is needed to solve the task (be very specific and detailed)
   - How this background knowledge relates to the task at hand (provide detailed connections and relationships)

**Remember: The more detailed and comprehensive your analysis, the better. Include specific examples, detailed explanations, step-by-step processes, and thorough descriptions throughout.**

Please provide a comprehensive and well-structured analysis in JSON format:
{{
    "aspects_covered": ["detailed aspect1 with explanation", "detailed aspect2 with explanation", ...],
    "background_information": {{
        "task_workflow": "DETAILED step-by-step workflow with all stages, inputs/outputs, decision points, and complete process flow. Be extremely thorough.",
        "key_points": ["detailed point1 with full explanation", "detailed point2 with full explanation", ...],
        "domain_knowledge": "DETAILED explanation of domain-specific knowledge, terminology, concepts, and their relationships. Be comprehensive and thorough.",
        "frameworks_methodologies": ["detailed framework1 with components and usage", "detailed framework2 with components and usage", ...],
        "challenges_solutions": "DETAILED description of common challenges with specific scenarios, root causes, and detailed step-by-step solutions with examples.",
        "best_practices": "DETAILED best practices with specific guidelines, checklists, and explanations of importance. Be comprehensive."
    }},
    "summary": "EXTREMELY DETAILED and comprehensive summary of the background knowledge, including all key points, detailed workflows, and thorough explanations..."
}}
"""
    elif "Strategy B" in strategy_name or "System Architecture" in strategy_name:
        system_prompt = (
            "You are an expert system architect and technical analyst. "
            "Your task is to analyze academic papers and documents about system architecture, workflow, and design "
            "related to a specific task, and provide insights on architectural patterns and design approaches."
        )
        user_prompt = f"""
Task Description (from task_keywords thinking):
{task_thinking}

Strategy: {strategy_name}

Documents Retrieved:
{files_text}

**IMPORTANT: Please provide EXTREMELY DETAILED and COMPREHENSIVE analysis. The more detailed, the better. Include specific architectural diagrams descriptions, detailed workflow steps, component interactions, and thorough explanations.**

Your task:
1. Analyze all the documents above and identify the system architectures, workflows, and design patterns they discuss. **Be very specific and detailed about each pattern and architecture.**

2. Summarize the key architectural and design information relevant to solving this task. **Provide EXTREMELY DETAILED descriptions**, including but not limited to:
   - **System architecture patterns and structures**: Provide DETAILED descriptions of each architecture pattern, including components, their roles, data flow, communication patterns, and how they work together. Include specific examples and detailed explanations.
   - **Workflow designs and process flows**: Provide EXTREMELY DETAILED, step-by-step workflow descriptions with all stages, transitions, decision points, data flows, error handling, and complete process flows. Include detailed diagrams descriptions and specific examples.
   - **Component interactions and interfaces**: Detail how components interact, what interfaces they use, data formats, protocols, and communication mechanisms. Be very specific and thorough.
   - **Design principles and constraints**: Provide detailed explanations of each design principle (e.g., privacy, real-time, scalability) with specific implementation strategies, trade-offs, and detailed guidelines. Include concrete examples.
   - **Architectural trade-offs and decisions**: Detail each trade-off with specific scenarios, pros/cons, decision criteria, and detailed explanations of why certain choices are made. Be comprehensive.
   - **Best practices for system design**: Provide detailed best practices with specific guidelines, patterns to follow, anti-patterns to avoid, and detailed explanations. Include real-world examples.

3. Provide a structured summary that clearly explains:
   - What architectural patterns and workflows are covered in these documents (with detailed descriptions)
   - What specific architectural/design information is needed to solve the task (be very specific and detailed)
   - How these architectural approaches relate to the task requirements (provide detailed connections and relationships)

**Remember: The more detailed and comprehensive your analysis, the better. Include specific architectural details, detailed workflow steps, component interactions, and thorough explanations throughout.**

Please provide a comprehensive and well-structured analysis in JSON format:
{{
    "architectural_patterns": ["detailed pattern1 with components and structure", "detailed pattern2 with components and structure", ...],
    "design_information": {{
        "system_architectures": "DETAILED description of system architectures with components, data flows, communication patterns, and how they work together. Be extremely thorough.",
        "workflow_designs": ["DETAILED step-by-step workflow1 with all stages and transitions", "DETAILED step-by-step workflow2 with all stages and transitions", ...],
        "component_interactions": "DETAILED description of component interactions, interfaces, data formats, protocols, and communication mechanisms. Be comprehensive.",
        "design_constraints": ["detailed constraint1 with implementation strategies", "detailed constraint2 with implementation strategies", ...],
        "architectural_tradeoffs": "DETAILED description of trade-offs with specific scenarios, pros/cons, decision criteria, and explanations. Be thorough.",
        "design_best_practices": "DETAILED best practices with specific guidelines, patterns, anti-patterns, and explanations. Include examples. Be comprehensive."
    }},
    "summary": "EXTREMELY DETAILED and comprehensive summary of the architectural and design knowledge, including all patterns, detailed workflows, and thorough explanations..."
}}
"""
    elif "Strategy C" in strategy_name or "Code Implementation" in strategy_name:
        system_prompt = (
            "You are an expert AI system architect and LLM prompt engineer. "
            "Your task is to analyze code repositories and design frameworks for solving tasks using Large Language Models (LLMs). "
            "Focus on high-level architecture, operation design, and how to migrate traditional ML/small model approaches to LLM-based solutions."
        )
        user_prompt = f"""
Task Description (from task_keywords thinking):
{task_thinking}

Strategy: {strategy_name}

Documents Retrieved (Code Repositories):
{files_text}

**IMPORTANT: Focus on FRAMEWORK DESIGN and LLM MIGRATION, not on specific libraries or dependencies. Think about how to solve the task at a high level using LLMs.**

Your task:
1. **Analyze the overall framework and architecture** in the provided code:
   - What is the high-level workflow and operation flow?
   - How are different components organized and connected?
   - What are the key operations/steps needed to solve the task?
   - How can these operations be efficiently designed and orchestrated?

2. **Design LLM-based solutions** to replace or enhance the small model implementations:
   - **Operation Design**: How to break down the task into well-defined operations that can be executed by LLMs? What operations are needed and how should they be structured?
   - **Prompt Engineering**: For each operation that was previously done by small models, design detailed prompts for LLMs. What should be the input format, what instructions should be given, and what output format is expected?
   - **Model-level Mechanisms**: How to implement global constraint checking, validation, error handling, and other model-level controls? What mechanisms are needed to ensure the LLM operations work correctly together?
   - **Data Flow**: What is the input/output format for each LLM operation? How should data flow between different operations? What transformations are needed?

3. **Migration Strategy**: 
   - How can the existing small model code be adapted to use LLMs instead?
   - What are the key differences in approach between small models and LLMs for this task?
   - How to design the system to leverage LLM capabilities while maintaining the original workflow structure?

4. **Framework Considerations**:
   - What is the overall system architecture needed to solve this task?
   - How should operations be orchestrated and sequenced?
   - What are the critical decision points and branching logic?
   - How to handle state management and context passing between operations?

**Focus Areas (in order of importance):**
1. **Overall Framework & Architecture**: How to structure the solution at a high level
2. **Operation Design**: How to break down the task into LLM-executable operations
3. **Prompt Design**: Detailed prompt templates for each LLM operation
4. **Data Processing & Flow**: Input/output formats and data transformations between operations
5. **Model-level Mechanisms**: Global constraints, validation, error handling
6. **Migration Strategy**: How to adapt small model code to LLM-based approach

**Do NOT focus on:**
- Specific library dependencies or installation requirements
- Environment setup details
- Low-level implementation details of non-LLM components

Please provide a comprehensive and well-structured analysis in JSON format:
{{
    "overall_framework": {{
        "architecture": "DETAILED description of the overall system architecture and framework design needed to solve this task. Explain the high-level structure, component organization, and how different parts work together.",
        "workflow": "DETAILED step-by-step workflow description. Explain the sequence of operations, decision points, and how the system processes the task from start to finish.",
        "key_operations": ["operation1: detailed description of what it does and how it fits in the framework", "operation2: ...", ...]
    }},
    "llm_migration": {{
        "operation_design": "DETAILED description of how to design operations for LLM execution. Explain how to break down the task into operations, how operations should be structured, and how they should interact.",
        "prompt_templates": [
            {{
                "operation_name": "name of the operation",
                "purpose": "what this operation does in the overall framework",
                "input_format": "detailed description of input format and structure",
                "prompt_template": "detailed prompt template with placeholders and instructions",
                "output_format": "detailed description of expected output format",
                "constraints": "any constraints or validation rules for this operation"
            }},
            ...
        ],
        "model_level_mechanisms": "DETAILED description of model-level mechanisms needed: global constraint checking, validation rules, error handling strategies, state management, context passing, etc. Be very specific about how these mechanisms work.",
        "migration_strategy": "DETAILED explanation of how to migrate from small model code to LLM-based approach. What changes are needed, what can be reused, and how to adapt the existing workflow."
    }},
    "data_processing": {{
        "input_output_formats": "DETAILED description of input/output formats for LLM operations. What data structures are needed, what format should be used, and how data should be structured.",
        "data_flow": "DETAILED description of how data flows between operations. What transformations are needed, how to pass context between operations, and how to maintain data consistency.",
        "preprocessing": "DETAILED description of any preprocessing needed before sending data to LLMs (if any).",
        "postprocessing": "DETAILED description of any postprocessing needed after receiving LLM outputs (if any)."
    }},
    "summary": "EXTREMELY DETAILED and comprehensive summary of the framework design, operation structure, LLM migration strategy, and how to solve this task using LLMs. Include specific examples of prompt designs, operation flows, and architectural decisions."
}}
"""
    elif "Strategy D" in strategy_name or "Evaluation" in strategy_name:
        system_prompt = (
            "You are an expert evaluator and metrics analyst. "
            "Your task is to analyze documents about evaluation metrics, benchmarks, and assessment methods "
            "related to a specific task, and provide insights on evaluation approaches and standards."
        )
        user_prompt = f"""
Task Description (from task_keywords thinking):
{task_thinking}

Strategy: {strategy_name}

Documents Retrieved:
{files_text}

**IMPORTANT: Please provide EXTREMELY DETAILED and COMPREHENSIVE analysis. The more detailed, the better. Include specific metric definitions, detailed evaluation procedures, step-by-step assessment workflows, and thorough explanations.**

Your task:
1. Analyze all the documents above and identify the evaluation metrics, benchmarks, and assessment methods they discuss. **Be very specific and detailed about each metric and method.**

2. Summarize the key evaluation information relevant to solving this task. **Provide EXTREMELY DETAILED descriptions**, including but not limited to:
   - **Standard evaluation metrics and their definitions**: Provide DETAILED definitions for each metric, including mathematical formulas, calculation methods, interpretation guidelines, and specific use cases. Include examples and detailed explanations.
   - **Benchmark datasets and evaluation protocols**: Detail each dataset with size, format, structure, quality, and provide DETAILED evaluation protocols with step-by-step procedures, data splits, evaluation criteria, and complete assessment workflows. Be extremely thorough.
   - **Assessment methodologies and procedures**: Provide DETAILED, step-by-step assessment workflows with all stages, evaluation criteria, scoring methods, and complete procedures. Include specific examples and detailed explanations.
   - **Performance standards and baselines**: Detail performance benchmarks with specific numbers, comparison methods, baseline implementations, and detailed explanations of what constitutes good performance. Be comprehensive.
   - **Evaluation best practices and guidelines**: Provide detailed best practices with specific guidelines, common mistakes to avoid, validation procedures, and detailed explanations. Include real-world examples.
   - **Metrics interpretation and analysis methods**: Detail how to interpret each metric, what values indicate good/bad performance, statistical analysis methods, and detailed interpretation guidelines. Be thorough.

3. Provide a structured summary that clearly explains:
   - What evaluation metrics and benchmarks are covered in these documents (with detailed descriptions)
   - What specific evaluation information is needed to assess task performance (be very specific and detailed)
   - How these evaluation approaches relate to the task requirements (provide detailed connections and relationships)

**Remember: The more detailed and comprehensive your analysis, the better. Include specific metric definitions, detailed evaluation procedures, step-by-step workflows, and thorough explanations throughout.**

Please provide a comprehensive and well-structured analysis in JSON format:
{{
    "evaluation_metrics": ["detailed metric1 with definition and formula", "detailed metric2 with definition and formula", ...],
    "evaluation_information": {{
        "standard_metrics": ["detailed metric1 with calculation method", "detailed metric2 with calculation method", ...],
        "benchmark_datasets": ["detailed dataset1 with protocol", "detailed dataset2 with protocol", ...],
        "assessment_methodologies": "DETAILED step-by-step assessment workflow with all stages, criteria, scoring methods, and complete procedures. Be extremely thorough.",
        "performance_standards": "DETAILED performance benchmarks with specific numbers, comparison methods, baselines, and explanations. Be comprehensive.",
        "evaluation_best_practices": "DETAILED best practices with guidelines, common mistakes, validation procedures, and explanations. Include examples. Be comprehensive.",
        "metrics_interpretation": "DETAILED interpretation guidelines with analysis methods, value meanings, and statistical considerations. Be thorough."
    }},
    "summary": "EXTREMELY DETAILED and comprehensive summary of the evaluation and metrics knowledge, including all metrics, detailed procedures, and thorough explanations..."
}}
"""
    else:
        # 默认prompt
        system_prompt = (
            "You are an expert analyst. "
            "Your task is to analyze multiple documents retrieved through web search "
            "for a specific task, and provide a comprehensive summary."
        )
        user_prompt = f"""
Task Description (from task_keywords thinking):
{task_thinking}

Strategy: {strategy_name}

Documents Retrieved:
{files_text}

Your task:
1. Analyze all the documents above and identify the key information they contain.
2. Summarize the relevant information needed to solve this task.
3. Provide a structured summary.

Please provide a comprehensive and well-structured analysis in JSON format:
{{
    "key_aspects": ["aspect1", "aspect2", ...],
    "relevant_information": {{
        "main_points": ["point1", "point2", ...],
        "details": "..."
    }},
    "summary": "Overall comprehensive summary..."
}}
"""
    
    return system_prompt, user_prompt


def get_node_generation_prompt(task_thinking: str, strategy_analysis: str, code_template: str, task_samples: str = "") -> tuple[str, str]:
    """
    生成node生成任务的prompt。
    只允许两种节点类型：LLM_Generator（调用 LLM）和 Retrieval_RAG（搜索引擎 RAG）。
    
    Args:
        task_thinking: 任务思考描述
        strategy_analysis: Strategy分析结果（JSON字符串）
        code_template: 代码模板（字符串）
        task_samples: 任务样本（JSONL格式的输入样本）
        
    Returns:
        (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are an expert system architect and multi-agent system designer. "
        "Your task is to design a **complete** pipeline of nodes (operators) to solve a specific task "
        "based on the task description and strategy analysis. "
        "You must **carefully identify every step the task requires** and **create a corresponding node for each**—do not omit necessary steps. "
        "You may ONLY use two types of nodes: LLM_Generator (call LLM to do reasoning/generation) and Retrieval_RAG (use search engine for RAG). "
        "**All verification, validation, parsing, and format-checking must be implemented via the LLM** (by writing clear requirements and rules in the prompt_template so the LLM performs checks and outputs the correct format). Do NOT write code to verify, parse, or validate LLM outputs—use the LLM to do it."
        " Each node must follow the provided node definition structure and work together to form a complete solution pipeline."
    )
    
    task_samples_section = ""
    if task_samples:
        task_samples_section = f"""
Task Samples (Example Inputs):
{task_samples}

These are real examples of the task inputs. Use them to understand the exact input format, data structure, and what the task expects.
"""
    
    user_prompt = f"""
Task Description:
{task_thinking}
{task_samples_section}
Strategy Analysis:
{strategy_analysis}

The code template for all nodes is (Only use for the all_code field in the node definition):
{code_template}

**IMPORTANT: Design a pipeline using ONLY two node types. Each node must strictly follow the node definition structure above.**

**STRICT RULES:**
- **Allowed node types**: LLM_Generator and Retrieval_RAG.
- **Verification and parsing via LLM, NOT code**: Any need for verification (e.g. format check, validity check, number validation), parsing (e.g. extracting structured data from text), or fixing malformed output must be implemented by the **LLM**: put the rules and expected output format in the **prompt_template** (System Prompt / User Prompt) so that the LLM performs the checks and returns well-formed output. Do NOT write Python code to validate (e.g. json.loads, try/except, re.match) or parse LLM responses—if output might be messy, add instructions in the prompt or add another LLM_Generator node that asks the LLM to clean/validate and re-output.
- For calculations or deterministic steps, use an LLM_Generator node: ask the LLM to perform the reasoning and output the result in the required format; do not use code.

Your task:
1. **Analyze the task and strategy analysis** to understand:
   - What is the overall task that needs to be solved?
   - What background knowledge, architectural patterns, and evaluation metrics are available?
   - **List exhaustively all operations and workflow steps the task requires** (e.g. input parsing, fact extraction, knowledge retrieval, reasoning, validation, synthesis, final answer formatting). Do not skip or merge steps mentally—write them down. Each of these should eventually map to at least one node.

2. **Design a complete pipeline of nodes** using ONLY LLM_Generator and Retrieval_RAG:
   - **For each step you identified above, create a corresponding node.** Do not generate too few nodes: the pipeline must have enough nodes to cover the entire task from input to final output. If the task typically needs e.g. extraction → retrieval → reasoning → synthesis → formatting, you must have nodes for each (or clearly combined in a justified way).
   - Break down the task into logical steps; each step is either (a) call LLM to do something, or (b) use search engine to retrieve then LLM to summarize/use.
   - **Before finalizing the node list, double-check: Is there a node that handles retrieval if the task needs external knowledge? Is there a node that produces the final answer in the required format? Are there nodes for every distinct logical phase (e.g. understand input, gather context, reason, output)?** Add nodes if any required step is missing.
   - Nodes are connected through dependencies (dependencies field).
   - Do NOT add any node that would require custom Python code (e.g. no "Calculator Tool", "Validator Tool", "Parser Tool" as Python code). Use LLM_Generator for such roles if needed.

3. **For each node, provide complete information** following the node definition:
   - **node_name**: A descriptive name (e.g., "xx_Agent")
   - **node_type**: One of [LLM_Generator, Retrieval_RAG].
   - **description**: Summary of the node's role in the pipeline
   - **dependencies**: List of upstream node names that this node depends on
   - **input**: What information this node reads from inputs (be specific based on task samples)
   - **output**: What this node produces (be specific about output format)
   - **constraints**: Global constraints this node must comply with (from task requirements)
   - **implementation**:
     - **logic_description**: Detailed description of the implementation logic (no code; describe what the node does in terms of LLM calls and/or search + LLM).
     - **prompt_template**: (For both node types) **MUST provide complete, detailed prompt content: System Prompt (marked as "System Prompt:") and User Prompt (marked as "User Prompt:") with placeholders. Be specific and include examples.**
     - **tools_needed**: For Retrieval_RAG nodes use ["Search"]; for LLM_Generator use [].
     - Do NOT include "code_snippet". Omit it or set to null.
   - **all_code**: **Minimal runnable code only**: (1) Read inputs from input_data. (2) For LLM_Generator: fill the prompt_template with input values and call self.llm_client.chat(node_messages, response_format=...). (3) For Retrieval_RAG: build search query from inputs, call self.search_engine.multi_turn_search(query), then fill prompt_template with retrieved context and call self.llm_client.chat. (4) Return output_data dict. Do NOT add code that verifies, parses, or validates the LLM response (no json.loads, re, try/except for parsing, no format checks)—all verification/parsing is done by the LLM via the prompt.**

   **CRITICAL:**
   - **Verification and parsing = LLM's job**: If a node needs to ensure valid JSON, correct format, or validated numbers, write these requirements **in the prompt_template** (e.g. \"Output only valid JSON.\", \"Validate each amount and output the approved breakdown.\"). Do NOT implement verification or parsing in all_code (no json.loads, re, or try/except to fix LLM output). Use the LLM to do verification and output clean results.
   - **LLM_Generator nodes**: Provide full System Prompt and User Prompt in prompt_template; put any validation/format rules there. all_code must only: extract inputs, build node_messages from prompt_template, call self.llm_client.chat, return {{output_key: response}}. No code that parses or validates the response.
   - **Retrieval_RAG nodes**: logic_description must state what to retrieve and how to summarize. prompt_template must include System Prompt and User Prompt; use a placeholder like {{retrieved_context}} or {{retrieved_chunks}} for the search result. all_code must only: build query from inputs, call self.search_engine.multi_turn_search(query), build node_messages from prompt_template with retrieved content, call self.llm_client.chat, return output. No code that parses or validates the response.
   - **Retrieval_RAG**: Design so you do NOT retrieve the question itself; retrieve only related knowledge (e.g. laws, case law, background) needed to answer. State this in logic_description and prompt_template.

4. **Design principles**:
   - Use only LLM_Generator and Retrieval_RAG.
   - **Completeness over brevity**: Ensure the pipeline has **enough nodes** for the task. List all logical steps the task requires (from task description and strategy analysis), then create one node (or more) for each step. When in doubt, add a dedicated node rather than overloading one node with multiple responsibilities. Too few nodes often lead to incomplete or poor results.
   - Each node has a single responsibility. Dependencies form a DAG.
   - Use LLM_Generator for reasoning, generation, extraction, validation, and any step that would otherwise need "code" (e.g. ask LLM to output structured JSON or numbers).
   - Use Retrieval_RAG when external knowledge retrieval (search) is needed, then LLM to summarize or use the retrieved context.

5. **Output format**:
   Provide a JSON object with this structure:
   {{
       "pipeline_description": "Overall description of the pipeline and how nodes work together",
       "nodes": [
           {{
               "node_name": "...",
               "node_type": "LLM_Generator or Retrieval_RAG only",
               "description": "...",
               "dependencies": ["..."],
               "input": ["..."],
               "output": ["..."],
               "constraints": "...",
               "implementation": {{
                   "logic_description": "...",
                   "prompt_template": "...",
                   "tools_needed": ["Search"] for Retrieval_RAG, [] for LLM_Generator
               }},
               "all_code": "Minimal code only: input extraction, then LLM call(s) or search+LLM, then return output_data. No verification/parsing blocks."
           }},
           ...
       ],
       "Connections": "Complete Python code for def execute_pipeline(self, initial_input_data): ... Execute nodes in dependency order; collect inputs from initial_input_data or results; call self.NodeName(input_data); store outputs; return final output. Import json if needed."
   }}

**Remember:**
- **Carefully check that the task needs are fully covered by nodes**: Before outputting, verify you have a node for every required step (e.g. input understanding, retrieval if needed, reasoning, synthesis, final answer). The number of nodes should be sufficient to solve the task completely—do not output a pipeline with too few nodes.
- Use only LLM_Generator and Retrieval_RAG.
- **All verification and parsing must be done by the LLM**: write rules and output-format requirements in the prompt_template; do not write code to verify or parse LLM output (no json.loads, re, try/except for validation/parsing in all_code).
- all_code must be minimal: read input -> (LLM call or search+LLM) -> return output. No code that checks or parses the LLM response.
- Dependencies must form a valid DAG. Use task samples to align input/output formats.
- For "Connections": generate the pipeline execution function that runs nodes in dependency order and passes data correctly.
"""
    
    return system_prompt, user_prompt


def get_debug_prompt(
    error_info: str,
    node_code: str,
    connections_code: str,
    sample_input: Dict,
    intermediate_results: Optional[List[Dict]] = None,
    expected_answer: Optional[str] = None,
) -> tuple[str, str]:
    """
    生成用于调试和修复代码的 prompt。
    与 utils.py 中 debug 调用样式一致，支持可选的 intermediate_results 与 expected_answer。
    """
    system_prompt = (
        "You are an expert Python debugger and code fixer. "
        "Your task is to analyze errors in pipeline code and fix them. "
        "You must provide complete, runnable code that fixes the issue."
    )

    # 中间结果：让 LLM 看到每一步的输入输出（可选）
    intermediate_section = ""
    if intermediate_results:
        intermediate_section = "\n# Intermediate outputs (all node inputs/outputs)\n"
        intermediate_section += (
            "The following are the inputs and outputs of each node that ran before the failure. "
            "Use this to see exactly where the pipeline state went wrong.\n\n"
        )
        for idx, step in enumerate(intermediate_results, 1):
            step_type = step.get("step") or step.get("node_name") or f"Step_{idx}"
            intermediate_section += f"--- [{idx}] {step_type} ---\n"
            if step.get("input_kwargs"):
                intermediate_section += "  input_kwargs: " + json.dumps(
                    step["input_kwargs"], ensure_ascii=False
                )[:1500] + "\n"
            if step.get("input_args"):
                intermediate_section += "  input_args: " + str(step["input_args"])[:800] + "\n"
            out = step.get("output", "")
            if out is not None:
                out_str = str(out)[:2000]
                if len(str(out)) > 2000:
                    out_str += " ... (truncated)"
                intermediate_section += "  output: " + out_str + "\n"
            if step.get("success") is False and step.get("error"):
                intermediate_section += "  error: " + str(step["error"])[:500] + "\n"
            intermediate_section += "\n"
        intermediate_section += "\n"

    expected_section = ""
    if isinstance(expected_answer, list):
        expected_answer = "\n".join(expected_answer)
    if expected_answer is not None and expected_answer.strip():
        expected_section = f"\n# Expected Answer (reference: pipeline should produce something in this direction)\n{expected_answer}\n"

    task_section = """
# Task
Analyze the error and fix the code. The error likely occurs because:
1. Input data format doesn't match what the code expects
2. Type mismatches (e.g., expecting dict but got str)
3. Missing keys in dictionaries
4. Incorrect data access patterns

# Requirements
1. Fix the node code to properly handle the input data format
2. Fix the Connections code if needed to pass data correctly
3. Ensure all code is complete and runnable
4. Maintain the same output format and structure
"""

    user_prompt = f"""
# Error Information
{error_info}
{intermediate_section}
# Sample Input Data
{json.dumps(sample_input, ensure_ascii=False, indent=2)}
{expected_section}
# Current Node Code (last node or where output is produced)
{node_code}

# Current Connections Code (pipeline execution code)
{connections_code}
{task_section}

# Output Format
Provide a JSON object with the following structure:
{{
    "fixed_node_code": "Complete fixed code for the node (if node needs fixing)",
    "fixed_connections_code": "Complete fixed Connections code (if connections needs fixing)",
    "explanation": "Brief explanation of what was wrong and how you fixed it"
}}

If only the node needs fixing, set "fixed_connections_code" to the original connections_code.
If only connections needs fixing, set "fixed_node_code" to the original node_code.
"""
    
    return system_prompt, user_prompt


def get_node_optimization_prompt(
    question: str,
    answer: str,
    node_name: str,
    node_type: str,
    node_description: str,
    node_implementation: Dict,
    node_all_code: str,
    intermediate_outputs: List[Dict],
    node_reward: float,
    node_index: int
) -> tuple[str, str]:
    """
    生成用于节点优化的 prompt
    
    Args:
        question: 输入问题
        answer: 正确答案
        node_name: 节点名称
        node_type: 节点类型 (LLM_Generator, Retrieval_RAG)
        node_description: 节点描述
        node_implementation: 节点实现信息（包含 prompt_template, tools_needed 等）
        node_all_code: 节点的完整代码
        intermediate_outputs: 所有节点的中间输出列表
        node_reward: 该节点的奖励值
        node_index: 该节点在 pipeline 中的索引位置
        
    Returns:
        (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are an expert system optimizer and code reviewer. "
        "Your task is to analyze a node in a multi-agent pipeline that has the lowest reward "
        "and optimize its internal structure to improve performance. "
        "All optimizations must be achieved via the LLM. You may: (1) improve existing LLM prompts, "
        "(2) introduce new LLM calls where needed, (3) optimize how multiple LLM calls within the same node "
        "communicate and interact—e.g. what is passed between calls, in what format, in what order, and how results are aggregated. "
        "Do NOT add Python code for rules, regex, normalization, or filtering—fix shortcomings by prompt engineering or by adding/adjusting LLM calls and their communication, not by code."
    )
    
    # 构建中间输出的上下文信息
    intermediate_context = ""
    for idx, output in enumerate(intermediate_outputs):
        intermediate_context += f"\n[Node {idx + 1}: {output.get('node_name', 'Unknown')}]\n"
        intermediate_context += f"Output: {output.get('output', '')[:500]}...\n"  # 限制长度
    
    # 根据节点类型构建不同的优化提示
    if node_type == "LLM_Generator":
        optimization_focus = """
        Focus on optimizing via the LLM (do not add Python code for validation/filtering):
        1. **Prompt Engineering**: Improve the system prompt and user prompt so the LLM produces better, more structured output. Add instructions in the prompt for the LLM to validate, normalize, or fix format—do not implement these in code.
        2. **LLM Calls**: Optimize the number and sequence of LLM calls. If something needs validation or correction, add or adjust an LLM call with clear instructions in the prompt, rather than adding code to parse or fix the output.
        3. **Inter-LLM Communication** (when the node has multiple LLM calls): Optimize how these calls communicate and interact. Consider: (a) What is passed from one call to the next—is the handoff clear and in a good format? (b) Is the order of calls optimal (e.g. extract-then-summarize vs. one shot)? (c) Should there be an extra LLM call to refine or reconcile intermediate results? (d) Are prompts for later calls explicitly given the outputs of earlier calls so the LLM can use them well? Fix any unclear handoff, missing context, or suboptimal ordering by redesigning the prompts and call sequence, not by adding parsing code.
        4. **Output Format**: In the prompt, specify the expected output format and ask the LLM to adhere to it. Do not add code (e.g. regex, json.loads) to normalize or parse the response.
        5. **Context Usage**: Improve how the prompt uses information from upstream nodes (and from earlier LLM calls in the same node) so the LLM can reason better.
        """
    elif node_type == "Retrieval_RAG":
        optimization_focus = """
        Focus on optimizing via the LLM (do not add Python code for filtering/normalization):
        1. **Retrieval Strategy**: Improve search query construction in the prompt or in the logic description; keep retrieval logic in code minimal (just build query and call search). Do not add Python code to filter or normalize search results—let the LLM do that in the summarization step.
        2. **Context Summarization**: Optimize the prompt so the LLM better summarizes or filters the retrieved context. Add instructions like "ignore irrelevant parts" or "output only the following structure" in the prompt; do not add regex or rule-based filtering in code.
        3. **Inter-LLM Communication** (if the node has multiple LLM calls, e.g. query-generation then summarization): Optimize how these calls interact. Ensure the query-generation prompt gets the right inputs and outputs a form that is clearly used for search; ensure the summarization prompt explicitly receives both the original task context and the retrieved text, and that the handoff is clear. Add or reorder LLM calls if needed (e.g. an extra step to refine the query or to filter before summarizing), all via prompts—no code for parsing or filtering.
        4. **Relevance**: Ask the LLM to focus on relevant content in the prompt (e.g. "Summarize only the parts relevant to..."). Do not add code to filter or screen the retrieved text.
        5. **Query Construction**: Improve how the query is built from inputs; keep it simple in code (e.g. string format from inputs), no custom parsing or normalization code.
        """
    else:
        optimization_focus = "Focus on optimizing via the LLM and prompts. Do not add Python code for rules, regex, normalization, or filtering."
    
    user_prompt = f"""
# Task Context
**Question**: {question}

**Expected Answer**: {answer}

# Node to Optimize
**Node Name**: {node_name}
**Node Type**: {node_type}
**Node Description**: {node_description}
**Node Reward**: {node_reward} (This is the lowest reward, indicating poor performance)
**Node Position**: Node {node_index + 1} in the pipeline

# Current Node Implementation
**Implementation Details**:
{json.dumps(node_implementation, ensure_ascii=False, indent=2)}

**Current Code**:
```python
{node_all_code}
```

# Pipeline Context
**All Intermediate Outputs** (to understand the data flow; when multiple samples exist, each [Sample N] block's node outputs correspond to the [Sample N] Question/Answer in Task Context above):
{intermediate_context}

# Analysis Task
Based on the question, expected answer, and the intermediate outputs from all nodes, analyze why this node has the lowest reward and provide optimization suggestions.

**Analysis Steps**:
1. **Identify the Problem**: 
   - What is the node's current output? (from intermediate_outputs)
   - How does it differ from what's expected?
   - What specific issues are causing the low reward?

2. **Root Cause Analysis**:
   - Is the prompt (for LLM_Generator/Retrieval_RAG) clear and specific enough?
   - Are the LLM calls structured optimally? If the node has multiple LLM calls, is the **communication between them** effective—e.g. is the handoff from one call to the next clear, in a good format, and in the right order? Are there missing or redundant steps?
   - Is the implementation handling all cases correctly?
   - Is the retrieval (for Retrieval_RAG) getting relevant information?
   - Are there any logical errors or missing validations?

3. **Optimization Strategy**:
   {optimization_focus}

**CRITICAL – Fix shortcomings via LLM, not code**: You may (1) improve existing **prompts**, (2) **introduce new LLM calls** (e.g. a refinement or validation step), (3) **optimize inter-LLM communication** when a node has multiple LLM calls—e.g. clarify what each call receives from the previous one, improve the handoff format in the prompt, reorder or add calls so the flow is clearer. Do NOT add Python code for rule-based checks, regex, normalization, or filtering. The code should remain minimal: prepare inputs → call LLM(s), passing outputs between calls as needed → return output.

# Output Format
Provide a JSON object with the following structure:
{{
    "analysis": {{
        "problem_identification": "Detailed description of what's wrong with the current node",
        "root_cause": "Analysis of why the node is performing poorly",
        "optimization_strategy": "Specific strategy to improve the node"
    }},
    "optimized_implementation": {{
        "prompt_template": "Updated prompt template (marked as System Prompt: and User Prompt:). Keep original if no changes needed.",
        "tools_needed": "Updated tools_needed (for Retrieval_RAG nodes). Keep original if no changes needed.",
        "logic_description": "Updated logic description explaining the optimization"
    }},
    "optimized_all_code": "Complete updated code for the node following the code_template structure. MUST be complete and runnable. Output the code in the same format as the original code. Do not output mark like ```python or ```",
    "optimization_explanation": "Detailed explanation of what was optimized and why"
}}

**IMPORTANT**:
- Fix any identified problems by improving the **prompt**, **adding or reordering LLM calls**, or **improving how multiple LLM calls in the same node communicate** (what is passed, in what format). Do NOT add Python code for validation, regex, normalization, rule-based filtering, or parsing of LLM output. optimized_all_code must stay minimal: get inputs → call LLM(s), passing outputs between calls as needed → return output.
- For LLM_Generator: Improve prompt_template and/or the number and sequence of LLM calls; if there are multiple calls, ensure each call’s prompt clearly receives and uses the outputs of previous calls. Do not add code to parse or validate responses.
- For Retrieval_RAG: Improve the summarization prompt and query construction; do not add code to filter or normalize retrieved content—instruct the LLM to do it in the prompt.
- The optimized_all_code MUST be complete and runnable but MUST NOT contain extra validation/parsing/regex/filtering code.
- If you add, remove, or reorder LLM calls, or change how they communicate (handoff format/order), explain the reasoning in logic_description.
"""
    
    return system_prompt, user_prompt

