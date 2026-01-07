"""
Prompt模板模块
包含所有LLM交互使用的prompt模板
"""
from typing import List, Dict
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
        "Your task is to design a pipeline of nodes (operators) to solve a specific task "
        "based on the task description and strategy analysis. "
        "Each node must follow the provided node definition structure and work together "
        "to form a complete solution pipeline."
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

**IMPORTANT: Please design a comprehensive pipeline of nodes to solve this task. Each node must strictly follow the node definition structure provided above.**

Your task:
1. **Analyze the task and strategy analysis** to understand:
   - What is the overall task that needs to be solved?
   - What background knowledge, architectural patterns, implementation details, and evaluation metrics are available?
   - What are the key operations and workflow steps needed to solve this task?

2. **Design a pipeline of nodes** that together solve the task:
   - Break down the task into logical operations/steps
   - Each operation should be implemented as a node
   - Nodes should be connected through dependencies (dependencies field)
   - Consider the workflow from the strategy analysis when designing the pipeline

3. **For each node, provide complete information** following the node definition:
   - **node_name**: A descriptive name (e.g., "xx_Agent")
   - **node_type**: One of [LLM_Generator, Python_Tool, Retrieval_RAG]
   - **description**: Summary of the node's role in the pipeline
   - **dependencies**: List of upstream node names that this node depends on
   - **input**: What information this node reads from inputs (be specific based on task samples)
   - **output**: What this node produces (be specific about output format)
   - **constraints**: Global constraints this node must comply with (from task requirements)
   - **implementation**:
     - **logic_description**: Detailed description of the implementation logic
     - **prompt_template**: (For LLM_Generator nodes) **MUST provide complete, detailed prompt content including system prompt (marked as System Prompt: in the prompt template) and user input (marked as User Prompt: in the prompt template) format. Be very specific and include examples.**
     - **code_snippet**: (For Python_Tool nodes) **MUST provide complete, runnable Python code. Include all necessary imports, error handling, and the full function implementation.**
     - **tools_needed**: (For Retrieval_RAG nodes) **Return Search if this node is a Retrieval_RAG node, otherwise return an empty list.**
   - **all_code**: (For all nodes) **MUST provide complete, runnable code for ALL nodes based on the code_template provided. Do not leave placeholders or incomplete implementations.**

   **CRITICAL: Code Generation Requirements:**
   - **For LLM_Generator nodes**: You MUST provide complete, detailed prompt templates. Include:
     * Full System Prompt with clear instructions
     * User Prompt with placeholders and examples
     * Expected output format specification
     * Any constraints or validation rules
     * You may need one or more LLM calls in one LLM_Generator node. If you need multiple LLM calls, you should define different System Prompts and User Prompts for each LLM call. And then elaborate the logic of how these LLM calls are connected and how the results are aggregated in the logic_description.
   
   - **For Python_Tool nodes**: You MUST provide complete, runnable Python code. Include:
     * All necessary imports
     * Full function definition with proper parameters
     * Complete implementation logic
     * Error handling
     * Return value specification
   
   
   - **For Retrieval_RAG nodes**: 
     * What information does this node retrieve from the knowledge base? Include it in the logic_description.
     * How to format/summarize the retrieved context based on LLM calls? Include the summarization prompt template in the prompt_template.
     * When you design the summarization prompt template, use retrieved_chunks as the mark for the retrieved context returned by the Search tool.
     * When you design the summarization prompt template, do not forget to add the requirment that cannot directly retrieve the question but can retrieve the information related to the question.
     * For example, for the legal case task, you cannot directly retrieve the specific case but can retrieve the background knowledge you need to solve the case like the related laws, case law, etc.
   

4. **Design principles**:
   - Nodes should be modular and reusable
   - Each node should have a clear, single responsibility
   - Dependencies should form a Directed Acyclic Graph (DAG)
   - Consider error handling and validation at appropriate nodes
   - Use LLM nodes for reasoning, generation, and complex text processing
   - Use Python tool nodes for deterministic operations (calculations, data transformations, etc.)
   - Use Retrieval_RAG nodes when external knowledge retrieval is needed

5. **Output format**:
   Please provide a JSON object with the following structure:
   {{
       "pipeline_description": "Overall description of the pipeline and how nodes work together",
       "nodes": [
           {{
               "node_name": "...",
               "node_type": "...",
               "description": "...",
               "dependencies": ["..."],
               "input": ["..."],
               "output": ["..."],
               "constraints": "...",
               "implementation": {{
                   "logic_description": "...",
                   "prompt_template": "...",  // Only for LLM_Generator nodes. Marked as System Prompt: and User Prompt: in the prompt template.
                   "code_snippet": "...",  // Only for Python_Tool nodes
                   "tools_needed": ["..."]  // Only for Retrieval_RAG nodes
               }},
               "all_code": "..."  // For all nodes, provide complete, runnable code based on the code_template provided. Do not leave placeholders or incomplete implementations.
           }},
           ... // ... more nodes
       ],
       "Connections": "Complete Python code that implements a pipeline execution function. Based on the dependencies of each node, manually design the execution order and data flow. The function should: 1) Execute nodes in the correct order according to their dependencies (nodes with no dependencies should be executed first, then nodes that depend on them, and so on), 2) Pass outputs from upstream nodes to downstream nodes as inputs, 3) Handle initial input data (raw task input from initial_input_data), 4) Return the final output. The function signature should be: def execute_pipeline(self, initial_input_data): ... The code should handle all nodes defined in the 'nodes' array and properly map inputs/outputs based on the 'input' and 'output' fields of each node. Manually design the execution sequence based on the dependency graph: nodes with empty dependencies list should be executed first using initial_input_data, then nodes that depend on those nodes should be executed using outputs from their dependencies, and so on. Example structure: def execute_pipeline(self, initial_input_data): # Step 1: Initialize a results dictionary to store outputs from each node # Step 2: Manually execute nodes in the correct order based on dependencies: #   - Execute nodes with no dependencies first (use initial_input_data) #   - Then execute nodes that depend on executed nodes (use results from dependencies) #   - Continue until all nodes are executed # Step 3: For each node: #   - Collect required inputs from initial_input_data and/or results dictionary based on the node's 'input' field #   - Call the node method (e.g., self.NodeName(input_data)) #   - Store the outputs in results dictionary using the 'output' field names as keys # Step 4: Return the final output (typically from the last node or a specific final output node). Remember to import the relevant libraries like json and other packages.
   }}

**Remember:**
- Be comprehensive and detailed in your node design
- Ensure all nodes follow the exact structure from the node definition
- Make sure dependencies form a valid DAG (no circular dependencies)
- Consider the strategy analysis when designing nodes (use background knowledge, architectural patterns, implementation details, and evaluation metrics)
- **CRITICALLY IMPORTANT**: You MUST provide complete, runnable code for ALL nodes. Do not leave placeholders or incomplete implementations.
- Use the task samples to understand the exact input/output formats and design nodes accordingly
- All code snippets must be complete, syntactically correct, and ready to use
- Make sure the nodes you design are enough to solve the task.
- **For the "Connections" field**: Generate complete, runnable Python code that implements a pipeline execution function. Based on the dependencies of each node, manually design the execution order and data flow. The function should:
  * Manually determine the execution order by analyzing the dependency graph: nodes with empty dependencies should be executed first, then nodes that depend on them, following the data flow direction
  * Maintain a results dictionary to store outputs from each executed node
  * For each node, collect its required inputs from either initial_input_data (for nodes with no dependencies) or from the results dictionary (for nodes with dependencies), based on the node's 'input' field and the 'output' fields of its dependency nodes
  * Call each node method with the collected input data (e.g., self.NodeName(input_data))
  * Store the node's outputs in the results dictionary using the output field names as keys
  * Return the final output (typically the output from the last node in the pipeline or a specific final output node)
  * Handle edge cases like nodes with no dependencies (they should get inputs from initial_input_data) and ensure all required inputs are available before executing a node
  * The execution order should follow the dependency graph: execute nodes in layers - first all nodes with no dependencies, then all nodes that only depend on the first layer, and so on
"""
    
    return system_prompt, user_prompt


def get_debug_prompt(error_info: str, node_code: str, connections_code: str, sample_input: Dict) -> tuple[str, str]:
    """
    生成用于调试和修复代码的 prompt
    
    Args:
        error_info: 错误信息（包括错误类型、消息、堆栈跟踪）
        node_code: 出错的节点代码
        connections_code: Connections 代码
        sample_input: 示例输入数据
        
    Returns:
        (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are an expert Python debugger and code fixer. "
        "Your task is to analyze errors in pipeline code and fix them. "
        "You must provide complete, runnable code that fixes the issue."
    )
    
    user_prompt = f"""
# Error Information
{error_info}

# Sample Input Data
{json.dumps(sample_input, ensure_ascii=False, indent=2)}

# Current Node Code (where error occurred)
{node_code}

# Current Connections Code (pipeline execution code)
{connections_code}

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
        node_type: 节点类型 (LLM_Generator, Python_Tool, Retrieval_RAG)
        node_description: 节点描述
        node_implementation: 节点实现信息（包含 prompt_template, code_snippet, tools_needed 等）
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
        "You must provide specific, actionable optimization suggestions and updated code."
    )
    
    # 构建中间输出的上下文信息
    intermediate_context = ""
    for idx, output in enumerate(intermediate_outputs):
        intermediate_context += f"\n[Node {idx + 1}: {output.get('node_name', 'Unknown')}]\n"
        intermediate_context += f"Output: {output.get('output', '')[:500]}...\n"  # 限制长度
    
    # 根据节点类型构建不同的优化提示
    if node_type == "LLM_Generator":
        optimization_focus = """
        Focus on optimizing:
        1. **Prompt Engineering**: Improve the system prompt and user prompt for better clarity, specificity, and instruction following
        2. **LLM Calls**: Optimize the number and sequence of LLM calls, add validation steps, or improve error handling
        3. **Output Format**: Ensure the output format is well-defined and matches the expected structure
        4. **Context Usage**: Better utilize information from upstream nodes
        """
    elif node_type == "Python_Tool":
        optimization_focus = """
        Focus on optimizing:
        1. **Code Logic**: Review and improve the Python code logic, add error handling, and validate inputs
        2. **Data Processing**: Optimize data transformations and calculations
        3. **Edge Cases**: Handle edge cases and boundary conditions better
        4. **Code Efficiency**: Improve code efficiency and readability
        """
    elif node_type == "Retrieval_RAG":
        optimization_focus = """
        Focus on optimizing:
        1. **Retrieval Strategy**: Improve search queries, add filters or constraints to retrieval
        2. **Context Summarization**: Optimize the prompt for summarizing retrieved context
        3. **Relevance Filtering**: Add mechanisms to filter irrelevant retrieved content
        4. **Query Construction**: Improve how search queries are constructed from inputs
        """
    else:
        optimization_focus = "Focus on optimizing the node's internal logic and implementation."
    
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
**All Intermediate Outputs** (to understand the data flow):
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
   - Are the LLM calls structured optimally?
   - Is the Python code (for Python_Tool) handling all cases correctly?
   - Is the retrieval (for Retrieval_RAG) getting relevant information?
   - Are there any logical errors or missing validations?

3. **Optimization Strategy**:
   {optimization_focus}

# Output Format
Provide a JSON object with the following structure:
{{
    "analysis": {{
        "problem_identification": "Detailed description of what's wrong with the current node",
        "root_cause": "Analysis of why the node is performing poorly",
        "optimization_strategy": "Specific strategy to improve the node"
    }},
    "optimized_implementation": {{
        "prompt_template": "Updated prompt template (for LLM_Generator/Retrieval_RAG nodes, marked as System Prompt: and User Prompt:). Keep original if no changes needed.",
        "code_snippet": "Updated code snippet (for Python_Tool nodes). Keep original if no changes needed.",
        "tools_needed": "Updated tools_needed (for Retrieval_RAG nodes). Keep original if no changes needed.",
        "logic_description": "Updated logic description explaining the optimization"
    }},
    "optimized_all_code": "Complete updated code for the node following the code_template structure. MUST be complete and runnable. Output the code in the same format as the original code. Do not output mark like ```python or ```",
    "optimization_explanation": "Detailed explanation of what was optimized and why"
}}

**IMPORTANT**:
- For LLM_Generator nodes: Provide complete, detailed prompt templates with System Prompt: and User Prompt: markers
- For Python_Tool nodes: Provide complete, runnable Python code with all necessary imports and error handling
- For Retrieval_RAG nodes: Optimize both the retrieval query construction and the summarization prompt
- The optimized_all_code MUST be complete, runnable, and follow the same structure as the original code
- If you add new LLM calls or modify existing ones, explain the reasoning in logic_description
- If you add retrieval constraints or filters, explain them in logic_description
"""
    
    return system_prompt, user_prompt

