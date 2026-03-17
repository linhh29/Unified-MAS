"""
Script to migrate nodes from JSON format to target block format.
Reads nodes from JSON file and converts them to block format similar to cot.py
"""

import json
import os
import inspect
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI
import re

# travelplanner, hosp_summ, j1eval, deepfund, healthbench, aime
DATANAME = "healthbench"
mas_option = "AFlow"  # This will be replaced by command line argument or user input

# Configuration
if DATANAME == "j1eval":
    NODES_JSON_PATH = f"xx/{DATANAME}/optimize/rounds/epoch_10_generated_nodes.json"
elif DATANAME == "hosp_summ":
    NODES_JSON_PATH = f"xx/{DATANAME}/optimize/rounds/epoch_10_generated_nodes.json"
elif DATANAME == "travelplanner":
    NODES_JSON_PATH = f"xx/{DATANAME}/optimize/rounds/epoch_10_generated_nodes.json"
elif DATANAME == "deepfund":
    NODES_JSON_PATH = f"xx/{DATANAME}/optimize/rounds/epoch_10_generated_nodes.json"
elif DATANAME == "healthbench":
    NODES_JSON_PATH = f"xx/{DATANAME}/optimize/rounds/epoch_10_generated_nodes.json"
elif DATANAME == "aime":
    NODES_JSON_PATH = f"xx/{DATANAME}/optimize/rounds/epoch_10_generated_nodes.json"

def get_block_path(option: str) -> tuple:
    """Get the reference block path based on the option"""
    if option == "MAS-Zero":
        block_path = "xx/cot.py"
        # Read reference block as example
        with open(block_path, 'r', encoding='utf-8') as f:
            block_example = f.read()

        key_concepts = """
        - `taskInfo` is an Info object with a `.content` attribute that contains the actual data
- `LLMAgentBase` is used for LLM calls: `LLMAgentBase(output_fields, name, model=self.node_model, temperature=0.0)`
- LLM agents are called like: `thinking, answer = agent(input_list, instruction)`
- The agent returns Info objects with `.content` attribute
- Use `self.make_final_answer(thinking, answer)` to format the final output
- The function should return a single value (string or Info object), not a dictionary

**Migration Requirements (CRITICAL - Follow these exactly):**

1. **Function Signature**: Convert `def {node_name}(self, input_data):` to `def forward(self, taskInfo):`

2. **Input Extraction**: 
   - Extract data from `taskInfo.content` (which may be a dict, string, or JSON)
   - If the original function expects `input_data.get("key")`, use: 
     ```python
     task_data = taskInfo.content if isinstance(taskInfo.content, dict) else json.loads(taskInfo.content) if isinstance(taskInfo.content, str) else {{}}
     ```
   - Map the original input keys ({inputs}) to extracted data from taskInfo

3. **LLM Client Replacement**:
   - If the code uses `self.llm_client.chat(messages, response_format='...')`, replace with:
     ```python
     agent = LLMAgentBase(['answer'], 'Agent Name', model=self.node_model, temperature=0.0)
     # Convert messages to instruction string or Info objects
     answer = agent([taskInfo], instruction)[1]  # Get answer (second return value)
     response = answer.content
     ```
   - For JSON responses, parse `answer.content` as JSON: `response_json = json.loads(response)`

4. **Search Engine Replacement**:
   - If the code uses `self.search_engine.multi_turn_search(query)`, you may need to:
     - Use {option}'s search capabilities if available
     - Or adapt to use LLM-based retrieval
     - Or comment out and add a TODO note

5. **Return Value**:
   - The original function returns: {outputs}
   - Convert to return a single value (string or Info object)
   - If multiple outputs, combine them into a single string or return the primary output
   - Use `self.make_final_answer()` if you have thinking and answer components

6. **Code Structure**:
   - Preserve all the core logic and calculations
   - Keep all necessary imports
   - Ensure the code is syntactically correct
   - Handle edge cases and errors gracefully

**Example Conversion Pattern:**
Original: `response = self.llm_client.chat(messages, response_format='json_object')`
Converted: 
```python
agent = LLMAgentBase(['answer'], '{node_name} Agent', model=self.node_model, temperature=0.0)
# Build instruction from messages
instruction = messages[0]['content'] if messages else ""
answer = agent([taskInfo], instruction)[1]
response = answer.content
# Parse JSON if needed: response_json = json.loads(response)
```

**Output Format:**
Provide the complete migrated code in the exact format shown in the COT example, including:
1. The `forward` function
2. The `func_string = inspect.getsource(forward)` line
3. The dictionary with "thought", "name", and "code" keys
4. Use the node name (converted to a valid Python variable name) as the dictionary variable name

**Output Format:**
You must return a JSON object with exactly two fields:
- `thinking`: A string explaining your migration approach, key decisions, and how you adapted the code to the target format
- `answer`: A string containing the complete migrated Python code (the entire block file content)

Example JSON format:
{{
    "thinking": "I converted the function signature from def NodeName(self, input_data) to def forward(self, taskInfo). I extracted input data from taskInfo.content and replaced self.llm_client with LLMAgentBase following the reference format...",
    "answer": "import inspect\\n\\n# %%%%%%%%%%%%%%%%%%%% NodeName %%%%%%%%%%%%%%%%%%%%\\ndef forward(self, taskInfo):\\n    ...\\n\\nfunc_string = inspect.getsource(forward)\\n\\nNODE_NAME = {{...}}\\n"
}}

Return ONLY the JSON object, no additional text or markdown.
        """
        return block_example, key_concepts
    elif option == "AFlow":
        block_example = """
        # For each operator, you need to define three things
        # 1. the operator class forward
        class Custom(Operator):
            def __init__(self, llm: AsyncLLM, name: str = "Custom"):
                super().__init__(llm, name)

            async def __call__(self, input, instruction):
                prompt = instruction + input
                response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
                return response
                
        # 2. the operator class output schema
        class GenerateOp(BaseModel):
            response: str = Field(default="", description="Your solution for this problem")

        # 3. a json description and interface of the operator
        "Custom": {
            "description": "Generates anything based on customized input and instruction.",
            "interface": "custom(input: str, instruction: str) -> dict with key 'response' of type str"
        }


        # 1. the operator class forward
        class AnswerGenerate(Operator):
            def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
                super().__init__(llm, name)

            async def __call__(self, input: str) -> Tuple[str, str]:
                prompt = ANSWER_GENERATION_PROMPT.format(input=input)
                response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
                return response
                
        # 2. the operator class output schema
        class AnswerGenerateOp(BaseModel):
            thought: str = Field(default="", description="The step by step thinking process")
            answer: str = Field(default="", description="The final answer to the question")


        # 3. a json description and interface of the operator
        "AnswerGenerate": {
            "description": "Generate step by step based on the input. The step by step thought process is in the field of 'thought', and the final answer is in the field of 'answer'.",
            "interface": "answer_generate(input: str) -> dict with key 'thought' of type str, 'answer' of type str"
        }

        # 1. the operator class forward
        class ScEnsemble(Operator):
            def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
                super().__init__(llm, name)

            async def __call__(self, solutions: List[str], problem: str):
                answer_mapping = {}
                solution_text = ""
                for index, solution in enumerate(solutions):
                    answer_mapping[chr(65 + index)] = index
                    solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

                prompt = SC_ENSEMBLE_PROMPT.format(question=problem, solutions=solution_text)
                response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

                answer = response.get("solution_letter", "")
                answer = answer.strip().upper()

                return {"response": solutions[answer_mapping[answer]]}
                
        # 2. the operator class output schema
        class ScEnsembleOp(BaseModel):
            thought: str = Field(default="", description="The thought of the most consistent solution.")
            solution_letter: str = Field(default="", description="The letter of most consistent solution.")


        # 3. a json description and interface of the operator
        "ScEnsemble": {
            "description": "Uses self-consistency to select the solution that appears most frequently in the solution list, improve the selection to enhance the choice of the best solution.",
            "interface": "sc_ensemble(solutions: List[str]) -> dict with key 'response' of type str"
        }
        """
        
        key_concepts = """
        - `Operator` is the base class that all operators inherit from
        - `AsyncLLM` is passed to the operator's `__init__` method as `llm` parameter
        - `_fill_node()` is the method used to call LLM with structured output formatting
        - `BaseModel` from Pydantic is used to define output schemas
        - `Field()` is used to define schema fields with descriptions
        - Each operator requires THREE components: Operator class, BaseModel schema class, and JSON description

        **Migration Requirements (CRITICAL - Follow these exactly):**

        1. **Three Required Components**:
           Each operator MUST have exactly three parts defined in this order:
           a) Operator class (inherits from `Operator`)
           b) BaseModel schema class (inherits from `BaseModel`)
           c) JSON description dictionary (as a string key-value pair)

        2. **Operator Class Structure**:
           - Class name: Convert `{node_name}` to PascalCase (e.g., "answer_generate" -> "AnswerGenerate")
           - Inheritance: MUST inherit from `Operator`
           - `__init__` method: 
             ```python
             def __init__(self, llm: AsyncLLM, name: str = "{OperatorName}"):
                 super().__init__(llm, name)
             ```
             - First parameter MUST be `llm: AsyncLLM`
             - Second parameter MUST be `name: str` with default value matching the class name
             - MUST call `super().__init__(llm, name)`
           
           - `__call__` method:
             ```python
             async def __call__(self, {input_params}):
                 # Your logic here
                 response = await self._fill_node(SchemaOp, prompt, mode="...")
                 return response
             ```
             - MUST be `async def`
             - Parameters correspond to the original function's inputs ({inputs})
             - Parameter types should match: `str`, `List[str]`, `int`, etc.
             - MUST use `await self._fill_node(SchemaOp, prompt, mode="...")` for LLM calls
             - Return the response directly (dict or tuple)

        3. **LLM Call Pattern**:
           - Replace any direct LLM calls with `self._fill_node()`:
             ```python
             response = await self._fill_node(SchemaOp, prompt, mode="xml_fill")
             ```
           - `SchemaOp` is the BaseModel schema class (see below)
           - `prompt` is the formatted prompt string
           - `mode` options:
             - `"single_fill"`: For single string output (returns dict with "response" key)
             - `"xml_fill"`: For structured output with multiple fields (returns dict with schema fields)
             - `"code_fill"`: For code generation (requires `function_name` parameter)
           - The response is a dictionary with keys matching the BaseModel schema fields

        4. **BaseModel Schema Class**:
           - Class name: Operator class name + "Op" suffix (e.g., "AnswerGenerate" -> "AnswerGenerateOp")
           - MUST inherit from `BaseModel`
           - Field definitions:
             ```python
             class SchemaOp(BaseModel):
                 field_name: str = Field(default="", description="Field description")
             ```
           - Each field MUST have:
             - Type annotation (str, int, List[str], etc.)
             - `Field(default="", description="...")` with meaningful description
           - Field names MUST match the keys in the returned dictionary from `__call__`
           - If returning a single value, use field name "response"
           - If returning multiple values, use descriptive field names (e.g., "thought", "answer")

        5. **Input Parameter Mapping**:
           - Original function inputs ({inputs}) become `__call__` method parameters
           - Extract inputs directly from parameters (no need for dict extraction)
           - If original code uses `input_data.get("key")`, convert to direct parameter: `def __call__(self, key: str)`
           - Parameter types should be explicit: `input: str`, `solutions: List[str]`, `problem: str`

        6. **Prompt Construction**:
           - Build prompt string from inputs and any template prompts
           - Use `.format()` for string formatting: `PROMPT_TEMPLATE.format(input=input, other=other)`
           - Concatenate strings if needed: `prompt = instruction + input`
           - Pass the final prompt string to `_fill_node()`

        7. **Return Value Handling**:
           - `_fill_node()` returns a dictionary with keys matching the BaseModel schema
           - For `mode="single_fill"`: Returns `{{"response": "..."}}`
           - For `mode="xml_fill"`: Returns dict with all schema fields (e.g., `{{"thought": "...", "answer": "..."}}`)
           - You can return the response directly: `return response`
           - Or extract and transform: `return {{"response": response.get("key")}}`
           - If original function returns multiple values, combine them into a dict or return the primary value

        8. **JSON Description Dictionary**:
           - Format: `"{OperatorName}": {{"description": "...", "interface": "..."}}`
           - Key: Operator name as string (same as class name)
           - "description": Clear description of what the operator does
           - "interface": Function signature format: `"operator_name(input: type) -> dict with key 'field' of type type"`
           - Interface should match the actual `__call__` signature and return type

        9. **Code Structure Requirements**:
           - Preserve all core logic and calculations from original code
           - Keep all necessary imports (AsyncLLM, Operator, BaseModel, Field, List, Tuple, etc.)
           - Maintain error handling and edge cases
           - Ensure code is syntactically correct and follows Python async/await patterns
           - All three components (Operator class, Schema class, JSON description) must be present

        10. **Mode Selection Guide**:
            - Use `mode="single_fill"` when:
              - Output is a single string value
              - Schema has only one field named "response"
            - Use `mode="xml_fill"` when:
              - Output has multiple structured fields
              - Schema has multiple fields (e.g., "thought" and "answer")
            - Use `mode="code_fill"` when:
              - Generating executable code
              - Requires `function_name` parameter: `mode="code_fill", function_name=entry_point`

        **Output Format:**
        You must return a JSON object with exactly two fields:
        - `thinking`: A string explaining your migration approach, key decisions, and how you adapted the code to the AFlow format
        - `answer`: A string containing the complete migrated Python code with ALL THREE components:
          1. The Operator class (with `__init__` and `__call__` methods)
          2. The BaseModel schema class (with Field definitions)
          3. The JSON description dictionary (as a string key-value pair)

        Example JSON format:
        {{
            "thinking": "I converted the function to AFlow format by creating an Operator class inheriting from Operator, a BaseModel schema class, and a JSON description. I replaced LLM calls with _fill_node() using xml_fill mode for structured output...",
            "answer": "from pydantic import BaseModel, Field\\nfrom typing import List\\nfrom scripts.async_llm import AsyncLLM\\nfrom scripts.operators import Operator\\n\\nclass NodeName(Operator):\\n    def __init__(self, llm: AsyncLLM, name: str = \\\"NodeName\\\"):\\n        super().__init__(llm, name)\\n\\n    async def __call__(self, input: str):\\n        prompt = f\\\"Process: {{input}}\\\"\\n        response = await self._fill_node(NodeNameOp, prompt, mode=\\\"single_fill\\\")\\n        return response\\n\\nclass NodeNameOp(BaseModel):\\n    response: str = Field(default=\\\"\\\", description=\\\"Output description\\\")\\n\\n\\\"NodeName\\\": {{\\n    \\\"description\\\": \\\"Operator description\\\",\\n    \\\"interface\\\": \\\"node_name(input: str) -> dict with key 'response' of type str\\\"\\n}}"
        }}

        Return ONLY the JSON object, no additional text or markdown.
        """
        return block_example, key_concepts

OUTPUT_BASE_DIR = f"xx/{DATANAME}/merged_nodes"

# LLM configuration for migration assistance
LLM_MODEL = "gemini-3-pro-preview"
OPENAI_API_KEY = 'xx'
OPENAI_API_BASE = 'xx'

# Initialize LLM client
_llm_client = None

def get_llm_client():
    """Initialize and return LLM client"""
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    return _llm_client


def migrate_node_with_llm(node_data: Dict, option: str) -> str:
    """
    Use LLM to help migrate a node from JSON format to target block format.
    
    Args:
        node_data: Dictionary containing node information (node_name, description, all_code, etc.)
        option: Migration option (e.g., "MAS-Zero", "mas_zero", etc.)
    
    Returns:
        str: Migrated code in block format
    """
    node_name = node_data.get("node_name", "")
    description = node_data.get("description", "")
    all_code = node_data.get("all_code", "")
    node_type = node_data.get("node_type", "")
    inputs = node_data.get("input", [])
    outputs = node_data.get("output", [])
    
    # Get block path based on option
    block_example, key_concepts = get_block_path(option)
    
    
    
    # Create migration prompt with option-specific references
    migration_prompt = f"""You are a code migration expert. Your task is to convert a node function to {option} block format.

**Reference Block Format:**
```python
{block_example}
```

**Source Node Information:**
- Node Name: {node_name}
- Node Type: {node_type}
- Description: {description}
- Inputs: {inputs}
- Outputs: {outputs}

**Source Code:**
```python
{all_code}
```

**Key {option} Concepts:**
{key_concepts}"""

    client = get_llm_client()
    

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert Python code migration specialist. Convert code accurately while preserving functionality. Always return your response as a valid JSON object."},
            {"role": "user", "content": migration_prompt}
        ],
        temperature=1.0,
        response_format={"type": "json_object"},
        reasoning_effort="high",
    )
    
    response_content = response.choices[0].message.content.strip()
    
    response_json = json.loads(response_content)
    thinking = response_json.get("thinking", "")
    migrated_code = response_json.get("answer", "")
    
    # Print thinking for debugging
    if thinking:
        print(f"  [Thinking] {thinking[:200]}..." if len(thinking) > 200 else f"  [Thinking] {thinking}")
    
    # Clean up the code (remove markdown code blocks if present)
    if migrated_code.startswith("```python"):
        migrated_code = migrated_code[9:]
    if migrated_code.startswith("```"):
        migrated_code = migrated_code[3:]
    if migrated_code.endswith("```"):
        migrated_code = migrated_code[:-3]
    migrated_code = migrated_code.strip()
    
    if not migrated_code:
        raise ValueError("Empty answer field in JSON response")
    
    return migrated_code


def sanitize_filename(name: str) -> str:
    """Convert node name to a valid filename"""
    # Replace spaces and special characters
    filename = name.replace(" ", "_").replace("-", "_")
    # Remove any remaining invalid characters
    filename = re.sub(r'[^\w\-_]', '', filename)
    return filename.lower() + ".py"


def migrate_nodes(option: str = "mas_zero"):
    """
    Main function to migrate all nodes from JSON to target block format.
    
    Args:
        option: Migration option (e.g., "mas_zero", "MAS-Zero", etc.) - creates a subdirectory with this name
    """
    # Read nodes JSON
    print(f"Reading nodes from: {NODES_JSON_PATH}")
    with open(NODES_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = data.get("nodes", [])
    print(f"Found {len(nodes)} nodes to migrate")
    
    # Create output directory
    output_dir = Path(OUTPUT_BASE_DIR) / option
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Migrate each node
    for i, node in enumerate(nodes, 1):
        node_name = node.get("node_name", f"Node_{i}")
        print(f"\n[{i}/{len(nodes)}] Migrating node: {node_name}")
    
        # Use LLM to migrate the node
        migrated_code = migrate_node_with_llm(node, option)
        
        # Save to file
        filename = DATANAME + "_" + sanitize_filename(node_name)
        output_path = output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(migrated_code)
        
        print(f"  ✓ Saved to: {output_path}")
        
    
    print(f"\n✓ Migration complete! Files saved to: {output_dir}")


def main():
    """Main entry point"""
    
    print(f"Starting migration with option: {mas_option}")
    migrate_nodes(mas_option)


if __name__ == "__main__":
    main()

