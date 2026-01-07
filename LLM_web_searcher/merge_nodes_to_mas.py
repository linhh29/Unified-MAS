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

# travelplanner, hosp_summ, j1eval
DATANAME = "travelplanner"
mas_option = "MAS-Zero"  # This will be replaced by command line argument or user input

# Configuration
if DATANAME == "j1eval":
    NODES_JSON_PATH = f"/data/qin/lhh/Unified-MAS/LLM_web_searcher/intermediate_result/{DATANAME}/optimize/rounds/round16_generated_nodes.json"
elif DATANAME == "hosp_summ":
    NODES_JSON_PATH = f"/data/qin/lhh/Unified-MAS/LLM_web_searcher/intermediate_result/{DATANAME}/optimize/rounds/round32_generated_nodes.json"
elif DATANAME == "travelplanner":
    NODES_JSON_PATH = f"/data/qin/lhh/Unified-MAS/LLM_web_searcher/intermediate_result/{DATANAME}/optimize/rounds/round45_generated_nodes.json"

def get_block_path(option: str) -> str:
    """Get the reference block path based on the option"""
    if option == "MAS-Zero":
        return "/data/qin/lhh/Unified-MAS/MAS-Zero/blocks/cot.py"

OUTPUT_BASE_DIR = f"/data/qin/lhh/Unified-MAS/LLM_web_searcher/intermediate_result/{DATANAME}/merged_nodes"

# LLM configuration for migration assistance
LLM_MODEL = "gemini-3-pro-preview"
OPENAI_API_KEY = 'sk-BDrpp8zrYLtMWyfY2YZJZZPjIOXwikCyZFfDWL8eUGDqnts2'
OPENAI_API_BASE = 'https://api.qingyuntop.top/v1'

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
    block_path = get_block_path(option)
    
    # Read reference block as example
    with open(block_path, 'r', encoding='utf-8') as f:
        block_example = f.read()
    
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

Return ONLY the JSON object, no additional text or markdown."""

    client = get_llm_client()
    
    try:
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
        
        # Parse JSON response
        try:
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
        except json.JSONDecodeError as e:
            print(f"  Warning: Failed to parse JSON response, trying to extract code directly: {e}")
            # Fallback: try to extract code if JSON parsing fails
            migrated_code = response_content.strip()
            if migrated_code.startswith("```python"):
                migrated_code = migrated_code[9:]
            if migrated_code.startswith("```"):
                migrated_code = migrated_code[3:]
            if migrated_code.endswith("```"):
                migrated_code = migrated_code[:-3]
            return migrated_code.strip()
    except Exception as e:
        print(f"Error in LLM migration for {node_name}: {e}")
        # Re-raise the exception since we don't have a fallback
        raise


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

