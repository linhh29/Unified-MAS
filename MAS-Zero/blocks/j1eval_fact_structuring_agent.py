import inspect
import json

# %%%%%%%%%%%%%%%%%%%% Fact_Structuring_Agent %%%%%%%%%%%%%%%%%%%%
def forward(self, taskInfo):
    # Step 1: Process input from taskInfo
    content = taskInfo.content
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except:
            pass
            
    # Extract raw_case_data similar to original logic
    raw_data = content.get("raw_case_data", {}) if isinstance(content, dict) else content

    # Step 2: Prepare Instructions
    # Combine system prompt and user prompt into a single instruction for the agent
    system_instruction = (
        "You are a professional Judicial Assistant in a Chinese Civil Court. "
        "Your task is to structure case data for a Judge. "
        "You must be strictly objective and neutral. "
        "Focus on extracting precise financial figures, dates, and contractual obligations. "
        "Do NOT invent facts not present in the input."
    )
    
    # We embed the data requirement in the prompt, but the data itself is passed via input
    task_instruction = f"""
Analyze the following civil case data provided in the input:
{json.dumps(raw_data, ensure_ascii=False)}

Task:
1. **Identify Parties**: Plaintiff (原告) and Defendant (被告).
2. **Extract Claims**: List each monetary claim.
   - **Crucial**: If a single claim string contains multiple components (e.g., "Principal X and Interest Y"), **split them** into separate entries.
   - Categories: "Principal" (本金), "Agreed Interest" (约定利息/贴息), "Late Penalty/Interest" (逾期利息/违约金), "Litigation Cost" (诉讼费).
   - For "Late Penalty/Interest", clearly describe the calculation basis (start date, rate, end date) in the description.
3. **Draft Fact Summary**: 
   - Write a neutral chronological narrative.
   - **Contract & Performance**: Mention contract signing, services provided, and settlement amounts.
   - **Financial Instruments**: Detail the Bills (Numbers, Amounts, Issue Date, Maturity Date).
   - **The Dispute**: 
     - State Plaintiff's version: Non-payment, calculation of specific interest amounts.
     - State Defendant's defense: **Specific arguments** about Statute of Limitations (时效), invalidity of interest claims, or calculation start dates.
   - **Evidence**: Briefly mention if evidence supports the existence of the underlying debt.

Output strictly in this JSON format:
{{
  "parties": {{"plaintiff": "Name", "defendant": "Name"}},
  "claims": [
    {{"category": "Principal", "amount": 181702.99, "description": "Unpaid bill amount..."}},
    {{"category": "Agreed Interest", "amount": 11810.69, "description": "Stick interest (贴息)..."}},
    {{"category": "Late Interest", "amount": 16850.5, "description": "Calculated from [Date]..."}}
  ],
  "fact_summary": "In August 2019, Plaintiff and Defendant signed..."
}}
"""

    full_instruction = system_instruction + "\n" + task_instruction

    # Instantiate LLM Agent with Chain-of-Thought capability
    agent = LLMAgentBase(['thinking', 'answer'], 'Fact Structuring Agent', model=self.node_model, temperature=0.0)

    # Step 3: LLM Call
    # Pass taskInfo as input; the instruction contains the specific data reference
    thinking, answer = agent([taskInfo], full_instruction)
    
    # Step 4: Parse Output
    response_text = answer.content
    try:
        # Handle potential markdown wrapping
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[0].strip()
        else:
            json_str = response_text
            
        output_json = json.loads(json_str)
    except:
        output_json = {"parties": {}, "claims": [], "fact_summary": ""}
        
    # Map to original output keys
    result_dict = {
        "structured_facts": output_json.get("fact_summary", ""),
        "plaintiff_claims_structured": output_json.get("claims", [])
    }

    # Return final answer as a single string (serialized JSON) or info object
    final_answer = self.make_final_answer(thinking, json.dumps(result_dict, ensure_ascii=False))
    return final_answer

func_string = inspect.getsource(forward)

FACT_STRUCTURING_AGENT = {
    "thought": "Parses raw case data into structured facts and claims using strict objective criteria and standardizes financial claims.",
    "name": "Fact_Structuring_Agent",
    "code": """{func_string}""".format(func_string=func_string)
}