import inspect
import json

# %%%%%%%%%%%%%%%%%%%% LIABILITY_ASSESSMENT_AGENT %%%%%%%%%%%%%%%%%%%%
def forward(self, taskInfo):
    # Step 1: Process input
    # Extract data from taskInfo.content (dict or JSON string)
    input_data = taskInfo.content if isinstance(taskInfo.content, dict) else json.loads(taskInfo.content) if isinstance(taskInfo.content, str) else {}
    
    facts = input_data.get("structured_facts", "")
    laws = input_data.get("relevant_laws", "")
    claims = input_data.get("plaintiff_claims_structured", [])
    
    # Step 2: LLM Call Preparation
    system_prompt = """You are a Senior Civil Judge in China. Your task is to determine liability and assess the *valid amount* for each claim based on Facts and Defendant's Defense.

**Liability Logic**:
- Identify the party at fault for the dispute (e.g., Breach of Promise, Contract Breach, Tort).
- In Betrothal/Caili cases: If the receiver unilaterally refuses to marry, they are typically 100% liable to return the Caili.

**Damages Assessment Logic**:
1. **Admitted vs Disputed**: Specific amounts admitted by the Defendant (e.g., "I acknowledge receiving 50,000") must be Approved.
2. **Defense Rebuttal**: If the Defendant disputes part of a claim (e.g., "The ring is actually with the Plaintiff," "I paid for this jewelry myself"), and the Plaintiff lacks conclusive counter-evidence in the facts, **DEDUCT** the disputed amount.
3. **Cash vs Property**: If Plaintiff claims a lump sum for *Cash + Goods (Jewelry)*:
   - Approve the undisputed Cash part.
   - **Strictly Scrutinize** the Goods part. If ownership/possession of items is disputed or if the remedy should be 'return of items' rather than 'cash value', **REJECT** the monetary claim for the goods.
4. **Final Amount**: If a claim category includes invalid parts, output the *Corrected/Reduced Amount* based on what is legally and factually supported.
"""

    # Reconstruct the specific user prompt formatting from the original code
    user_prompt = f"Facts (includes Defendant's Defense):\n{facts}\n\nLaws:\n{laws}\n\nPlaintiff Claims:\n{json.dumps(claims, ensure_ascii=False)}\n\nTask:\n1. Liability: Determine Fault (0.0 - 1.0).\n2. Claim Assessment: For each claim category, determine the 'approved' amount. If the claimed amount (e.g. 81603) includes disputed items (e.g. jewelry value) that are not validly proven for cash compensation, **REDUCE** the amount to the valid portion (e.g. 50000).\n\nOutput JSON:\n{{\n  \"reasoning\": \"Step-by-step legal analysis...\",\n  \"defendant_liability_ratio\": 1.0,\n  \"claim_assessments\": {{\n    \"Category Name from Input\": {{\"approved\": true, \"amount\": 50000.00, \"note\": \"Reasoning for adjustment...\"}}\n  }}\n}}"

    # Update taskInfo content to serve as the specific input for the Agent
    taskInfo.content = user_prompt

    # Instantiate the LLM Agent
    # We use LLMAgentBase to support Chain-of-Thought (thinking) and the final Answer
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Liability Assessment Agent', model=self.node_model, temperature=0.0)

    # Get the response from the agent
    # We pass the formatted user_prompt via taskInfo and the system_prompt as instruction
    thinking, answer = cot_agent([taskInfo], system_prompt)
    
    # Step 3: Collect output and Update Claims
    try:
        res_json = json.loads(answer.content)
    except:
        res_json = {"reasoning": "Analysis failed.", "defendant_liability_ratio": 0.0, "claim_assessments": {}}
        
    liability_ratio = res_json.get("defendant_liability_ratio", 0.0)
    assessments = res_json.get("claim_assessments", {})
    
    accepted_categories = []
    updated_claims = []
    
    # Update the claims structure with the approved amounts
    for claim in claims:
        cat = claim.get("category")
        if cat in assessments and assessments[cat].get("approved", False):
            accepted_categories.append(cat)
            new_claim = claim.copy()
            new_claim["amount"] = assessments[cat].get("amount", 0.0)
            new_claim["note"] = assessments[cat].get("note", "Approved")
            updated_claims.append(new_claim)
        else:
            pass
            
    result = {
        "liability_analysis": res_json.get("reasoning", ""),
        "defendant_liability_ratio": liability_ratio,
        "accepted_claim_categories": accepted_categories,
        "plaintiff_claims_structured": updated_claims
    }
    
    # Return the final answer using the MAS-Zero helper
    final_answer = self.make_final_answer(thinking.content, json.dumps(result, ensure_ascii=False))
    return final_answer

func_string = inspect.getsource(forward)

LIABILITY_ASSESSMENT_AGENT = {
    "thought": "Determines liability ratio and validates claim amounts based on evidence strength and defendant's defense.",
    "name": "Liability_Assessment_Agent",
    "code": """{func_string}""".format(func_string=func_string)
}