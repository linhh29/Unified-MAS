import inspect
import re
import json

# %%%%%%%%%%%%%%%%%%%% Damages_Calculator %%%%%%%%%%%%%%%%%%%% 
def forward(self, taskInfo):
    # Parse input data from taskInfo
    try:
        if isinstance(taskInfo.content, dict):
            input_data = taskInfo.content
        elif isinstance(taskInfo.content, str):
            input_data = json.loads(taskInfo.content)
        else:
            input_data = {}
    except:
        input_data = {}

    # Helper to clean and parse floats
    def parse_float(val):
        try:
            clean_val = str(val).replace(',', '')
            match = re.search(r"([-+]?\d*\.?\d+)", clean_val)
            return float(match.group(1)) if match else 0.0
        except:
            return 0.0

    # Step 1: Extract basic inputs
    claims = input_data.get("plaintiff_claims_structured", [])
    direct_ratio = input_data.get("defendant_liability_ratio", None)
    accepted_cats = input_data.get("accepted_claim_categories", [])
    
    # Extract text for analysis
    analysis_source = input_data.get("Liability_Assessment_Agent", {})
    if isinstance(analysis_source, dict):
        analysis_text = analysis_source.get("liability_analysis", "")
    else:
        analysis_text = str(analysis_source)
    if not analysis_text:
        analysis_text = input_data.get("liability_analysis", "")
        
    analysis_lower = analysis_text.lower()
    defense_text = input_data.get("defendant_defence", "").lower()

    # Step 2: Smart Liability Ratio Determination
    # Default to 1.0 (100%) unless evidence suggests otherwise
    ratio = 1.0 
    ratio_source = "Default"

    # Priority 1: Structured Input
    if direct_ratio is not None and str(direct_ratio).strip() != "":
        try:
            r = float(direct_ratio)
            if 0 <= r <= 1.0:
                ratio = r
                ratio_source = "Structured Input"
        except:
            pass
            
    # Priority 2: Regex Extraction from Analysis (e.g., "liability: 50%", "fault: 70%")
    if ratio_source == "Default":
        # Look for explicit percentage assignments
        pct_match = re.search(r"(?:liability|fault|responsibility|share).*?(\d{1,3})%", analysis_text, re.IGNORECASE)
        if pct_match:
            r = float(pct_match.group(1))
            if 0 <= r <= 100:
                ratio = r / 100.0
                ratio_source = "Analysis Regex"
        
    # Priority 3: Keyword Analysis (Mutual Fault inference)
    if ratio_source == "Default":
        # Keywords implying 50/50 or shared fault
        mutual_keywords = [
            "mutual fault", "both parties", "equal responsibility", "shared liability", "joint liability",
            "mixed fault", "contributory negligence", "同等责任", "双方均有", "各半", "willing to pay half"
        ]
        secondary_keywords = ["secondary", "minor", "subordinate", "次要", "30%"]
        
        # Check Analysis first
        if any(kw in analysis_lower for kw in mutual_keywords):
            ratio = 0.5
            ratio_source = "Analysis Keywords (Mutual)"
        elif any(kw in analysis_lower for kw in secondary_keywords):
            ratio = 0.3
            ratio_source = "Analysis Keywords (Secondary)"
        # Fallback: Check Defense if Analysis is silent/ambiguous but Defendant admits partial
        elif any(kw in defense_text for kw in ["pay half", "承担一半", "half responsibility"]):
            ratio = 0.5
            ratio_source = "Defense Fallback"

    # Step 3: Global Rejection Check
    rejection_keywords = [
        "no causal", "lack of causal", "not liable", "dismiss all", 
        "insufficient evidence to prove liability", "unrelated to defendant",
        "无因果", "不承担责任", "驳回全部"
    ]
    force_reject_all = False
    for kw in rejection_keywords:
        if kw in analysis_lower and "not " + kw not in analysis_lower: # Simple negation check
            force_reject_all = True
            break

    # Step 4: Calculate Valid Amounts
    total_loss = 0.0
    details = []
    accepted_cats_norm = [str(c).lower().strip() for c in accepted_cats]
    
    # Ensure claims is a list
    if not isinstance(claims, list):
        claims = []

    for claim in claims:
        cat = str(claim.get('category', '')).strip()
        cat_lower = cat.lower()
        claimed_amt = parse_float(claim.get('amount', 0))
        
        # Determine Acceptance
        is_accepted = False
        if not accepted_cats: 
            # If no accepted list provided, assume all non-rejected are valid unless global reject
            is_accepted = True 
        else:
            for ac in accepted_cats_norm:
                if ac in cat_lower or cat_lower in ac:
                    is_accepted = True
                    break
        
        # Override: Look for specific "recognized" amounts in analysis text
        # Pattern: "Category... recognized/valid... X amount"
        recognized_amt = claimed_amt
        if is_accepted:
            try:
                # Simplified heuristic to find verified amounts in analysis text
                # Looks for the category name followed by validation keywords and a number
                safe_cat = re.escape(cat_lower.split()[0]) if cat_lower else ""
                if safe_cat:
                    pattern = rf"{safe_cat}[^\d\n]*?(?:recognized|valid|support|pay|认定|支持|confirm)[^\d\n]*?(\d+\.?\d+)"
                    override_match = re.search(pattern, analysis_lower)
                    if override_match:
                        recognized_amt = float(override_match.group(1))
            except:
                pass

        if force_reject_all:
            is_accepted = False
            status = "Rejected (Global Liability)"
        elif not is_accepted:
            status = "Rejected (Category)"
        else:
            status = "Accepted"
            
        if is_accepted:
            total_loss += recognized_amt
            details.append(f"{cat}: {recognized_amt:.2f}")
        else:
            details.append(f"{cat}: Rejected")
            
    # Step 5: Final Calculation
    final_amt = total_loss * ratio
    
    calc_str = f"Total Recognized Loss: {total_loss:.2f} * Liability {ratio*100:.0f}% ({ratio_source}) = {final_amt:.2f}. Breakdown: {'; '.join(details)}"

    result = {
        "final_compensation_amount": round(final_amt, 2),
        "calculation_details": calc_str,
        "liability_ratio": ratio
    }
    
    return json.dumps(result)

func_string = inspect.getsource(forward)

DAMAGES_CALCULATOR = {
    "thought": "This node performs deterministic calculations for damages based on liability analysis and claim structure. I extracted the input data from the taskInfo object and preserved the regex-based logic for liability ratio determination and amount verification. The final result is returned as a JSON string to satisfy the single return value requirement.",
    "name": "Damages_Calculator",
    "code": """{func_string}""".format(func_string=func_string)
}