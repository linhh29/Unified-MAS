import inspect
import json
import re

# %%%%%%%%%%%%%%%%%%%% Refinement_Agent %%%%%%%%%%%%%%%%%%%%
def forward(self, taskInfo):
    # ---------- Step 1: Process input ----------
    # Extract input data from taskInfo
    input_data = taskInfo.content if isinstance(taskInfo.content, dict) else {}
    if not input_data and isinstance(taskInfo.content, str):
        try:
            input_data = json.loads(taskInfo.content)
        except:
            input_data = {}

    draft = input_data.get('draft_itinerary', [])
    report = input_data.get('validation_report', {})
    context_raw = input_data.get('context')
    constraints = input_data.get('constraints', {})

    # Determine if we need to regenerate from scratch
    # If draft has errors or is empty, or validation failed, regenerate.
    needs_regeneration = False
    if not draft or (isinstance(draft, list) and len(draft) > 0 and 'error' in draft[0]):
        needs_regeneration = True
    elif not report.get('is_valid', False):
        needs_regeneration = True

    # If the draft is actually valid, just return it
    if not needs_regeneration:
        return self.make_final_answer("Draft is valid, no regeneration needed.", {"final_itinerary": draft})

    # ---------- Step 2: Prepare Context ----------
    # We MUST use raw context for regeneration to ensure we see all transport links
    if context_raw and isinstance(context_raw, (list, dict)):
        context_str = json.dumps(context_raw, indent=2)
    else:
        context_str = input_data.get('filtered_context', '')

    # ---------- Step 3: Extract Constraints ----------
    duration = constraints.get('duration_days', 7)
    budget = constraints.get('budget', 0)
    origin = constraints.get('origin_city', 'Origin')
    dest = constraints.get('destination_city', 'Destination')
    start_date = constraints.get('start_date', '')
    people = constraints.get('people_count', 1)
    
    accommodation_constraints = ""
    if "pets" in str(constraints).lower() or "pet" in str(constraints).lower():
        accommodation_constraints += "User likely has pets. EXCLUDE accommodations with 'No pets' in house_rules. "
    else:
        # Add a safe default or checking mechanism
        accommodation_constraints += "Check if pets are mentioned. "
    
    accommodation_constraints += "Prefer 'Entire home/apt' if strictly requested or typically preferred for groups."
    
    cuisine_preferences = "Mexican, Italian, Mediterranean, Indian, or local specialties"

    # ---------- Step 4: Construct Instruction ----------
    cot_instruction = f"""
    You are an expert travel planner. The user wants a {duration}-day trip for {people} people from {origin} to {dest} with a budget of ${budget}.

    Context Data:
    {context_str}

    Task:
    Generate a valid JSON itinerary based STRICTLY on the provided Context.
    
    User Request: Create the plan for {people} people, {duration} days, from {origin} to {dest}. Start date: {start_date}. Ensure all fields are filled using Context data strictly. Handle any multi-city travel if the data implies it.

    **CRITICAL: Route Discovery**
    The destination '{dest}' might be a region (e.g., 'Colorado'). You must scan the 'Description' fields in the Context for transport options starting from {origin} to find the actual entry city (e.g., 'from {origin} to [City]'). Then look for connections between cities (e.g., 'from [City] to [Next City]') to form a complete loop or path ending back at {origin}. Do not invent flights; if 'No flight' is listed, use 'Self-driving' or 'Taxi' found in the context.

    Rules:
    1. **Transport**: Use exact strings from context (e.g., "Self-driving, from ... cost: ..."). 
    2. **Budget**: Total cost must be under ${budget}. Sum: Transport + Accommodation (per night * nights) + Food (per person) + Attractions.
    3. **Accommodations**: 
       - Constraints: {accommodation_constraints}
       - Ensure the accommodation city matches the current city.
    4. **Dining**: Select restaurants matching these cuisines: {cuisine_preferences}.
    5. **Structure**: 
       - Plan for exactly {duration} days.
       - 'current_city' field: 'from [Start] to [End]' on travel days, otherwise '[City]'.
       - 'accommodation': '-' on the final return day.

    Output Format:
    Return a single JSON object (no markdown):
    {{
      "thought_process": "Identify route: Origin -> City A -> City B -> Origin. Calculate costs...",
      "itinerary": [
        {{
          "days": 1,
          "current_city": "from [Start] to [End]",
          "transportation": "Full string from context",
          "breakfast": "Name, City",
          "attraction": "Name, City; Name, City",
          "lunch": "Name, City",
          "dinner": "Name, City",
          "accommodation": "Name, City"
        }}
      ]
    }}
    """

    # Instantiate a new LLM agent specifically for CoT
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Refinement Agent',  model=self.node_model, temperature=0.0)

    # Prepare the inputs for the CoT agent
    cot_agent_inputs = [taskInfo]

    # ---------- Step 5: Call LLM ----------
    thinking, answer = cot_agent(cot_agent_inputs, cot_instruction)
    
    # Handle response format
    try:
        response_text = answer.content
        response_text = re.sub(r'```json|```', '', response_text).strip()
        parsed_json = json.loads(response_text)
        
        final_plan = []
        if 'itinerary' in parsed_json and isinstance(parsed_json['itinerary'], list):
            final_plan = parsed_json['itinerary']
        elif isinstance(parsed_json, list):
            final_plan = parsed_json
        else:
             # Fallback search
            for key, value in parsed_json.items():
                if isinstance(value, list):
                    final_plan = value
                    break
        
        if not final_plan:
            # If still empty, return error
            result = {"final_itinerary": [{"error": "Failed to parse itinerary from LLM response."}]}
        else:
            result = {"final_itinerary": final_plan}

    except Exception as e:
        print(f"Refinement Agent Error: {e}")
        result = {"final_itinerary": [{"error": f"Error generating itinerary: {str(e)}"}]}

    final_answer = self.make_final_answer(thinking.content, result)
    return final_answer

func_string = inspect.getsource(forward)

REFINEMENT_AGENT = {
    "thought": "Refines the itinerary or generates a new one from scratch if the draft is invalid, handling multi-city routing and constraints. Uses Chain-of-Thought to plan routes and budget.",
    "name": "Refinement_Agent",
    "code": """{func_string}""".format(func_string=func_string)
}