import inspect

# %%%%%%%%%%%%%%%%%%%% Validation_Tool %%%%%%%%%%%%%%%%%%%% 
def forward(self, taskInfo):
    import re
    import math
    import json

    # --- 1. Input Extraction ---
    # Extract data from taskInfo.content
    # The content might be a dict or a JSON string
    content = taskInfo.content
    if isinstance(content, dict):
        input_data = content
    elif isinstance(content, str):
        try:
            input_data = json.loads(content)
        except:
            input_data = {}
    else:
        input_data = {}

    # --- Helper: Clean Text ---
    def clean_text(text):
        if not text: return ""
        text = str(text)
        # Remove text in parentheses e.g. "City (State)" -> "City "
        text = re.sub(r'\(.*?\)', '', text)
        # Remove punctuation for cleaner matching, except hyphens/spaces
        text = re.sub(r'[^\w\s-]', '', text)
        return text.strip().lower()

    plan = input_data.get('draft_itinerary', [])
    constraints = input_data.get('constraints', {})
    context_str = input_data.get('filtered_context', '')

    # Default values if constraints are missing
    try:
        budget = float(constraints.get('budget', 0) or 0)
        people_count = int(constraints.get('people_count', 1) or 1)
        duration_days = int(constraints.get('duration_days', 0) or 0)
    except:
        budget = 10000
        people_count = 1
        duration_days = 0

    # --- 2. Initial Structure & Error Check ---
    errors = []
    
    # Check if plan is a list
    if not isinstance(plan, list):
        return self.make_final_answer("Invalid Input", {
            "validation_report": {
                "is_valid": False,
                "errors": ["Plan format is invalid (not a list)."],
                "final_cost": 0
            },
            "calculated_cost": 0
        })
    
    # Check for empty plan
    if not plan:
        return self.make_final_answer("Empty Plan", {
            "validation_report": {
                "is_valid": False,
                "errors": ["Plan is empty."],
                "final_cost": 0
            },
            "calculated_cost": 0
        })

    # Check for error messages in the plan content
    first_item = plan[0]
    if isinstance(first_item, dict):
        # Case-insensitive check for error keys
        for k, v in first_item.items():
            if 'error' in k.lower() or 'message' in k.lower():
                # If the value looks like an error description
                if isinstance(v, str) and len(v) > 10:
                    return self.make_final_answer("Planner Error", {
                        "validation_report": {
                            "is_valid": False,
                            "errors": [f"Planner returned an error: {v}"],
                            "final_cost": 0
                        },
                        "calculated_cost": 0
                    })

    # Check duration constraint
    if duration_days > 0 and len(plan) != duration_days:
        errors.append(f"Plan has {len(plan)} days, but constraints require {duration_days} days.")

    # --- 3. Robust Database Parsing ---
    db = {
        'flights': {},     # ID -> Price
        'hotels': {},      # Name -> {price, occupancy}
        'restaurants': {}, # Name -> Cost
        'attractions': [], # Name
        'transport': {}    # Key -> Cost
    }

    sections = re.split(r'(?=Description:)', context_str)
    for section in sections:
        if not section.strip(): continue
        lower_sec = section.lower()
        content_match = re.search(r'Content:(.*)', section, re.DOTALL)
        if not content_match: continue
        lines = [l.strip() for l in content_match.group(1).split('\n') if l.strip()]

        # Parse Flights
        if 'flight' in lower_sec:
            for line in lines:
                # Look for F followed by digits
                m_id = re.search(r'(F\d+)', line)
                # Look for price (digits potentially with decimal)
                m_price = re.search(r'\s+(\d+\.?\d*)\s+', line)
                if m_id and m_price:
                    db['flights'][m_id.group(1)] = float(m_price.group(1))

        # Parse Restaurants
        elif 'restaurant' in lower_sec:
            for line in lines:
                # Format: Name Cost Cuisines Rating City
                # We look for the first number which is usually cost or rating.
                # However, the cost is usually the first separate number after the name.
                # Strategy: Split by multiple spaces
                parts = re.split(r'\s{2,}', line)
                if len(parts) >= 2:
                    # Skip header
                    if 'average cost' in parts[1].lower(): continue
                    
                    name = clean_text(parts[0])
                    cost = 0
                    # Try to find cost in the second column
                    try:
                        cost = float(re.sub(r'[^\d.]', '', parts[1]))
                    except:
                        pass
                    
                    if cost > 0:
                        db['restaurants'][name] = cost

        # Parse Accommodations
        elif 'accommodation' in lower_sec:
            for line in lines:
                parts = re.split(r'\s{2,}', line)
                if len(parts) >= 2:
                    if 'price' in parts[1].lower(): continue
                    
                    name = clean_text(parts[0])
                    price = 0
                    try:
                        price = float(re.sub(r'[^\d.]', '', parts[1]))
                    except: pass
                    
                    # Look for occupancy in subsequent parts
                    occ = 2
                    # Heuristic: look for small integer (1-20) in later columns
                    for p in parts[2:]:
                        try:
                            val = float(re.sub(r'[^\d.]', '', p))
                            if 1 <= val <= 20 and val.is_integer():
                                occ = int(val)
                                break # Assume first small int is occupancy
                        except: pass
                        
                    if price > 0:
                        db['hotels'][name] = {'price': price, 'occupancy': occ}

        # Parse Attractions
        elif 'attraction' in lower_sec:
            for line in lines:
                parts = re.split(r'\s{2,}', line)
                if parts and 'name' not in parts[0].lower():
                    db['attractions'].append(clean_text(parts[0]))
        
        # Parse Transport (Taxi/Self-driving)
        elif 'taxi' in lower_sec or 'self-driving' in lower_sec:
            # Format: "mode, from Origin to Dest, ... cost: X"
            for line in lines:
                if 'cost' in line.lower():
                    try:
                        cost_m = re.search(r'cost:\s*(\d+)', line.lower())
                        if cost_m:
                            cost = float(cost_m.group(1))
                            # Extract Origin and Dest
                            from_to = re.search(r'from\s+(.*?)\s+to\s+(.*?)(,|$)', line, re.IGNORECASE)
                            if from_to:
                                origin = clean_text(from_to.group(1))
                                dest = clean_text(from_to.group(2))
                                mode = "self-driving" if "self-driving" in lower_sec else "taxi"
                                # Store with a standardized key
                                key = f"{mode}_{origin}_{dest}"
                                db['transport'][key] = cost
                    except: pass

    # --- 4. Validation Loop ---
    total_cost = 0.0
    cars_needed = math.ceil(people_count / 4)
    
    # State tracking
    current_location = clean_text(constraints.get('origin_city', ''))
    
    normalized_plan = []
    for day in plan:
        if isinstance(day, dict):
            normalized_plan.append({k.lower().strip(): v for k, v in day.items()})
        else:
            normalized_plan.append({})

    for i, day in enumerate(normalized_plan):
        d_idx = i + 1
        
        # 4a. Validate Current City & Transition
        curr_city_raw = day.get('current_city', '-')
        if not curr_city_raw or curr_city_raw == '-':
            errors.append(f"Day {d_idx}: 'current_city' is missing.")
            continue
        
        curr_city_clean = clean_text(curr_city_raw)
        
        # Detect Travel
        # Format expected: "from Origin to Dest"
        is_travel_day = 'from' in curr_city_clean and 'to' in curr_city_clean
        
        if is_travel_day:
            # Extract implied origin and dest from city string
            m = re.search(r'from\s+(.*?)\s+to\s+(.*)', curr_city_clean)
            if m:
                origin = m.group(1).strip()
                dest = m.group(2).strip()
                
                # Check continuity
                if current_location and origin not in current_location and current_location not in origin:
                    # Allow soft mismatch but flag if completely different
                    pass # Relaxed check as city names might vary slightly
                
                current_location = dest # Update location
                
                # Validate Transportation
                trans = day.get('transportation', '-')
                if not trans or trans == '-':
                    errors.append(f"Day {d_idx}: Travel detected but 'transportation' is missing.")
                else:
                    trans_clean = clean_text(trans)
                    # Check Flight
                    f_match = re.search(r'(F\d+)', trans)
                    if f_match:
                        fid = f_match.group(1)
                        if fid in db['flights']:
                            total_cost += db['flights'][fid] * people_count
                        else:
                            errors.append(f"Day {d_idx}: Flight {fid} not found in database.")
                    # Check Ground Transport
                    else:
                        # Try to match with DB transport keys
                        # Key format: "mode_origin_dest"
                        matched_transport = False
                        for key, cost in db['transport'].items():
                            mode_key, origin_key, dest_key = key.split('_')
                            # Check if mode, origin, and dest are in the transportation string or city string
                            # Heuristic: trans string usually contains "Self-driving" or "Taxi"
                            if mode_key in trans_clean and origin_key in curr_city_clean and dest_key in curr_city_clean:
                                total_cost += cost * cars_needed
                                matched_transport = True
                                break
                        if not matched_transport:
                             errors.append(f"Day {d_idx}: valid ground transport (Self-driving/Taxi) matching route not found in database for '{trans}'.")
        else:
            # Stationary day
            # current_location should match curr_city_clean approximately
            pass

        # 4b. Validate Accommodation
        acc = day.get('accommodation', '-')
        if acc and acc != '-':
            acc_clean = clean_text(acc.split(',')[0])
            found = False
            for h_name, h_data in db['hotels'].items():
                if h_name in acc_clean or acc_clean in h_name:
                    rooms = math.ceil(people_count / h_data['occupancy'])
                    total_cost += rooms * h_data['price']
                    found = True
                    break
            if not found:
                errors.append(f"Day {d_idx}: Accommodation '{acc}' not found in database.")
        elif i < len(plan) - 1:
            # If not the last day, accommodation is usually required unless explicit overnight travel
            # We won't strict fail but it is suspicious
            pass

        # 4c. Validate Meals
        for meal in ['breakfast', 'lunch', 'dinner']:
            item = day.get(meal, '-')
            if item and item != '-':
                item_clean = clean_text(item.split(',')[0])
                found = False
                for r_name, cost in db['restaurants'].items():
                    if r_name in item_clean or item_clean in r_name:
                        total_cost += cost * people_count
                        found = True
                        break
                if not found:
                    errors.append(f"Day {d_idx}: {meal.title()} '{item}' not found in database.")

        # 4d. Validate Attractions
        att = day.get('attraction', '-')
        if att and att != '-':
            for a in att.split(';'):
                if not a.strip(): continue
                a_clean = clean_text(a.split(',')[0])
                found = False
                for db_att in db['attractions']:
                    if db_att in a_clean or a_clean in db_att:
                        found = True
                        break
                if not found:
                    errors.append(f"Day {d_idx}: Attraction '{a}' not found in database.")

    # --- 5. Final Budget Check ---
    if total_cost > budget:
        errors.append(f"Total cost ${total_cost:.2f} exceeds budget ${budget:.2f}.")

    return self.make_final_answer("Validation logic applied", {
        "validation_report": {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "final_cost": total_cost
        },
        "calculated_cost": total_cost
    })

func_string = inspect.getsource(forward)

VALIDATION_TOOL = {
    "thought": "Validates the draft itinerary against budget and structural constraints using deterministic code.",
    "name": "Validation_Tool",
    "code": """{func_string}""".format(func_string=func_string)
}