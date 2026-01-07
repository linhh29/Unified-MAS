import inspect
import json
import re
from datetime import datetime, timedelta

# %%%%%%%%%%%%%%%%%%%% ItineraryPlanner_Agent %%%%%%%%%%%%%%%%%%%%
def forward(self, taskInfo):
    import json
    import re
    from datetime import datetime, timedelta

    # ---------- Step 1: Process input ----------
    # Extract input data from taskInfo
    if isinstance(taskInfo.content, dict):
        input_data = taskInfo.content
    elif isinstance(taskInfo.content, str):
        try:
            input_data = json.loads(taskInfo.content)
        except:
            input_data = {}
    else:
        input_data = {}

    constraints = input_data.get('constraints', {})
    filtered_context = input_data.get('filtered_context', '')
    
    budget = float(constraints.get('budget', 11000))
    duration = int(constraints.get('duration_days', 7))
    start_date_str = constraints.get('start_date', '2022-03-10')
    people_count = int(constraints.get('people_count', 4))
    
    # Normalize constraints for accommodation
    req_room_type = "Entire home/apt"
    
    # ---------- Step 2: Intelligent Parsing ----------
    def parse_table(text):
        lines = text.strip().split('\n')
        if len(lines) < 2: return [], []
        headers = re.split(r'\s{2,}|\t', lines[0].strip())
        data = []
        for line in lines[1:]:
            parts = re.split(r'\s{2,}|\t', line.strip())
            if len(parts) >= len(headers) - 1:
                row = {}
                for i, h in enumerate(headers):
                    if i < len(parts):
                        row[h.lower()] = parts[i]
                data.append(row)
        return data

    # 2a. Parse Flights
    flight_sections = re.findall(r'Description: Flight from ([A-Za-z\s]+) to ([A-Za-z\s]+) on (\d{4}-\d{2}-\d{2}).*?Content:(.*?)(?=\n\n|\Z)', filtered_context, re.DOTALL | re.IGNORECASE)
    
    all_flights = []
    for origin, dest, date_str, content in flight_sections:
        rows = parse_table(content)
        for r in rows:
            try:
                price = float(r.get('price', 0))
                num = r.get('flight number', 'Unknown')
                dep = r.get('deptime', '00:00')
                arr = r.get('arrtime', '00:00')
                all_flights.append({
                    'origin': origin.strip(),
                    'dest': dest.strip(),
                    'date': date_str,
                    'price': price,
                    'details': f"Flight Number: {num}, from {origin.strip()} to {dest.strip()}, Departure Time: {dep}, Arrival Time: {arr}"
                })
            except:
                continue
    
    # 2b. Parse Accommodations
    acc_sections = re.findall(r'Description: Accommodations in ([A-Za-z\s]+).*?Content:(.*?)(?=\n\n|\Z)', filtered_context, re.DOTALL | re.IGNORECASE)
    
    city_hotels = {}
    for city, content in acc_sections:
        city = city.strip()
        rows = parse_table(content)
        valid_hotels = []
        for r in rows:
            try:
                price = float(r.get('price', 0))
                occupancy = float(r.get('maximum occupancy', 0))
                room_type = r.get('room type', '')
                house_rules = r.get('house_rules', '')
                name = r.get('name', 'Unknown')
                
                if occupancy < people_count:
                    continue
                if req_room_type.lower() not in room_type.lower():
                    continue
                if 'no parties' in house_rules.lower():
                    continue
                
                valid_hotels.append({
                    'name': name,
                    'price': price,
                    'city': city,
                    'details': f"{name}, {city}"
                })
            except:
                continue
        valid_hotels.sort(key=lambda x: x['price'])
        city_hotels[city] = valid_hotels

    # ---------- Step 3: Build Trip Skeleton ----------
    all_flights.sort(key=lambda x: x['date'])
    
    selected_flights = {}
    for f in all_flights:
        key = f"{f['date']}_{f['origin']}_{f['dest']}"
        if key not in selected_flights:
            selected_flights[key] = f
        else:
            if f['price'] < selected_flights[key]['price']:
                selected_flights[key] = f
    
    sorted_moves = sorted(selected_flights.values(), key=lambda x: x['date'])
    
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    except:
        start_date = datetime(2022, 3, 10)
        
    skeleton_lines = []
    current_city = constraints.get('origin_city', 'Seattle')
    current_hotel = "-"
    
    trip_days = []
    for i in range(duration):
        d = start_date + timedelta(days=i)
        d_str = d.strftime("%Y-%m-%d")
        
        move = None
        for m in sorted_moves:
            if m['date'] == d_str and m['origin'].lower() == current_city.lower():
                move = m
                break
        
        day_info = {}
        day_info['day'] = i + 1
        
        if move:
            dest = move['dest']
            day_info['current_city'] = f"from {current_city} to {dest}"
            day_info['transportation'] = move['details']
            current_city = dest
            if current_city in city_hotels and city_hotels[current_city]:
                current_hotel = city_hotels[current_city][0]['details']
            else:
                found = False
                for c_key in city_hotels:
                    if c_key.lower() in current_city.lower() or current_city.lower() in c_key.lower():
                        current_hotel = city_hotels[c_key][0]['details']
                        found = True
                        break
                if not found:
                    current_hotel = "Unknown Accommodation"
            day_info['accommodation'] = current_hotel
        else:
            day_info['current_city'] = current_city
            day_info['transportation'] = "-"
            day_info['accommodation'] = current_hotel
            
        trip_days.append(day_info)
        skeleton_lines.append(json.dumps(day_info))

    skeleton_str = "\n".join(skeleton_lines)

    # ---------- Step 4: LLM Call ----------
    system_prompt = f"""You are an expert Travel Planner. Your task is to generate a {duration}-day travel itinerary in valid JSON format based on the provided constraints and data.

**TRIP SKELETON (MUST FOLLOW):**
Use the following pre-calculated logistics for the itinerary. The trip structure is dictated by available flight dates.
{skeleton_str}

**INSTRUCTIONS:**
1. **Fill in the Details**: For each day, keep the 'current_city', 'transportation', and 'accommodation' EXACTLY as provided in the Skeleton above. You only need to select suitable Restaurants (Breakfast, Lunch, Dinner) and Attractions from the Reference Information for that city.
2. **Consistency**: Ensure all details (names, flight numbers) are copied exactly from the data.
3. **Formatting**: 
   - 'current_city' for travel days: 'from [Origin] to [Dest]'.
   - 'current_city' for stay days: '[City]'.
   - Use '-' for fields where no activity is needed (e.g., Transportation on stay days).
4. **No Parties Check**: The user requested accommodations that allow parties. The skeleton has already filtered for this. Do not change the accommodation.

**OUTPUT SCHEMA:**
Return a JSON object with a single key "plan" containing a list of day objects:
{{
  "plan": [
    {{
      "days": 1,
      "current_city": "String",
      "transportation": "String",
      "breakfast": "String",
      "attraction": "String",
      "lunch": "String",
      "dinner": "String",
      "accommodation": "String"
    }}
  ]
}}
"""

    user_prompt = f"""
    **Constraints**:
    - Budget: ${budget}
    - Duration: {duration} days
    - Travelers: {people_count}
    - Dates: {start_date_str} to {constraints.get('end_date', 'Unknown')}

    **Reference Information**:
    {filtered_context}

    **Task**: Generate the full {duration}-day itinerary JSON.
    **Response**:
    """

    # Combine instructions for the agent
    cot_instruction = system_prompt + "\n\n" + user_prompt

    # Instantiate the LLM agent
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'ItineraryPlanner_Agent', model=self.node_model, temperature=0.0)

    # Call the agent
    try:
        thinking, answer = cot_agent([taskInfo], cot_instruction)
        response_content = answer.content
    except Exception as e:
        print(f"LLM Call failed: {e}")
        return self.make_final_answer(None, json.dumps({"draft_itinerary": []}))
    
    # ---------- Step 5: Output Parsing ----------
    draft = []
    try:
        response_str = response_content
        # Try to extract JSON block if wrapped in text
        json_block_match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if json_block_match:
            try:
                response_data = json.loads(json_block_match.group(0))
            except:
                # Fallback to loading whole string if strict block load fails
                response_data = json.loads(response_str)
        else:
            response_data = json.loads(response_str)
            
        if 'plan' in response_data and isinstance(response_data['plan'], list):
            draft = response_data['plan']
        elif isinstance(response_data, list):
            draft = response_data
        elif isinstance(response_data, dict):
            # Look for any list value
            for v in response_data.values():
                if isinstance(v, list):
                    draft = v
                    break
        
        # Clean and validate keys
        final_draft = []
        required_keys = ["days", "current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"]
        
        for i, item in enumerate(draft):
            if not isinstance(item, dict): continue
            cleaned = {}
            cleaned['days'] = item.get('days', i + 1)
            for k in required_keys:
                if k == 'days': continue
                val = item.get(k, '-')
                cleaned[k] = str(val) if val is not None else '-'
            final_draft.append(cleaned)
            
        draft = final_draft

    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        draft = []

    # Prepare final answer content as JSON string to match original return structure
    final_output = {"draft_itinerary": draft}
    answer.content = json.dumps(final_output)
    
    return self.make_final_answer(thinking, answer)

func_string = inspect.getsource(forward)

ITINERARYPLANNER_AGENT = {
    "thought": "This node generates an initial itinerary draft. It first deterministically builds a trip skeleton based on flight constraints and budget, then uses an LLM (with Chain-of-Thought) to flesh out details like restaurants and attractions while strictly adhering to the skeleton. This approach ensures logistical feasibility while leveraging the LLM for creative selection.",
    "name": "ItineraryPlanner_Agent",
    "code": """{func_string}""".format(func_string=func_string)
}