import inspect
import re
import math
import json
from datetime import datetime

# %%%%%%%%%%%%%%%%%%%% DataFilter_Tool %%%%%%%%%%%%%%%%%%%%
def forward(self, taskInfo):
    # Extract input data from taskInfo
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

    reference_info = input_data.get('reference_information', [])
    constraints = input_data.get('constraints', {})
    query = input_data.get('query', '') or input_data.get('question', '')
    query_lower = query.lower()

    # --- 1. Constraint Extraction ---
    def clean_cost(s):
        if not s or str(s).strip() in ['-', 'Unknown']: return 0.0
        try:
            val = re.sub(r'[^\d.]', '', str(s))
            return float(val) if val and val != '.' else 0.0
        except: return 0.0

    budget = clean_cost(constraints.get('budget', 0))
    if budget == 0:
        m_budget = re.search(r'\$(\d+(?:,\d+)?)', query)
        if m_budget: budget = float(m_budget.group(1).replace(',', ''))
        else: budget = float('inf')

    people = 1
    if 'people_count' in constraints and constraints['people_count']:
        try: people = int(constraints['people_count'])
        except: pass
    else:
        m_people = re.search(r'(\d+)\s+people', query_lower)
        if m_people: people = int(m_people.group(1))

    start_date_str = constraints.get('start_date')
    end_date_str = constraints.get('end_date')
    total_nights = 1
    if start_date_str and end_date_str:
        try:
            d1 = datetime.strptime(start_date_str, "%Y-%m-%d")
            d2 = datetime.strptime(end_date_str, "%Y-%m-%d")
            total_nights = (d2 - d1).days
        except: pass
    if total_nights < 1: total_nights = 1

    avoid_driving = any(x in query_lower for x in ['no driving', 'not drive', 'avoid driving'])
    avoid_flying = any(x in query_lower for x in ['no flight', 'not take flight', 'avoid flying', 'without flying'])
    require_entire = 'entire' in query_lower
    has_children = any(x in query_lower for x in ['children', 'kid', 'under 10'])
    has_pets = 'pet' in query_lower
    
    target_cuisines = [c.strip() for c in query_lower.replace(',', ' ').split() 
                       if c in ['american', 'chinese', 'french', 'italian', 'indian', 'mexican', 'mediterranean', 'seafood', 'bbq', 'pizza', 'bakery', 'cafe']]

    # --- 2. Smart City Discovery ---
    known_cities = set()
    if constraints.get('origin_city'): known_cities.add(constraints['origin_city'].lower())
    if constraints.get('destination_city'): known_cities.add(constraints['destination_city'].lower())

    # Scan for implicit cities in the data headers and descriptions
    for item in reference_info:
        desc = item.get('Description', '')
        # "Attractions in Toledo"
        m_in = re.search(r'(?:Attractions|Accommodations|Restaurants)\s+in\s+([A-Za-z\.\s]+)$', desc, re.IGNORECASE)
        if m_in:
            known_cities.add(m_in.group(1).strip().lower())
        # "from Toledo to Cleveland"
        m_route = re.search(r'from\s+([A-Za-z\.\s]+?)\s+to\s+([A-Za-z\.\s]+)', desc, re.IGNORECASE)
        if m_route:
            known_cities.add(m_route.group(1).strip().lower())
            known_cities.add(m_route.group(2).strip().lower())

    # --- 3. Robust Table Parser ---
    def parse_table(content, category):
        if not content or "no flight" in content.lower(): return []
        lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
        if len(lines) < 2: return []
        
        # Header processing
        header_line = lines[0]
        headers = [h.lower() for h in re.split(r'\s{2,}', header_line)]
        
        results = []
        for line in lines[1:]:
            row = {}
            parts = re.split(r'\s{2,}', line)
            
            # Handle Pandas Index: if row has 1 more column than header and first is int
            if len(parts) == len(headers) + 1 and parts[0].replace('.', '').isdigit():
                parts.pop(0)
            
            # Anchor-based recovery if split failed (len(parts) < len(headers))
            if len(parts) < len(headers):
                anchor_idx = -1
                regex_anchor = None
                
                if category == 'attraction' and 'latitude' in headers:
                    # Find first float pattern
                    regex_anchor = r'\s+(-?\d+\.\d+)\s+'
                elif category == 'restaurant' and 'average cost' in headers:
                    # Find first int pattern (cost)
                    regex_anchor = r'\s+(\d+)\s+'
                elif category == 'accommodation' and 'price' in headers:
                    # Find price float pattern
                    regex_anchor = r'\s+(\d+(?:\.\d+)?)\s+'
                
                if regex_anchor:
                    match = re.search(regex_anchor, line)
                    if match:
                        name_part = line[:match.start()].strip()
                        rest_part = line[match.start():].strip()
                        # Re-split the rest
                        rest_parts = re.split(r'\s{2,}', rest_part)
                        parts = [name_part] + rest_parts
            
            # Map to headers
            for i, h in enumerate(headers):
                if i < len(parts):
                    row[h] = parts[i]
            results.append(row)
        return results

    valid_transport = []
    valid_accoms = []
    valid_restaurants = []
    valid_attractions = []

    for item in reference_info:
        desc = item.get('Description', '').strip()
        content = item.get('Content', '').strip()
        desc_lower = desc.lower()

        # Transport
        if 'from ' in desc_lower and 'to ' in desc_lower:
            m_route = re.search(r'from\s+([A-Za-z\.\s]+?)\s+to\s+([A-Za-z\.\s]+)', desc, re.IGNORECASE)
            if not m_route: continue
            r_origin, r_dest = m_route.group(1).strip(), m_route.group(2).strip()
            
            # Only filter if we have absolutely no match in known cities (loose filter)
            if r_origin.lower() not in known_cities and r_dest.lower() not in known_cities:
                continue

            t_type = 'Other'
            if 'flight' in desc_lower: t_type = 'Flight'
            elif 'taxi' in desc_lower: t_type = 'Taxi'
            elif 'driving' in desc_lower: t_type = 'Self-driving'

            if avoid_flying and t_type == 'Flight': continue
            if avoid_driving and t_type == 'Self-driving': continue

            if t_type == 'Flight':
                rows = parse_table(content, 'flight')
                for row in rows:
                    price = clean_cost(row.get('price', 'inf'))
                    if price == 0 or price == float('inf'): continue
                    total_cost = price * people
                    if total_cost <= budget:
                        valid_transport.append({
                            'origin': r_origin, 'dest': r_dest, 'cost': total_cost, 'type': 'Flight',
                            'desc': f"Flight {row.get('flight number','')}, {r_origin}->{r_dest}, Dep: {row.get('departure time','')}, Arr: {row.get('arrival time','')}, Total: ${total_cost:.0f}"
                        })
            else:
                # Taxi/Driving
                m_cost = re.search(r'cost:\s*([\d\.]+)', content)
                base_cost = float(m_cost.group(1)) if m_cost else 0.0
                real_cost = base_cost
                
                # Taxi usually per car, Driving usually per car (gas)
                # Assuming 4 pax per taxi, 5 per car
                if t_type == 'Taxi': real_cost = base_cost * math.ceil(people / 4.0)
                if t_type == 'Self-driving': real_cost = base_cost * math.ceil(people / 5.0)
                
                if real_cost <= budget:
                    dur = re.search(r'duration:\s*([^,]+)', content)
                    dist = re.search(r'distance:\s*([^,]+)', content)
                    d_str = dur.group(1) if dur else "?"
                    k_str = dist.group(1) if dist else "?"
                    valid_transport.append({
                        'origin': r_origin, 'dest': r_dest, 'cost': real_cost, 'type': t_type,
                        'desc': f"{t_type}, {r_origin} to {r_dest}, duration: {d_str}, distance: {k_str}, cost: {int(real_cost)}"
                    })
            continue

        # Entities
        m_city = re.search(r'in\s+([A-Za-z\.\s]+)$', desc, re.IGNORECASE)
        if not m_city: continue
        city_name = m_city.group(1).strip()
        # Loose check: if it's in known_cities or destination is a state matching this city
        if city_name.lower() not in known_cities:
             pass

        if 'attraction' in desc_lower:
            rows = parse_table(content, 'attraction')
            for row in rows:
                if row.get('name'):
                    valid_attractions.append({'city': city_name, 'name': row['name']})

        elif 'restaurant' in desc_lower:
            rows = parse_table(content, 'restaurant')
            for row in rows:
                cost = clean_cost(row.get('average cost', 0))
                cuisines = row.get('cuisines', '')
                is_match = any(tc in cuisines.lower() for tc in target_cuisines) if target_cuisines else True
                valid_restaurants.append({
                    'city': city_name, 'name': row.get('name'), 'cuisines': cuisines,
                    'cost': cost, 'rating': row.get('aggregate rating', '0'), 'is_match': is_match
                })

        elif 'accommodation' in desc_lower:
            rows = parse_table(content, 'accommodation')
            for row in rows:
                price = clean_cost(row.get('price', 0))
                rtype = row.get('room type', '')
                rules = row.get('house_rules', '').lower()
                min_n = clean_cost(row.get('minimum nights', 1))
                try: max_occ = float(row.get('maximum occupancy', 1))
                except: max_occ = 1.0

                if require_entire and 'entire' not in rtype.lower(): continue
                if has_pets and 'no pets' in rules: continue
                if total_nights < min_n: continue
                
                # Occupancy check
                rooms_needed = 1
                if 'entire' not in rtype.lower(): rooms_needed = math.ceil(people / max_occ)
                elif max_occ < people: continue

                valid_accoms.append({
                    'city': city_name, 'name': row.get('name'), 'type': rtype,
                    'price': price, 'rules': rules
                })

    # --- 4. Output Generation ---
    output = []
    output.append(f"Analysis: People={people}, Budget=${budget}, Duration={total_nights} days.")
    output.append(f"Constraints: NoFlight={avoid_flying}, NoDrive={avoid_driving}, Pets={has_pets}")
    output.append("")

    output.append("### TRANSPORTATION")
    if not valid_transport:
        output.append("No valid transportation options found.")
    else:
        # Group by route
        routes = {}
        for t in valid_transport:
            key = f"{t['origin']} -> {t['dest']}"
            if key not in routes: routes[key] = []
            routes[key].append(t)
        for route, opts in sorted(routes.items()):
            output.append(f"Route: {route}")
            for opt in opts:
                output.append(f"- {opt['desc']}")
    output.append("")

    # Group entities by City
    all_cities = sorted(list(set([x['city'] for x in valid_accoms + valid_restaurants + valid_attractions])))
    
    for city in all_cities:
        output.append(f"### CITY: {city}")
        
        # Accommodations
        city_acc = [a for a in valid_accoms if a['city'] == city]
        city_acc.sort(key=lambda x: x['price'])
        output.append(f"**Accommodations** ({len(city_acc)} found):")
        for a in city_acc[:10]: # Limit to top 10 cheap options
            output.append(f"- {a['name']} | {a['type']} | Price: ${a['price']:.0f} | Rules: {a['rules']}")
        if not city_acc: output.append("- None matching constraints.")

        # Restaurants
        city_rst = [r for r in valid_restaurants if r['city'] == city]
        # Sort by match first, then rating, then cost
        city_rst.sort(key=lambda x: (not x['is_match'], -float(x['rating']) if x['rating'].replace('.','').isdigit() else 0))
        output.append(f"**Restaurants** ({len(city_rst)} found):")
        for r in city_rst[:15]:
            match_mark = "[MATCH] " if r['is_match'] else ""
            output.append(f"- {match_mark}{r['name']} | {r['cuisines']} | Rating: {r['rating']} | AvgCost: ${r['cost']:.0f}")
        if not city_rst: output.append("- None found.")

        # Attractions
        city_att = [a['name'] for a in valid_attractions if a['city'] == city]
        output.append(f"**Attractions** ({len(city_att)} found):")
        if city_att:
            output.append(", ".join(city_att[:20]))
        else:
            output.append("- None found.")
        output.append("")

    final_answer_text = "\n".join(output)
    return self.make_final_answer("Data filtering and parsing completed based on extracted constraints.", final_answer_text)

func_string = inspect.getsource(forward)

DATA_FILTER_TOOL = {
    "thought": "Filters the raw semi-structured reference information based on the extracted constraints to remove irrelevant cities, wrong dates, or impossible costs.",
    "name": "DataFilter_Tool",
    "code": """{func_string}""".format(func_string=func_string)
}