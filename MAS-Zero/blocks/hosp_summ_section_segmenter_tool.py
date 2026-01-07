import inspect

# %%%%%%%%%%%%%%%%%%%% Section_Segmenter_Tool %%%%%%%%%%%%%%%%%%%%
def forward(self, taskInfo):
    import re
    import json

    # ---------- Step 1: Process the input data ----------
    # Extract content from taskInfo
    content = taskInfo.content
    task_data = {}
    
    if isinstance(content, dict):
        task_data = content
    elif isinstance(content, str):
        try:
            task_data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: if string is not JSON, treat it as the 'instruct' input directly
            task_data = {'instruct': content}
    
    raw_text = task_data.get('instruct', '')
    
    # ---------- Step 2: Implement the node logic ----------
    # Extended header map to capture all sections in the provided input and common variations
    # Keys are lowercased for normalization logic
    header_map = {
        'chief complaint': 'chief_complaint',
        'history of present illness': 'hpi',
        'past medical history': 'pmh',
        'social history': 'social_history',
        'family history': 'family_history',
        'major surgical or invasive procedure': 'procedures',
        'allergies': 'allergies',
        'physical exam': 'physical_exam',
        'pertinent results': 'results',
        'brief hospital course': 'hospital_course',
        'discharge diagnosis': 'discharge_diagnosis',
        'discharge instructions': 'instructions',
        'discharge medications': 'medications',
        'medications on admission': 'admission_meds',
        'service': 'service',
        'attending': 'attending',
        'discharge disposition': 'disposition',
        'discharge condition': 'condition',
        'review of systems': 'ros'
    }
    
    # Sort keys by length (descending) to match specific long headers before shorter substrings
    sorted_headers = sorted(header_map.keys(), key=len, reverse=True)
    
    # Create a regex pattern: \n followed by optional whitespace, then the header, then a colon
    # Using re.IGNORECASE to handle casing variations
    # re.escape ensures special characters in headers are treated as literals
    combined_headers = '|'.join(map(re.escape, sorted_headers))
    pattern = re.compile(r'\n\s*(' + combined_headers + r'):', re.IGNORECASE)
    
    # Initialize result dict
    sections = {}
    
    # Split the text. Prepend newline to ensure the first header is caught if at start.
    parts = pattern.split('\n' + raw_text)
    
    # The first part is preamble (before any known header)
    if parts:
        sections['preamble'] = parts[0].strip()
    
    # Iterate through split parts. re.split returns [text, captured_header, text, captured_header, ...]
    # We iterate from index 1, in steps of 2
    for i in range(1, len(parts) - 1, 2):
        raw_header = parts[i]
        content = parts[i+1].strip()
        
        # Normalize key using the map. Lowercase the captured header for lookup.
        # Normalize whitespace in header (e.g. 'Chief   Complaint' -> 'chief complaint')
        clean_header = re.sub(r'\s+', ' ', raw_header).strip().lower()
        normalized_key = header_map.get(clean_header, 'other')
        
        sections[normalized_key] = content
        
    # Fallback if no segmentation occurred (or only preamble found but it covers everything)
    if len(sections) <= 1:
        if 'preamble' in sections:
             # If preamble is empty and raw_text exists, treat as body
             if not sections['preamble'] and raw_text:
                 sections['body'] = raw_text
             # If only preamble and it has content, it remains as preamble (correct for files with no headers)
        elif not sections:
             sections['body'] = raw_text

    # ---------- Step 3: Collect the output ----------
    output_data = {
        'segmented_sections': sections
    }
    
    # Return the final answer wrapped in an Info object via make_final_answer
    # No 'thinking' component for this deterministic tool
    return self.make_final_answer('', output_data)

func_string = inspect.getsource(forward)

SECTION_SEGMENTER_TOOL = {
    "thought": "This tool parses unstructured clinical text into semantic sections using regex heuristics based on common clinical headers. It normalizes section names and structures the raw notes into a dictionary for downstream processing.",
    "name": "Section_Segmenter_Tool",
    "code": """{func_string}""".format(func_string=func_string)
}