import inspect
import json

# %%%%%%%%%%%%%%%%%%%% Final_Refinement_Agent %%%%%%%%%%%%%%%%%%%% 
def forward(self, taskInfo):
    # ---------- Step 1: Process the input data ----------
    input_data = taskInfo.content
    if isinstance(input_data, str):
        try:
            input_data = json.loads(input_data)
        except:
            input_data = {}
    elif not isinstance(input_data, dict):
        input_data = {}

    # Retrieve summaries
    history_sum = input_data.get('diagnostic_summary', 'No history provided.')
    results_sum = input_data.get('results_summary', 'No results provided.')
    course_sum = input_data.get('course_summary', 'No course provided.')

    # Retrieve raw segments from Section_Segmenter_Tool
    segments = input_data.get('segmented_sections')
    if not isinstance(segments, dict):
        segments = {}
        
    # Fallback keys logic
    preamble = segments.get('preamble', '')
    meds_raw = segments.get('discharge_medications', '') 
    pe_raw = segments.get('physical_exam', '') 
    
    # ---------- Step 2: Implement the node logic for LLM_Generator ----------
    system_prompt = (
        "You are an expert Clinical Documentation Specialist. Your task is to generate a structured Discharge Summary based on the provided patient records and partial summaries.\n\n"
        "CRITICAL OUTPUT RULES:\n"
        "1. **Format**: The output must be organized into exactly these four distinct sections/paragraphs:\n"
        "   - **Patient Summary**: A concise narrative starting with 'The patient is a [Age]-year-old [Gender]...' covering the admission reason, HPI, key procedures, and hospital course/outcome.\n"
        "   - **Past Medical History**: A list or comma-separated summary of conditions.\n"
        "   - **Discharge Medications**: A list of medications prescribed at discharge (generic names preferred).\n"
        "   - **Physical Examination**: Key findings from the admission exam (Vital signs, HEENT, CV, etc.).\n\n"
        "2. **Data Usage**: \n"
        "   - Use the **Demographics** section for Age/Gender.\n"
        "   - Use the **Medications** section for the discharge list.\n"
        "   - Use the **Physical Exam** section for specific findings (e.g., 'warm, dry skin', 'murmur').\n"
        "   - Use the **Summaries** for the narrative history and course.\n\n"
        "3. **Constraint**: Start your response immediately with 'Answer:'. Do not use markdown headers like '##'. Just separate paragraphs clearly."
    )
    
    user_prompt_template = (
        "Sources:\n"
        "[Demographics]: {preamble}\n"
        "[History Summary]: {history_sum}\n"
        "[Hospital Course Summary]: {course_sum}\n"
        "[Results Summary]: {results_sum}\n"
        "[Discharge Medications (Raw)]: {meds_raw}\n"
        "[Physical Exam (Raw)]: {pe_raw}\n\n"
        "Task: Generate the final Discharge Summary."
    )
    
    user_content = user_prompt_template.format(
        preamble=preamble,
        history_sum=history_sum,
        course_sum=course_sum,
        results_sum=results_sum,
        meds_raw=meds_raw,
        pe_raw=pe_raw
    )
    
    # Helper to wrap string content for Agent input
    class InputInfo:
        def __init__(self, content):
            self.content = content

    # Instantiate Agent
    agent = LLMAgentBase(['answer'], 'Final_Refinement_Agent', model=self.node_model, temperature=0.0)
    
    # Prepare inputs and call agent
    # We wrap the user_content string into an object as LLMAgentBase expects inputs with .content attribute
    agent_inputs = [InputInfo(user_content)]
    
    # Get the response
    # agent(...) returns a tuple of Info objects corresponding to the output fields
    answer_info = agent(agent_inputs, system_prompt)[0]
    response = answer_info.content

    # ---------- Step 3: Validate and Format Output ----------
    cleaned_response = response.strip()
    if not cleaned_response.startswith("Answer:"):
        if "Answer:" in cleaned_response:
            cleaned_response = cleaned_response[cleaned_response.find("Answer:"):]
        else:
            cleaned_response = "Answer: " + cleaned_response

    return cleaned_response

func_string = inspect.getsource(forward)

FINAL_REFINEMENT_AGENT = {
    "thought": "Aggregates partial summaries and raw segments into a final structured discharge summary.",
    "name": "Final_Refinement_Agent",
    "code": """{func_string}""".format(func_string=func_string)
}