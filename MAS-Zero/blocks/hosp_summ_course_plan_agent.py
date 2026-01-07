import inspect
import json
import copy

# %%%%%%%%%%%%%%%%%%%% Course_Plan_Agent %%%%%%%%%%%%%%%%%%%%
def forward(self, taskInfo):
    # Instruction for the Chain-of-Thought (CoT) approach
    # The system prompt from the original node serves as the instruction here.
    cot_instruction = (
        "You are an expert medical summarizer. Create a structured Discharge Summary based on the provided patient records.\n\n"
        "Follow this exact 3-paragraph structure:\n\n"
        "**Paragraph 1: Admission & History**\n"
        "- Start with patient age, gender, and primary diagnosis (include specific genetic details if available, e.g., del 5q).\n"
        "- Briefly summarize the HPI/Reason for admission.\n"
        "- State relevant Past Medical History (PMH) and Allergies clearly.\n\n"
        "**Paragraph 2: Hospital Course & Treatment**\n"
        "- Summarize the primary treatment (e.g., chemotherapy cycle, agents, dosages if relevant).\n"
        "- Mention supportive care (antiemetics, prophylaxis) and specific medications used.\n"
        "- Note clinical progress, tolerance of treatment, and key complications (or lack thereof).\n"
        "- Incorporate significant lab trends/results if relevant to the course (e.g., anemia management).\n\n"
        "**Paragraph 3: Discharge Plan**\n"
        "- Summarize discharge instructions, follow-up appointments (dates/doctors).\n"
        "- Mention home medications/treatments and specific warning signs to return for.\n\n"
        "**Guidelines**:\n"
        "- Be concise but comprehensive.\n"
        "- Do not list raw data; synthesize it into the narrative.\n"
        "- Ensure Allergies and PMH are explicitly mentioned in Para 1."
    )

    # Instantiate a new LLM agent specifically for CoT
    # We enable 'thinking' to allow the model to reason before generating the final summary.
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Course Plan Agent',  model=self.node_model, temperature=0.0)

    # Extract inputs from taskInfo
    content = taskInfo.content
    if isinstance(content, str):
        try:
            input_data = json.loads(content)
        except:
            input_data = {}
    elif isinstance(content, dict):
        input_data = content
    else:
        input_data = {}

    sections = input_data.get('segmented_sections', {})
    results_summary = input_data.get('results_summary', "See Pertinent Results section.")
    
    # Helper function to extract specific content from sections
    def get_content(keywords, exclusions=None, join_all=False):
        matches = []
        for k, v in sections.items():
            k_lower = k.lower()
            if exclusions and any(exc in k_lower for exc in exclusions):
                continue
            if any(kw in k_lower for kw in keywords):
                header = f"--- {k.replace('_', ' ').upper()} ---dict"
                matches.append(f"{header}\n{v}")
        
        if not matches:
            return ""
        
        if join_all:
            return "\n\n".join(matches)
        else:
            return matches[0]

    # 1. Specific Extractions for Para 1 (History)
    preamble = get_content(['preamble', 'patient_info', 'demographics'], join_all=True)
    allergies = get_content(['allergies', 'allergy'], join_all=True)
    if not allergies: allergies = "No Allergies Documented"
    
    hpi = get_content(['history_of_present_illness', 'hpi', 'chief_complaint'], exclusions=['past', 'social', 'family'], join_all=True)
    pmh = get_content(['past_medical_history', 'pmh'], exclusions=['present'], join_all=True)
    
    # 2. Extractions for Para 2 (Course)
    procedures = get_content(['major_surgical', 'procedure', 'surgery', 'invasive'])
    course_content = get_content(
        ['brief_hospital_course', 'hospital_course', 'icu_course', 'floor_course', 'course', 'assessment', 'plan'], 
        exclusions=['instruction', 'medication', 'discharge'], 
        join_all=True
    )
    meds_all = get_content(['medication', 'meds'], join_all=True)
    
    # 3. Extractions for Para 3 (Plan)
    discharge_info = get_content(['discharge_instructions', 'discharge_diagnosis', 'diagnosis', 'disposition', 'condition', 'followup', 'return'], join_all=True)

    # Construct the user content for the LLM
    user_prompt_template = (
        "**Patient Demographics**:\n{preamble}\n\n"
        "**Allergies**:\n{allergies}\n\n"
        "**HPI**:\n{hpi}\n\n"
        "**PMH**:\n{pmh}\n\n"
        "**Key Results/Labs**:\n{results_summary}\n\n"
        "**Procedures**:\n{procedures}\n\n"
        "**Hospital Course Notes**:\n{course_content}\n\n"
        "**Medications**:\n{meds_all}\n\n"
        "**Discharge Plan & Instructions**:\n{discharge_info}\n\n"
        "Task: Generate the 3-paragraph summary."
    )
    
    user_content = user_prompt_template.format(
        preamble=preamble,
        allergies=allergies,
        hpi=hpi,
        pmh=pmh,
        results_summary=results_summary,
        procedures=procedures,
        course_content=course_content,
        meds_all=meds_all,
        discharge_info=discharge_info
    )

    # Prepare the inputs for the CoT agent
    # We create a copy of taskInfo and inject the pre-formatted user_content strings
    agent_input = copy.copy(taskInfo)
    agent_input.content = user_content
    cot_agent_inputs = [agent_input]

    # Get the response from the CoT agent
    thinking, answer = cot_agent(cot_agent_inputs, cot_instruction)
    final_answer = self.make_final_answer(thinking, answer)
    
    # Return only the final answer
    return final_answer   

func_string = inspect.getsource(forward)

COURSE_PLAN_AGENT = {
    "thought": "This node processes segmented medical record sections and a results summary to generate a structured discharge summary. I extracted the input handling logic to parse the 'segmented_sections' dictionary. The complex string formatting logic (preamble, allergies, etc.) is preserved to build the specific context required for the prompt. The LLM interaction is handled via `LLMAgentBase` with a 'thinking' step to encourage reasoning before generation.",
    "name": "Course_Plan_Agent",
    "code": """{func_string}""".format(func_string=func_string)
}