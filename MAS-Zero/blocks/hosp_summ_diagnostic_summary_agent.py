import inspect
import re
import json

# %%%%%%%%%%%%%%%%%%%% Diagnostic_Summary_Agent %%%%%%%%%%%%%%%%%%%% 
def forward(self, taskInfo):
    # ---------- Step 1: Process the input data ----------
    # Extract content from taskInfo
    # taskInfo.content might be a dict or a JSON string depending on the upstream node
    input_data = taskInfo.content if isinstance(taskInfo.content, dict) else {}
    if not input_data and isinstance(taskInfo.content, str):
        try:
            input_data = json.loads(taskInfo.content)
        except:
            input_data = {}

    sections = input_data.get('segmented_sections', {})
    preamble = sections.get('preamble', '')
    
    # Robust Regex for Age (looking for "age: 54" or "Age 54" pattern)
    age_match = re.search(r"age\s*[:\.]?\s*(\d+)", preamble, re.IGNORECASE)
    patient_age = age_match.group(1) if age_match else "unknown"
    
    # Robust Regex for Gender
    gender_match = re.search(r"(?:gender|sex)\s*[:\.]?\s*([MF])", preamble, re.IGNORECASE)
    raw_gender = gender_match.group(1).upper() if gender_match else ""
    if raw_gender == 'F':
        patient_gender = "female"
    elif raw_gender == 'M':
        patient_gender = "male"
    else:
        patient_gender = "patient"

    # Helper to safely get section content with multiple possible keys
    def get_section(keys, default='N/A'):
        for k in keys:
            if k in sections and sections[k]:
                return sections[k]
        return default

    # Extract specific sections with fallback keys
    hpi = get_section(['hpi', 'history_of_present_illness'], 'N/A')
    pmh = get_section(['past_medical_history', 'pmh'], 'N/A')
    meds = get_section(['discharge_medications', 'medications_on_discharge'], 'N/A')
    dx = get_section(['discharge_diagnosis', 'discharge_diagnoses'], 'N/A')
    
    # For hospital course, use the specific section if available, else upstream summary
    raw_course = get_section(['brief_hospital_course', 'hospital_course'])
    course_text = raw_course if raw_course != 'N/A' else input_data.get('course_summary', '')

    # ---------- Step 2: Pre-process Text to Remove Conflicting Redactions ----------
    # If we found an age, replace "___ year old" in the text with the actual age
    # to prevent the LLM from hallucinating or keeping the redaction.
    if patient_age != "unknown":
        redaction_pattern = r"[_]{2,}\s*-?\s*year\s*-?\s*old"
        replacement_text = f"{patient_age}-year-old"
        hpi = re.sub(redaction_pattern, replacement_text, hpi, flags=re.IGNORECASE)
        course_text = re.sub(redaction_pattern, replacement_text, course_text, flags=re.IGNORECASE)

    # ---------- Step 3: Implement the node logic for LLM_Generator ----------
    system_prompt = (
        "You are an expert Medical Summarizer. Synthesize the patient records into a comprehensive 4-paragraph narrative summary.\n\n"
        "CRITICAL DATA OVERRIDE:\n"
        "- Patient Age: {patient_age}\n"
        "- Gender: {patient_gender}\n\n"
        "INSTRUCTIONS:\n"
        "1. **First Sentence**: You MUST start with \"The patient is a {patient_age}-year-old {patient_gender} who was admitted...\"\n"
        "   - DO NOT output \"___-year-old\". Use \"{patient_age}-year-old\" even if the text below has redactions.\n"
        "2. **Structure**: Create exactly 4 paragraphs:\n"
        "   - **Paragraph 1 (Admission & HPI)**: Introduction, reason for admission (hypoxia/procedure), and history of present illness. Include the presenting symptoms and initial discovery details.\n"
        "   - **Paragraph 2 (History)**: Summarize Past Medical History (PMH). List chronic conditions and allergies.\n"
        "   - **Paragraph 3 (Hospital Course)**: Narrative of the stay. Include the procedure, post-op course, complications (or lack thereof), and clinical improvement.\n"
        "   - **Paragraph 4 (Discharge)**: Discharge diagnoses, list of specific discharge medications (names only), and follow-up plan.\n"
        "3. **Redaction Handling**: The input text contains '___' for redacted dates/names. Replace these with generic terms like 'in the past', 'previously', or 'the hospital'. Exception: Use the provided Age and Gender.\n"
        "4. **Style**: Professional, smooth narrative flow.\n"
    ).format(patient_age=patient_age, patient_gender=patient_gender)
    
    user_prompt = (
        "Patient Records:\n"
        "[HPI]:\n{hpi}\n\n"
        "[PMH]:\n{pmh}\n\n"
        "[Hospital Course]:\n{course_text}\n\n"
        "[Discharge Meds]:\n{meds}\n\n"
        "[Discharge Diagnosis]:\n{dx}\n\n"
        "Task: Generate the narrative summary now."
    ).format(
        hpi=hpi,
        pmh=pmh,
        course_text=course_text,
        meds=meds,
        dx=dx
    )
    
    # Combine the system and user prompt into the instruction for the agent
    instruction = f"{system_prompt}\n\n{user_prompt}"

    # Instantiate the LLM agent
    agent = LLMAgentBase(['thinking', 'answer'], 'Diagnostic_Summary_Agent', model=self.node_model, temperature=0.0)

    # Get the response from the agent
    # We pass [taskInfo] to provide context, though our instruction contains the specific formatted data.
    thinking, answer = agent([taskInfo], instruction)
    
    # ---------- Step 4: Return final result ----------
    final_answer = self.make_final_answer(thinking, answer)
    return final_answer

func_string = inspect.getsource(forward)

DIAGNOSTIC_SUMMARY_AGENT = {
    "thought": "This node synthesizes a diagnostic summary from segmented patient records. I migrated it to extract data from taskInfo, applied the original regex-based redaction handling, and used an LLMAgentBase with a combined instruction prompt to generate the narrative.",
    "name": "Diagnostic_Summary_Agent",
    "code": """{func_string}""".format(func_string=func_string)
}