import inspect
import json

# %%%%%%%%%%%%%%%%%%%% Results_Processing_Agent %%%%%%%%%%%%%%%%%%%% 
def forward(self, taskInfo):
    # Extract input data from taskInfo
    content = taskInfo.content
    if isinstance(content, dict):
        sections = content.get('segmented_sections', {})
    elif isinstance(content, str):
        try:
            data = json.loads(content)
            sections = data.get('segmented_sections', {})
        except:
            sections = {}
    else:
        sections = {}
    
    # Extract relevant sections. Note: Imaging/Echo often sits inside 'Pertinent Results'
    proc_section = sections.get('major_surgical_or_invasive_procedure', 'Not documented')
    diagnosis_section = sections.get('discharge_diagnosis', 'Not documented')
    results_section = sections.get('pertinent_results') or sections.get('results', 'Not documented')
    imaging_section = sections.get('imaging', '') # Fallback if segmented separately
    hospital_course = sections.get('brief_hospital_course', 'Not documented')
    
    # Context construction
    # Using single triple quotes to avoid conflict with outer wrapper string
    combined_context = f'''
    [Discharge Diagnosis]: {diagnosis_section}
    [Major Procedure]: {proc_section}
    
    [Pertinent Results (Labs, Imaging, Studies)]:
    {results_section}
    {imaging_section}
    
    [Hospital Course (Context for procedures/findings)]:
    {hospital_course}
    '''

    # Instruction for the Agent (System Prompt)
    instruction = (
        "You are an expert Medical Scribe. Your task is to write a narrative summary of the diagnostic results and procedures for a hospital discharge summary.\n\n"
        "### GUIDELINES:\n"
        "1. **Format**: Output a SINGLE, continuous paragraph. **Do not use bullet points, lists, or line breaks.**\n"
        "2. **Content Focus**:\n"
        "   - **Labs**: Summarize significant abnormalities using qualitative terms (e.g., 'elevated creatinine', 'anemia', 'leukocytosis'). **Do not list raw numbers** unless they are critical for context.\n"
        "   - **Imaging/Diagnostics**: Summarize the key impressions of scans (CT, MRI) and tests (Echo, EKG). Focus on the final impression (e.g., 'CT head revealed chronic ischemic changes').\n"
        "   - **Procedures**: Mention major invasive procedures (e.g., pacemaker placement) and their outcome.\n"
        "3. **Style**: Use smooth, professional medical prose (e.g., 'Laboratory results showed...', 'Imaging studies revealed...', 'The patient underwent...').\n"
        "4. **Input Handling**: Imaging and procedure details may be embedded within the 'Pertinent Results' section; extract and synthesize them."
    )

    # Prepare the inputs for the agent
    # We construct the specific user prompt content expected by the logic
    user_prompt_content = f"Analyze the following clinical data and generate the narrative summary:\n\n{combined_context}"
    
    # Update taskInfo content to pass the formatted text to the agent
    taskInfo.content = user_prompt_content

    # Instantiate a new LLM agent
    # Using 'thinking' and 'answer' to match the reference format and enable CoT capabilities
    agent = LLMAgentBase(['thinking', 'answer'], 'Results_Processing_Agent', model=self.node_model, temperature=0.0)

    # Get the response from the agent
    thinking, answer = agent([taskInfo], instruction)
    
    # Return only the final answer using the helper
    return self.make_final_answer(thinking, answer)

func_string = inspect.getsource(forward)

RESULTS_PROCESSING_AGENT = {
    "thought": "Extracts and summarizes pertinent lab results, imaging, and procedures from segmented clinical sections, ensuring numerical accuracy and professional medical prose.",
    "name": "Results_Processing_Agent",
    "code": """{func_string}""".format(func_string=func_string)
}