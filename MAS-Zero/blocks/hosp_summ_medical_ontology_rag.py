import inspect
import json

# %%%%%%%%%%%%%%%%%%%% MEDICAL_ONTOLOGY_RAG %%%%%%%%%%%%%%%%%%%% 
def forward(self, taskInfo):
    # Step 1: Prepare Context from taskInfo
    # Handle potential JSON string or dict in taskInfo.content
    input_content = taskInfo.content
    if isinstance(input_content, str):
        try:
            input_data = json.loads(input_content)
        except:
            input_data = {}
    elif isinstance(input_content, dict):
        input_data = input_content
    else:
        input_data = {}

    sections = input_data.get('segmented_sections', {})
    
    # Combine relevant sections where acronyms/jargon appear
    context_text = (
        f"Procedures: {sections.get('major_surgical_or_invasive_procedure', '')}\n"
        f"HPI: {sections.get('history_of_present_illness', '')}\n"
        f"Labs/Exam: {sections.get('physical_exam', '')} {sections.get('pertinent_results', '')}"
    )
    # Truncate to avoid context limit
    context_text = context_text[:4000]

    # Step 2: Targeted Extraction Agent
    # Instruction to get only abbreviations and specific eponymous terms
    extraction_instruction = (
        "Identify and list medical acronyms, abbreviations, and eponymous surgical procedures found in the text. "
        "Strictly follow these rules:\n"
        "1. INCLUDE acronyms (e.g., OSH, BRBPR, EGD, Hct, MICU, NGT, PPI, CABG).\n"
        "2. INCLUDE eponymous procedures (e.g., Billroth II, Whipple).\n"
        "3. EXCLUDE common full disease names (e.g., Hypertension, Diabetes, Pneumonia).\n"
        "4. EXCLUDE common anatomical terms (e.g., abdomen, heart).\n"
        "5. EXCLUDE standard units of measure unless ambiguous.\n"
        "6. Output ONLY a comma-separated list of the top 8 most clinically relevant terms needing definition. If none, return 'None'."
    )

    # Use taskInfo to pass context, modifying content temporarily for the agent
    taskInfo.content = context_text
    
    extraction_agent = LLMAgentBase(['thinking', 'answer'], 'Term Extraction Agent', model=self.node_model, temperature=0.0)
    thinking_1, terms_answer = extraction_agent([taskInfo], extraction_instruction)
    
    # Process extraction result
    terms_response = terms_answer.content
    search_terms = []
    if "None" not in terms_response and terms_response.strip():
        cleaned_response = terms_response.replace("Terms:", "").replace("Answer:", "").strip()
        search_terms = [t.strip() for t in cleaned_response.split(',') if t.strip()]

    # Step 3: Retrieval/Definition Agent
    if not search_terms:
        # No terms to define, return empty summary
        terms_answer.content = "No specific medical terminology definitions required."
        return self.make_final_answer(thinking_1, terms_answer)
    
    # Prepare query for Definition Agent
    # Note: Using LLM as knowledge base to replace external search engine for portability
    query_terms = ", ".join(search_terms[:8])
    definition_instruction = (
        "You are a precise Medical Dictionary. "
        "Your task is to define the provided medical terms/acronyms.\n"
        "Format Requirement:\n"
        "- Term: Definition (1 sentence)\n"
        "- Term: Definition (1 sentence)\n\n"
        "Do NOT write paragraphs. Do NOT provide general medical advice. ONLY define the specific terms requested."
    )
    
    # Update taskInfo content for the definition request
    taskInfo.content = f"Requested Terms: {query_terms}\nGenerate the definitions list."
    
    definition_agent = LLMAgentBase(['thinking', 'answer'], 'Definition Agent', model=self.node_model, temperature=0.0)
    thinking_2, definition_answer = definition_agent([taskInfo], definition_instruction)
    
    # Return the final definitions
    return self.make_final_answer(thinking_2, definition_answer)

func_string = inspect.getsource(forward)

MEDICAL_ONTOLOGY_RAG = {
    "thought": "I separate the task into two steps: extracting medical terms from the clinical text and then generating definitions for those terms using the model's knowledge base. This ensures only relevant acronyms are defined, keeping the summary semantic and concise.",
    "name": "Medical_Ontology_RAG",
    "code": """{func_string}""".format(func_string=func_string)
}