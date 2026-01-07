import inspect
import json

# %%%%%%%%%%%%%%%%%%%% Legal_Knowledge_Retriever %%%%%%%%%%%%%%%%%%%% 
def forward(self, taskInfo):
    # Helper class to mimic Info object structure expected by LLMAgentBase
    class Info:
        def __init__(self, content):
            self.content = content

    # Step 1: Extract Facts
    # Parsing the content from taskInfo based on its type
    if isinstance(taskInfo.content, dict):
        task_data = taskInfo.content
    elif isinstance(taskInfo.content, str):
        try:
            task_data = json.loads(taskInfo.content)
        except:
            # If not valid JSON, treat the whole string as facts
            task_data = {"structured_facts": taskInfo.content}
    else:
        task_data = {}

    facts = task_data.get('structured_facts', '')
    if not facts:
        facts = str(task_data)

    # Step 2: Dynamic Query Generation
    query_system_prompt = (
        "You are a Senior Legal Research Assistant. Your task is to analyze the case facts to identify the core legal disputes "
        "and generate targeted search queries. \n"
        "Key Objectives:\n"
        "1. Identify the specific type of dispute (e.g., Return of Caili, Divorce Property Split).\n"
        "2. Identify specific controversies (e.g., 'Is transfer X considered Caili or a gift?', 'Calculation of amount', 'Exclusions from Caili').\n"
        "3. Generate 3-5 specific search keywords including specific Judicial Interpretations if applicable.\n"
        "4. Output ONLY the keywords separated by spaces."
    )
    
    query_user_prompt = f"Case Facts: {facts}\n\nKeywords:"
    
    # Create an agent for query generation
    query_agent = LLMAgentBase(['answer'], 'Query Generation Agent', model=self.node_model, temperature=0.0)
    
    # Prepare input for the agent
    query_input = Info(query_user_prompt)
    query_outputs = query_agent([query_input], query_system_prompt)
    # Assuming agent returns a list corresponding to output_fields
    query_response = query_outputs[0]
    query = query_response.content.strip()
    
    # Fallback if query is empty
    if not query:
        query = "民法典 彩礼返还 司法解释"

    # Step 3: Retrieval
    # Ensure the query includes "司法解释" explicitly if not present
    if "司法解释" not in query and "规定" not in query:
        query += " 司法解释 规定"

    # Attempt to use search engine if available in self, otherwise mock/skip
    try:
        if hasattr(self, 'search_engine'):
            retrieved_context = self.search_engine.multi_turn_search(query)
        else:
            # Fallback/Mock for migration context
            retrieved_context = "Search Engine not configured. Please rely on internal knowledge or configure self.search_engine."
    except Exception as e:
        retrieved_context = f"Search execution failed: {str(e)}"

    # Step 4: Summarize/Format with LLM
    system_prompt = (
        "You are a High Court Judge's Assistant. Your task is to extract and cite the exact legal provisions applicable to the case.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Identify the specific legal source (e.g., Civil Code, Judicial Interpretation on Caili).\n"
        "2. Focus on articles that define the SCOPE of the legal concept and EXCLUSIONS (e.g., what is NOT considered Caili, such as daily consumption or small gifts).\n"
        "3. Cite the full text. Format: **[Source Name] Article [Number]**: [Content]\n"
        "4. Explain relevance concisely, applying the rules to specific disputed amounts in the facts.\n"
        "5. PRIORITIZE finding 'Provisions of the Supreme People's Court on Several Issues Concerning the Application of Law in the Trial of Cases Involving Disputes over Betrothal Gifts' if applicable."
    )
    
    user_prompt = (
        f"Case Context: {facts}\n\n"
        f"Search Results: {retrieved_context}\n\n"
        "Provide the relevant legal articles and analysis."
    )
    
    # Create the main retrieval agent with CoT
    retrieval_agent = LLMAgentBase(['thinking', 'answer'], 'Legal_Knowledge_Retriever', model=self.node_model, temperature=0.0)
    
    # Prepare input
    analysis_input = Info(user_prompt)
    thinking, answer = retrieval_agent([analysis_input], system_prompt)
    
    # Return the result using make_final_answer
    return self.make_final_answer(thinking, answer)

func_string = inspect.getsource(forward)

LEGAL_KNOWLEDGE_RETRIEVER = {
    "thought": "This node performs legal knowledge retrieval by first generating targeted search keywords based on the facts, and then synthesizing relevant laws from search results. The migration replaces direct LLM client calls with LLMAgentBase agents and handles the search engine dependency.",
    "name": "Legal_Knowledge_Retriever",
    "code": """{func_string}""".format(func_string=func_string)
}