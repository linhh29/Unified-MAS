code_template = """
def {node_name}(self, input_data):
    \"""
    node_id: {node_id}
    node_type: {node_type}
    description: {description}
    dependencies: {dependencies}
    input: {input}
    output: {output}
    \"""
    # ---------- Step 1: Process the input data ----------
    # input_data is a dictionary with the keys as the input names and the values as the input values
    # First, extract the input values from the input_data dictionary
    # Fill your code here

    # ---------- Step 2: Implement the node logic for one of the node types (LLM_Generator, Retrieval_RAG) ----------
    # Second, for LLM_Generator nodes, use LLMs to process the input data
    # For example, define the system prompt and user prompt: 
    # node_messages = [
    #     {"role": "system", "content": System Prompt from the prompt_template},
    #     {"role": "user", "content": User Prompt from the prompt_template (embed the input values) + Constraints from the constraints field},
    # ]
    # Then, call the LLM to get the output. If there are multiple LLM calls, you should call the LLMs according to the logic_description field.
    # For example, use self.llm_client.chat(node_messages, response_format='json_object') to get the json format output
    # Use self.llm_client.chat(node_messages, response_format='normal') to get the normal text output
    # Fill your code here




    # For Retrieval_RAG nodes, find the information that this node needs to retrieve from the logic_description
    # Use self.search_engine.multi_turn_search(information needed to retrieve) to get the retrieved context
    # Based on the retrieved context, use the summarization prompt template (marked as User Prompt: in the prompt_template) to summarize the retrieved context
    # Use self.llm_client.chat(node_messages, response_format='json_object') to get the json format output
    # Use self.llm_client.chat(node_messages, response_format='normal') to get the normal text output
    # Fill your code here

    # ---------- Step 3: Collect the output ----------
    # Finally, collect the output into a dictionary with the keys as the output names and the values as the output values
    # Fill your code here


    return output_data
"""

# print(LLM_Generator_code_template.format(node_name="Fact Structuring Agent", description="Extracts facts from the raw text"))