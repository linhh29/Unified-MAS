#!/usr/bin/env python3
"""Fix all_code in generated_nodes.json: f-string expression cannot include backslash.
   Move the JSON schema literal into a variable so the f-string only has {raw_query_text} and {schema_and_rules}.
"""
import json
import sys

def main():
    path = "intermediate_result/travelplanner/search/generated_nodes.json"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    with open(path, "r") as f:
        data = json.load(f)

    schema_literal = (
        '{\\n  "origin": string|null,\\n  "destinations": [ { "city": string, "coords": null }, ... ],\\n'
        '  "start_date": "YYYY-MM-DD"|null,\\n  "end_date": "YYYY-MM-DD"|null,\\n  "days": int|null,\\n'
        '  "travelers": int|null,\\n  "budget": { "currency": "USD", "max": number }|null,\\n'
        '  "transport_pref": string|null,\\n  "accommodations_pref": string|null,\\n'
        '  "must_haves": [string],\\n  "nice_to_haves": [string],\\n'
        '  "strictness": { "dates":"must"|"soft", "budget":"must"|"soft" },\\n'
        '  "notes": string|null,\\n  "ambiguous": true|false,\\n'
        '  "candidates": [ {"field":"...","interpretations":["..."]} ]\\n}\\n\\n'
        "Parsing rules: convert dates to ISO, parse travelers and budget, set strictness as described. "
        "Example output provided in prompt. Now produce JSON only for this input."
    )

    fixed_all_code = '''def Input_Normalization(self, input_data):
    """
    node_id: Input_Normalization
    node_type: LLM_Generator
    description: Extract canonical query from raw text
    dependencies: []
    input: raw_query_text, context_hint
    output: canonical_query
    """
    raw_query_text = input_data.get('raw_query_text')
    context_hint = input_data.get('context_hint')
    schema_and_rules = (
        "{\\n  \\"origin\\": string|null,\\n  \\"destinations\\": [ { \\"city\\": string, \\"coords\\": null }, ... ],\\n"
        "  \\"start_date\\": \\"YYYY-MM-DD\\"|null,\\n  \\"end_date\\": \\"YYYY-MM-DD\\"|null,\\n  \\"days\\": int|null,\\n"
        "  \\"travelers\\": int|null,\\n  \\"budget\\": { \\"currency\\": \\"USD\\", \\"max\\": number }|null,\\n"
        "  \\"transport_pref\\": string|null,\\n  \\"accommodations_pref\\": string|null,\\n"
        "  \\"must_haves\\": [string],\\n  \\"nice_to_haves\\": [string],\\n"
        "  \\"strictness\\": { \\"dates\\":\\"must\\"|\\"soft\\", \\"budget\\":\\"must\\"|\\"soft\\" },\\n"
        "  \\"notes\\": string|null,\\n  \\"ambiguous\\": true|false,\\n"
        "  \\"candidates\\": [ {\\"field\\":\\"...\\",\\"interpretations\\":[\\"...\\"]} ]\\n}\\n\\n"
        "Parsing rules: convert dates to ISO, parse travelers and budget, set strictness as described. Example output provided in prompt. Now produce JSON only for this input."
    )
    node_messages = [
        {"role": "system", "content": "You are a precise extractor that converts a travel request into a canonical JSON. Do NOT invent facts. Carefully validate numbers and dates; if ambiguous, set field null and include `ambiguous:true` and `candidates` array explaining possibilities. Output only valid JSON and nothing else."},
        {"role": "user", "content": f"Input query: \\"{raw_query_text}\\"\\n\\nRequired JSON schema and rules (produce exactly these keys):\\n{schema_and_rules}"}
    ]
    response = self.llm_client.chat(node_messages, response_format='json_object')
    output_data = {"canonical_query": response}
    return output_data'''

    for node in data.get("nodes", []):
        if node.get("node_name") == "Input_Normalization":
            node["all_code"] = fixed_all_code
        elif node.get("node_name") == "Retrieve_Catalog":
            # Fix f-string with backslash in expression: move JSON template to a variable
            old_retrieve = '''    # Now call LLM to assemble and validate the retrieval grouping
    node_messages = [
        {"role":"system","content":"You are a retrieval summarizer. You were given a set of retrieved raw text chunks (from the user's provided 'Given information') and must output a JSON object that groups and returns the chunks verbatim with their source descriptions. DO NOT add extra facts or external info. Output JSON only."},
        {"role":"user","content": f"Retrieved chunks for the planning query. Return JSON grouping as {\\"flights\\":[],\\"hotels\\":[],\\"pois\\":[],\\"restaurants\\":[],\\"transports\\":[] } and include each chunk as {\\"source\\":...,\\"content\\":...}. Here are the raw chunks: {all_chunks}"}
    ]'''
            new_retrieve = '''    # Now call LLM to assemble and validate the retrieval grouping
    json_grouping_instruction = "Return JSON grouping as {\\"flights\\":[],\\"hotels\\":[],\\"pois\\":[],\\"restaurants\\":[],\\"transports\\":[] } and include each chunk as {\\"source\\":...,\\"content\\":...}."
    node_messages = [
        {"role":"system","content":"You are a retrieval summarizer. You were given a set of retrieved raw text chunks (from the user's provided 'Given information') and must output a JSON object that groups and returns the chunks verbatim with their source descriptions. DO NOT add extra facts or external info. Output JSON only."},
        {"role":"user","content": f"Retrieved chunks for the planning query. {json_grouping_instruction} Here are the raw chunks: {all_chunks}"}
    ]'''
            if old_retrieve in node["all_code"]:
                node["all_code"] = node["all_code"].replace(old_retrieve, new_retrieve)
            else:
                print("Warning: Retrieve_Catalog all_code pattern not found, skipping")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Fixed Input_Normalization and Retrieve_Catalog all_code in", path)

if __name__ == "__main__":
    main()
