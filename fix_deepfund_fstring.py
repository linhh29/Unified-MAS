#!/usr/bin/env python3
"""Fix all_code in deepfund (and similar) generated_nodes.json: f-string expression cannot
   include backslash; f-string single '}' is not allowed. Use template variable + .format().
"""
import json
import sys
from pathlib import Path


def fix_input_normalizer_v2():
    # Simpler: one big triple-quoted string for the template, then .format(raw_sample=...)
    return '''def Input_Normalizer(self, input_data):
    """
    node_id: Input_Normalizer
    node_type: LLM_Generator
    description: Normalize and filter raw input JSON into canonical point-in-time JSON.
    dependencies: []
    input: raw_sample
    output: normalized
    """
    raw_sample = input_data.get('raw_sample')
    _t = (
        "Input: {raw_sample}\\n\\nTask: Produce a JSON object with these keys: {\\n"
        "  'trading_date': 'ISO8601 date string (YYYY-MM-DD)',\\n"
        "  'ticker': 'UPPERCASE ticker string',\\n"
        "  'company_news_filtered': [ {{'title','summary','publisher','publish_time','date'}} ],\\n"
        "  'policy_news_filtered': [ {{'topic','title','summary','publisher','publish_time','date'}} ],\\n"
        "  'insider_trades_filtered': [ {{'transaction_date','ticker','executive','executive_title','security_type','acquisition_or_disposal','shares','share_price','filed_date_or_empty'}} ],\\n"
        "  'technical_data_sorted_desc': [ {{'date','open','high','low','close','volume'}} ],\\n"
        "  'fundamentals': {{'cashflow','total_assets'}},\\n"
        "  'positions': {{ ... as input ... }},\\n"
        "  'issues': ['list of strings describing missing/malformed fields or empty if none']\\n"
        "}\\n\\nRules:\\n"
        "- Only include news items whose publish_time or date <= trading_date (treat publish_time and date as timestamps). Drop others.\\n"
        "- Convert all dates/publish_time to ISO8601 (if original is YYYYMMDDThhmmss convert to YYYY-MM-DD or full timestamp). Use YYYY-MM-DD if time absent.\\n"
        "- Sort technical_data_sorted_desc by date descending (most recent first).\\n"
        '- If insider trade records lack filed_date, set filed_date_or_empty to empty string "".\\n'
        "- If any required top-level keys are missing in input, include a descriptive string in issues.\\n"
        "- Output EXACT JSON and nothing else.\\n\\n"
        "Example small output (must follow exact keys):\\n"
        "{{ \\"trading_date\\":\\"2025-09-26\\",\\"ticker\\":\\"AAPL\\",\\"company_news_filtered\\":[],\\"policy_news_filtered\\":[],\\"insider_trades_filtered\\":[],\\"technical_data_sorted_desc\\":[],\\"fundamentals\\":{{\\"cashflow\\":26.0,\\"total_assets\\":106553.88}},\\"positions\\":{{ }},\\"issues\\":[] }}\\n\\n"
        "Now process the provided input and produce the JSON object."
    )
    user_content = _t.format(raw_sample=raw_sample)
    node_messages = [
        {"role": "system", "content": "You are a strict JSON normalizer for trading sample data. You MUST output ONLY a single valid JSON object exactly matching the schema described in the User Prompt. Do NOT include extra commentary."},
        {"role": "user", "content": user_content}
    ]
    response = self.llm_client.chat(node_messages, response_format='json_object')
    output_data = {"normalized": response}
    return output_data'''


def fix_retrieve_policy_context():
    return '''def Retrieve_Policy_Context(self, input_data):
    """
    node_id: Retrieve_Policy_Context
    node_type: Retrieval_RAG
    description: Retrieve external policy/regulatory context and summarize.
    dependencies: [Input_Normalizer]
    input: normalized.trading_date, normalized.ticker, normalized.policy_news_filtered
    output: retrieved_policy_context
    """
    normalized = input_data.get('normalized')
    trading_date = normalized.get('trading_date')
    ticker = normalized.get('ticker')
    query = f"policy regulatory news {ticker} sector OR US monetary policy around {trading_date} OR major regulatory actions {trading_date}"
    retrieved = self.search_engine.multi_turn_search(query)
    _t = (
        "Retrieved chunks: {retrieved}\\n\\nTask: From these retrieved items (which are external policy/regulatory news <= {trading_date}), produce JSON: {{\\n"
        "  'items': [ {{'id':'1','title':'','publisher':'','date':'ISO8601','short_summary':'one-sentence', 'relevance_score':0.0, 'url':''}} ... up to 6 items],\\n"
        "  'summary': 'one-paragraph summary of policy/regulatory context and how it might broadly affect the sector or ticker'\\n"
        "}}\\n\\nRules:\\n"
        "- Relevance scores 0..1. Provide dates ISO8601. If any item duplicates another, merge and note both URLs in url field separated by ' | '. Do NOT include any of the original sample's policy_news items; only external retrieved sources. Output JSON only."
    )
    user_content = _t.format(retrieved=retrieved, trading_date=trading_date)
    node_messages = [
        {"role": "system", "content": "You are a retrieval summarizer. You will receive retrieved chunks from a search engine. Do NOT invent facts beyond retrieved chunks. Output ONLY a JSON object in the schema requested."},
        {"role": "user", "content": user_content}
    ]
    response = self.llm_client.chat(node_messages, response_format='json_object')
    output_data = {"retrieved_policy_context": response}
    return output_data'''


def fix_news_ie_and_events():
    return '''def News_IE_and_Events(self, input_data):
    """
    node_id: News_IE_and_Events
    node_type: LLM_Generator
    description: Extract structured events from company news.
    dependencies: [Input_Normalizer]
    input: normalized.company_news_filtered, normalized.trading_date, normalized.ticker
    output: news_events
    """
    normalized = input_data.get('normalized')
    articles = normalized.get('company_news_filtered')
    trading_date = normalized.get('trading_date')
    ticker = normalized.get('ticker')
    _t = (
        "trading_date: {trading_date}\\nTicker: {ticker}\\nArticles: {articles}\\n\\n"
        "For each article produce: {{'article_id':'','events':[{{'event_id':'','type':'earnings|guidance|merger|regulatory|product|management|partnership|other',"
        "'main_actors':[{{'name':'','role':''}}],'action':'short phrase','object':'','impact_direction': -1|0|1,'impact_scope':'company|sector|economy',"
        "'event_time':'ISO8601 or null','evidence_span':'exact substring from title or summary','confidence':0.0}}]}}\\n\\n"
        "Rules:\\n- Max 12 events per article.\\n- evidence_span must exactly match a substring from title or summary (not paraphrase). "
        "If no good span, repeat the most relevant sentence from summary verbatim.\\n- confidence is between 0 and 1.\\n- Dates must be ISO8601 or null.\\n"
        "- Output only a single JSON mapping of article_id to events and nothing else."
    )
    user_content = _t.format(trading_date=trading_date, ticker=ticker, articles=articles)
    node_messages = [
        {"role": "system", "content": "You are a high-precision financial event extractor. Do not hallucinate. Output ONLY JSON with the exact schema described."},
        {"role": "user", "content": user_content}
    ]
    response = self.llm_client.chat(node_messages, response_format='json_object')
    output_data = {"news_events": response}
    return output_data'''


def main():
    default_path = "intermediate_result/deepfund/search/generated_nodes.json"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(default_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixes = {
        "Input_Normalizer": fix_input_normalizer_v2(),
        "Retrieve_Policy_Context": fix_retrieve_policy_context(),
        "News_IE_and_Events": fix_news_ie_and_events(),
    }
    for node in data.get("nodes", []):
        name = node.get("node_name")
        if name in fixes:
            node["all_code"] = fixes[name]
            print("Fixed node:", name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Saved:", path)


if __name__ == "__main__":
    main()
