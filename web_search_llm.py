"""
Web Search LLM - 整合 LLM 和搜索引擎的主类（同步版本）
"""
import asyncio
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import requests
from bs4 import BeautifulSoup

from .llm_client import LLMClient
from .search_engines import SearchEngineBase


class WebSearchLLM:
    """能联网搜索的 LLM 类（同步接口）"""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        search_engine: Optional[SearchEngineBase] = None,
        max_search_results: int = 10,
        max_rounds: int = 5,
        dataset_name: Optional[str] = None,
        mode: str = "search",
    ):
        """
        Args:
            llm_client: LLM 客户端实例，如果不提供则创建默认实例
            search_engine: 搜索引擎实例
            max_search_results: 最大搜索结果数量
            max_rounds: multi-turn search 的最大轮数
            dataset_name: 当前数据集名称（用于日志文件命名）
        """
        # 初始化 LLM 客户端
        if llm_client is None:
            self.llm_client = LLMClient()
        else:
            self.llm_client = llm_client
        
        # 初始化搜索引擎 & 配置
        self.search_engine = search_engine
        self.max_search_results = max_search_results
        self.max_rounds = max_rounds
        self.dataset_name = dataset_name

        intermediate_dir = Path(__file__).parent / "intermediate_result"
        intermediate_dir.mkdir(exist_ok=True)
        ds_name = self.dataset_name
        # 为每个dataset创建子目录
        dataset_dir = intermediate_dir / ds_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        if mode == "optimize":
            optimize_dir = dataset_dir / "optimize"
            optimize_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = optimize_dir / "validate_multi_turn_search_log.jsonl"
        elif mode == "search":
            # 在 dataset 下创建 search 子目录
            search_dir = dataset_dir / "search"
            search_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = search_dir / "multi_turn_search_log.jsonl"
        elif mode == "test":
            test_dir = dataset_dir / "test"
            test_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = test_dir / "test_multi_turn_search_log.jsonl"
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def _fetch_webpage_snippet(self, url: str, max_length: int = 5000) -> str:
        """
        抓取网页内容的前 N 个字符作为 snippet
        
        Args:
            url: 网页 URL
            max_length: 最大字符数
            
        Returns:
            网页内容的前 N 个字符
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            
            if response.status_code != 200:
                return f"[Error: HTTP {response.status_code}]"
            
            # 尝试解析 HTML
            try:
                soup = BeautifulSoup(response.content, 'html.parser')
                # 移除 script 和 style 标签
                for script in soup(["script", "style"]):
                    script.decompose()
                # 提取文本
                text = soup.get_text(separator=' ', strip=True)
                # 清理多余空白
                text = ' '.join(text.split())
                return text[:max_length]
            except:
                # 如果 HTML 解析失败，尝试直接解码
                try:
                    content = response.content.decode('utf-8', errors='ignore')
                    return content[:max_length]
                except:
                    return "[Error: Failed to decode content]"
        except Exception as e:
            return f"[Error: {str(e)}]"
    
    def multi_turn_search(
        self,
        target_description: str,
    ) -> Dict[str, Any]:
        """
        使用 LLM 控制的 multi-turn 搜索流程。
        
        给定一个目标描述（例如若干 Strategy / Query / Reasoning 的文本），
        由 LLM 负责：
        1）判断当前是否已经有足够匹配该目标的搜索结果；
        2）如果不够，生成下一轮搜索 query；
        3）在达到充足信息或轮数上限时终止，并给出总结。
        
        Returns:
            {
                "final_summary": str,          # LLM 对已找到内容的总结
                "rounds": [                    # 每一轮的搜索记录
                    {
                        "round": int,
                        "query": str,
                        "reasoning": str,
                        "results": [ {...}, ... ]  # 每条为 search_engine.search 返回的一条结果
                    },
                    ...
                ],
                "all_results": [ {...}, ... ]  # 聚合后的所有结果
            }
        """
        rounds: List[Dict[str, Any]] = []
        all_results: List[Dict[str, str]] = []
        final_summary: str = ""

        for round_idx in range(1, self.max_rounds + 1):
            # 构造历史搜索轮次的简要描述，供 LLM 决策
            if rounds:
                history_lines: List[str] = []
                for r in rounds[-2:]:  # 只给最近 2 轮，避免上下文过长
                    history_lines.append(f"Round {r['round']} query: {r['query']}")
                    history_lines.append(f"Round {r['round']} reasoning: {r.get('reasoning', '')}")
                    for i, res in enumerate(r["results"][:2]):  # 每轮最多展示前 2 条结果
                        title = res.get("title", "")[:200]
                        snippet = res.get("snippet", "")[:400]
                        history_lines.append(
                            f"  Result {i+1}: title={title!r}, snippet={snippet!r}"
                        )
                history_str = "\n".join(history_lines)
            else:
                history_str = "No search has been executed yet (this is the first round)."

            controller_system = (
                "You are a web search controller. "
                "Your job is to decide, step by step, how to search the web so that the user can "
                "find content that matches the target description. "
                "At each round, you will see the target description and a summary of past searches "
                "and results, and you must decide whether more search is needed. "
                "You MUST respond with a valid JSON object only, no extra text."
            )

            # 根据底层搜索引擎类型给出不同的查询构造指引
            engine_type = type(self.search_engine).__name__ 
            if engine_type == "GitHubSearchEngine":
                from_engine_name = "GitHub"
                engine_hint = (
                    "Current search backend: GitHub repository search.\n"
                    "- You MUST construct queries that look like GitHub repo searches, NOT natural language questions.\n"
                    "- Focus on a few core keywords: domain, task, entities, and techniques.\n"
                    "- Prefer short keyword-style queries, optionally with GitHub qualifiers such as "
                    "'language:python', 'in:name,description,readme', 'stars:>10'.\n"
                    "- Avoid 'survey of', 'methods for', 'towards', or very long sentences in the query.\n"
                )
            elif engine_type == "ScholarSearchEngine":
                from_engine_name = "Scholar"
                engine_hint = (
                    "Current search backend: Google Scholar (academic papers).\n"
                    "- You should construct queries that look like paper titles or combinations of technical terms.\n"
                    "- It is good to include phrases like 'survey', 'review', 'state of the art' when searching for overviews.\n"
                    "- Focus on scientific keywords (task, domain, methodology) rather than implementation details.\n"
                )
            elif engine_type == "GoogleSearchEngine":
                from_engine_name = "Google"
                engine_hint = (
                    "Current search backend: general Google web search.\n"
                    "- You may mix natural language with key technical terms.\n"
                    "- Focus on retrieving background knowledge, blog posts, documentation, or tutorials relevant to the target description.\n"
                )

            controller_user = f"""
Target description:
{target_description}

Search round: {round_idx} / {self.max_rounds}

Past search rounds:
{history_str}

Search engine context:
- Backend type: {engine_type}
- Instructions:
{engine_hint}

Your task in THIS round:
1. Carefully read the target description and past search results.
2. Decide whether we already have enough information that clearly matches the target description.
3. If yes, set "done": true and summarize the useful information we already have.
4. If no, set "done": false and propose the NEXT web search query to run.

Output JSON schema (you must strictly follow):
{{
  "done": bool,                 // true if we already have enough matching information
  "need_search": bool,          // whether to run another web search in this round
  "next_query": str,            // the next search query to run (empty if done=true)
  "reasoning": str,             // your reasoning for this decision
  "summary": str                // if done=true, summarize what has been found and why it matches
}}
"""

            controller_messages = [
                {"role": "system", "content": controller_system},
                {"role": "user", "content": controller_user},
            ]

            # 调用 LLM 决策本轮是否需要继续搜索（同步执行）
            decision_str = self.llm_client.chat(
                controller_messages,
                response_format="json_object",
            )
            raw_decision = json.loads(decision_str)
            if isinstance(raw_decision, list):
                decision = raw_decision[0] if raw_decision else {}
            else:
                decision = raw_decision
            done = bool(decision.get("done", False))
            need_search = bool(decision.get("need_search", not done))
            next_query = (decision.get("next_query") or "").strip()
            reasoning = decision.get("reasoning", "")
            summary = decision.get("summary", "")

            if done or (not need_search) or (not next_query):
                final_summary = summary or final_summary
                break

            # 运行本轮搜索（同步）
            results = self._search(next_query)
            # 为每条结果标注其来自第几个 turn / round
            for r in results:
                # 使用 'round' 作为字段名，表示来自第几轮搜索
                r["turn_index"] = round_idx
                r["from"] = from_engine_name

            rounds.append(
                {
                    "round": round_idx,
                    "query": next_query,
                    "reasoning": reasoning,
                    "results": results,
                }
            )
            all_results.extend(results)

        # 使用 LLM 在 all_results 中筛选出与 target_description 最接近的最多 10 条结果
        if all_results:
            filter_system = (
                "You are an expert search result ranker. "
                "Your job is to select the most semantically relevant results to the target description."
            )
            filter_user = (
                "Target description:\n"
                f"{target_description}\n\n"
                "Here is a JSON array named 'results', where each item has fields: turn_index, title, snippet, url.\n"
                "Your task:\n"
                "1. Read the target description and all results.\n"
                "2. Select the TOP 10 results that are most semantically relevant to the target description.\n"
                "3. If there are fewer than 10 relevant ones, just return all relevant indices.\n\n"
                "Output JSON ONLY in the following format:\n"
                "{\n"
                '  "selected_indices": [i1, i2, ...]  // indices from the original `results` array\n'
                "}\n\n"
                "Do NOT include any extra keys, comments or text outside this JSON object."
            )

            filter_messages = [
                {"role": "system", "content": filter_system},
                {"role": "user", "content": filter_user + "\n\nresults:\n" + json.dumps(all_results, ensure_ascii=False)},
            ]

            selection_str = self.llm_client.chat(
                filter_messages,
                response_format="json_object",
            )
            selection = json.loads(selection_str)
            selected_indices = selection.get("selected_indices", [])
            # 根据索引过滤，最多 10 条
            filtered_results = [
                all_results[i]
                for i in selected_indices
                if isinstance(i, int) and 0 <= i < len(all_results)
            ][:10]
            all_results = filtered_results

        # 本次 multi-turn 搜索的结果结构（all_results 为 LLM 筛选后的最多 10 条）
        result = {
            "final_summary": final_summary,
            "rounds": rounds,
            "all_results": all_results,
        }

        # 为日志构造一个映射：key 是 target_description，value 是结果
        log_record = {target_description: result}

        # 将 multi-turn 搜索的整体结果写入 intermediate_result 目录下的日志文件
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
            print(f"\n[Multi-turn Search] Logged result to: {self.log_file}")
        except Exception as e:
            print(f"[Multi-turn Search] Failed to log result: {e}")

        return result
    
    async def multi_turn_search_async(
        self,
        target_description: str,
    ) -> Dict[str, Any]:
        """
        异步版本的 multi-turn 搜索流程。
        
        使用 asyncio.to_thread 将同步的 LLM 调用和搜索操作包装成异步执行，
        从而不会阻塞事件循环。
        
        Args:
            target_description: 目标描述文本
            
        Returns:
            与同步版本相同的返回结构
        """
        rounds: List[Dict[str, Any]] = []
        all_results: List[Dict[str, str]] = []
        final_summary: str = ""

        for round_idx in range(1, self.max_rounds + 1):
            # 构造历史搜索轮次的简要描述，供 LLM 决策
            if rounds:
                history_lines: List[str] = []
                for r in rounds[-2:]:  # 只给最近 2 轮，避免上下文过长
                    history_lines.append(f"Round {r['round']} query: {r['query']}")
                    history_lines.append(f"Round {r['round']} reasoning: {r.get('reasoning', '')}")
                    for i, res in enumerate(r["results"][:2]):  # 每轮最多展示前 2 条结果
                        title = res.get("title", "")[:200]
                        snippet = res.get("snippet", "")[:400]
                        history_lines.append(
                            f"  Result {i+1}: title={title!r}, snippet={snippet!r}"
                        )
                history_str = "\n".join(history_lines)
            else:
                history_str = "No search has been executed yet (this is the first round)."

            controller_system = (
                "You are a web search controller. "
                "Your job is to decide, step by step, how to search the web so that the user can "
                "find content that matches the target description. "
                "At each round, you will see the target description and a summary of past searches "
                "and results, and you must decide whether more search is needed. "
                "You MUST respond with a valid JSON object only, no extra text."
            )

            # 根据底层搜索引擎类型给出不同的查询构造指引
            engine_type = type(self.search_engine).__name__ 
            if engine_type == "GitHubSearchEngine":
                from_engine_name = "GitHub"
                engine_hint = (
                    "Current search backend: GitHub repository search.\n"
                    "- You MUST construct queries that look like GitHub repo searches, NOT natural language questions.\n"
                    "- Focus on a few core keywords: domain, task, entities, and techniques.\n"
                    "- Prefer short keyword-style queries, optionally with GitHub qualifiers such as "
                    "'language:python', 'in:name,description,readme', 'stars:>10'.\n"
                    "- Avoid 'survey of', 'methods for', 'towards', or very long sentences in the query.\n"
                )
            elif engine_type == "ScholarSearchEngine":
                from_engine_name = "Scholar"
                engine_hint = (
                    "Current search backend: Google Scholar (academic papers).\n"
                    "- You should construct queries that look like paper titles or combinations of technical terms.\n"
                    "- It is good to include phrases like 'survey', 'review', 'state of the art' when searching for overviews.\n"
                    "- Focus on scientific keywords (task, domain, methodology) rather than implementation details.\n"
                )
            elif engine_type == "GoogleSearchEngine":
                from_engine_name = "Google"
                engine_hint = (
                    "Current search backend: general Google web search.\n"
                    "- You may mix natural language with key technical terms.\n"
                    "- Focus on retrieving background knowledge, blog posts, documentation, or tutorials relevant to the target description.\n"
                )

            controller_user = f"""
Target description:
{target_description}

Search round: {round_idx} / {self.max_rounds}

Past search rounds:
{history_str}

Search engine context:
- Backend type: {engine_type}
- Instructions:
{engine_hint}

Your task in THIS round:
1. Carefully read the target description and past search results.
2. Decide whether we already have enough information that clearly matches the target description.
3. If yes, set "done": true and summarize the useful information we already have.
4. If no, set "done": false and propose the NEXT web search query to run.

Output JSON schema (you must strictly follow):
{{
  "done": bool,                 // true if we already have enough matching information
  "need_search": bool,          // whether to run another web search in this round
  "next_query": str,            // the next search query to run (empty if done=true)
  "reasoning": str,             // your reasoning for this decision
  "summary": str                // if done=true, summarize what has been found and why it matches
}}
"""

            controller_messages = [
                {"role": "system", "content": controller_system},
                {"role": "user", "content": controller_user},
            ]

            # 异步调用 LLM 决策本轮是否需要继续搜索
            decision_str = await asyncio.to_thread(
                self.llm_client.chat,
                controller_messages,
                "json_object",
            )
            raw_decision = json.loads(decision_str)
            if isinstance(raw_decision, list):
                decision = raw_decision[0] if raw_decision else {}
            else:
                decision = raw_decision
            done = bool(decision.get("done", False))
            need_search = bool(decision.get("need_search", not done))
            next_query = (decision.get("next_query") or "").strip()
            reasoning = decision.get("reasoning", "")
            summary = decision.get("summary", "")

            if done or (not need_search) or (not next_query):
                final_summary = summary or final_summary
                break

            # 异步运行本轮搜索
            results = await asyncio.to_thread(self._search, next_query)
            new_results = []

            # remove aime 
            for r in results:
                if "aime" not in r["title"].lower() and "aime" not in r["snippet"].lower():
                    new_results.append(r)
            results = new_results
            
            # 为每条结果标注其来自第几个 turn / round
            for r in results:
                # 使用 'round' 作为字段名，表示来自第几轮搜索
                r["turn_index"] = round_idx
                r["from"] = from_engine_name

            rounds.append(
                {
                    "round": round_idx,
                    "query": next_query,
                    "reasoning": reasoning,
                    "results": results,
                }
            )
            all_results.extend(results)

        # 使用 LLM 在 all_results 中筛选出与 target_description 最接近的最多 10 条结果
        if all_results:
            filter_system = (
                "You are an expert search result ranker. "
                "Your job is to select the most semantically relevant results to the target description."
            )
            filter_user = (
                "Target description:\n"
                f"{target_description}\n\n"
                "Here is a JSON array named 'results', where each item has fields: turn_index, title, snippet, url.\n"
                "Your task:\n"
                "1. Read the target description and all results.\n"
                "2. Select the TOP 10 results that are most semantically relevant to the target description.\n"
                "3. If there are fewer than 10 relevant ones, just return all relevant indices.\n\n"
                "Output JSON ONLY in the following format:\n"
                "{\n"
                '  "selected_indices": [i1, i2, ...]  // indices from the original `results` array\n'
                "}\n\n"
                "Do NOT include any extra keys, comments or text outside this JSON object."
            )

            filter_messages = [
                {"role": "system", "content": filter_system},
                {"role": "user", "content": filter_user + "\n\nresults:\n" + json.dumps(all_results, ensure_ascii=False)},
            ]

            # 异步调用 LLM 进行结果筛选
            selection_str = await asyncio.to_thread(
                self.llm_client.chat,
                filter_messages,
                "json_object",
            )
            selection = json.loads(selection_str)
            selected_indices = selection.get("selected_indices", [])
            # 根据索引过滤，最多 10 条
            filtered_results = [
                all_results[i]
                for i in selected_indices
                if isinstance(i, int) and 0 <= i < len(all_results)
            ][:10]
            all_results = filtered_results

        # 本次 multi-turn 搜索的结果结构（all_results 为 LLM 筛选后的最多 10 条）
        result = {
            "final_summary": final_summary,
            "rounds": rounds,
            "all_results": all_results,
        }

        # 为日志构造一个映射：key 是 target_description，value 是结果
        log_record = {target_description: result}

        # 异步写入日志文件
        def _write_log():
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                print(f"\n[Multi-turn Search] Logged result to: {self.log_file}")
            except Exception as e:
                print(f"[Multi-turn Search] Failed to log result: {e}")
        
        await asyncio.to_thread(_write_log)

        return result
    
    def _search(self, query: str) -> List[Dict[str, str]]:
        """
        Execute search with concurrency control.
        
        Args:
            query: Search query
            
        Returns:
            List of search results (each result is a dict from the search engine)
        """
        # 清理查询字符串：将双引号替换为空格，避免搜索失败
        query = query.replace('"', ' ').replace('"', ' ')
        # 清理多余的空格
        query = ' '.join(query.split())
        
        print(f"\n[Search] Executing search for: '{query}'")
        try:
            # 同步调用底层搜索引擎
            results = self.search_engine.search(
                query,
                num_results=self.max_search_results,
            )
            print(f"[Search Result] Found {len(results)} results for '{query}'")
            if results:
                print("  Results detail:")
                for idx, res in enumerate(results, 1):
                    title = res.get("title", "No title")
                    url = res.get("url", "")

                    # if self.is_optimize:
                    #     # 从 URL 抓取网页内容作为 snippet
                    #     snippet = self._fetch_webpage_snippet(url, max_length=5000)
                    #     # 更新结果中的 snippet
                    #     res["snippet"] = snippet
                    # else:
                    snippet = res.get("snippet", "No snippet")
                    
                    print(f"    [{idx}] Title: {title}")
                    print(f"        URL: {url}")
                    print(f"        Snippet: {snippet[:200]}...\n" if len(snippet) > 200 else f"        Snippet: {snippet}\n")
            return results
        except Exception as e:
            print(f"[Error] Search error for '{query}': {e}")
            return []
    
    