"""
LLM Web Searcher - 能联网搜索的 LLM 系统
"""
from .llm_client import LLMClient
from .search_engines import (
    SearchEngineBase,
    SearchEngineFactory,
    GoogleSearchEngine,
    ScholarSearchEngine,
    GitHubSearchEngine,
)
from .web_search_llm import WebSearchLLM
from .utils import (
    normalize_arxiv_url,
    sanitize_filename,
    find_pdf_links,
    download_pdf,
)
from .content_fetcher import (
    fetch_urls_from_log,
    read_file_content,
)
from .strategy_analyzer import (
    classify_results_by_strategy,
    analyze_strategy_files,
    analyze_all_strategies,
)
from .prompts import (
    get_task_keywords_prompt,
    get_search_queries_prompt,
    get_strategy_analysis_prompt,
)

__all__ = [
    "LLMClient",
    "SearchEngineBase",
    "SearchEngineFactory",
    "GoogleSearchEngine",
    "ScholarSearchEngine",
    "GitHubSearchEngine",
    "WebSearchLLM",
    "normalize_arxiv_url",
    "sanitize_filename",
    "find_pdf_links",
    "download_pdf",
    "fetch_urls_from_log",
    "read_file_content",
    "classify_results_by_strategy",
    "analyze_strategy_files",
    "analyze_all_strategies",
    "get_task_keywords_prompt",
    "get_search_queries_prompt",
    "get_strategy_analysis_prompt",
]

__version__ = "0.1.0"


