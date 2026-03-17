"""
搜索引擎模块 - 提供多个搜索引擎实现

当前仅保留 3 个搜索引擎，分别基于不同的后端：
- Google: 使用 serper.dev 提供的 Google Search API (`https://google.serper.dev/search`)
- Google Scholar: 使用 `scholarly` Python 包从 Google Scholar 抓取结果
  参考项目: https://github.com/scholarly-python-package/scholarly
- GitHub: 使用 GitHub 官方 REST Search API
  文档: https://docs.github.com/en/rest/search/search?apiVersion=2022-11-28
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import json

import requests
from scholarly import scholarly as scholarly_client


class SearchEngineBase(ABC):
    """搜索引擎基类"""

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        执行搜索

        Args:
            query: 搜索查询
            num_results: 返回结果数量

        Returns:
            搜索结果列表，每个结果包含 title, url, snippet
        """
        raise NotImplementedError


class GoogleSearchEngine(SearchEngineBase):
    """基于 serper.dev 的 Google 搜索引擎实现（需要 serper API Key）"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: serper.dev 的 API Key
        """
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """使用 serper.dev 的 Google Search 接口搜索网页"""
        if not self.api_key:
            print("Warning: serper API key not provided for GoogleSearchEngine. Returning empty results.")
            return []

        try:
            payload = {"q": query, "num": num_results}
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            }
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=20)
            if resp.status_code != 200:
                print(f"Google (serper) search error: {resp.status_code} - {resp.text}")
                return []
            data = resp.json()

            results: List[Dict[str, str]] = []
            # serper.dev 返回的 organic 结果列表
            for item in data.get("organic", [])[:num_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", "") or item.get("title", ""),
                    }
                )
            return results
        except Exception as e:
            print(f"Google (serper) search error: {e}")
            return []


class ScholarSearchEngine(SearchEngineBase):
    """基于 serper.dev scholar 接口的 Google Scholar 搜索引擎实现（需要 serper API Key）"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: serper.dev 的 API Key
        """
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/scholar"
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        使用 serper.dev 的 scholar 接口从 Google Scholar 搜索学术论文。
        
        参考 serper 示例：
        POST https://google.serper.dev/scholar
        Headers: X-API-KEY, Content-Type: application/json
        Body: {"q": "..."}
        """
        if not self.api_key:
            print("Warning: serper API key not provided for ScholarSearchEngine. Returning empty results.")
            return []
        
        try:
            payload = {"q": query, "num": num_results}
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            }
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=20)
            if resp.status_code != 200:
                print(f"Scholar (serper) search error: {resp.status_code} - {resp.text}")
                return []
            data = resp.json()

            results: List[Dict[str, str]] = []
            # serper scholar 返回的 organic 结果列表（字段结构与 web 搜索类似）
            for item in data.get("organic", [])[:num_results]:
                title = item.get("title", "")
                url = item.get("link", "")
                # 摘要可以来自 snippet 或 publication_info.summary 等字段
                snippet = (
                    item.get("snippet", "")
                    or item.get("publication_info", {}).get("summary", "")
                    or title
                )
                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    }
                )
            return results
        except Exception as e:
            print(f"Scholar (serper) search error: {e}")
            return []


class GitHubSearchEngine(SearchEngineBase):
    """基于 GitHub REST Search API 的 GitHub 搜索引擎实现"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: GitHub Personal Access Token（可选，但强烈建议提供以提高速率限制）
        """
        self.api_key = api_key
        # 参考文档: https://docs.github.com/en/rest/search/search?apiVersion=2022-11-28
        self.base_url = "https://api.github.com/search/repositories"

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """使用 GitHub 官方 Search API 搜索代码仓库"""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        params = {
            "q": query,
            "per_page": num_results,
        }

        try:
            resp = requests.get(self.base_url, headers=headers, params=params, timeout=20)
            if resp.status_code != 200:
                print(f"GitHub search error: {resp.status_code} - {resp.text}")
                return []
            data = resp.json()

            results: List[Dict[str, str]] = []
            for item in data.get("items", [])[:num_results]:
                # GitHub repo 对象字段参考: name, full_name, html_url, description, language
                title = item.get("full_name") or item.get("name", "")
                url = item.get("html_url", "")
                snippet = (
                    item.get("description", "")
                    or json.dumps({"language": item.get("language")}, ensure_ascii=False)
                )
                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    }
                )
            return results
        except Exception as e:
            print(f"GitHub search error: {e}")
            return []


class SearchEngineFactory:
    """搜索引擎工厂类"""

    @staticmethod
    def create_engine(engine_name: str, api_key=None) -> SearchEngineBase:
        """
        创建搜索引擎实例

        Args:
            engine_name: 搜索引擎名称 ('google', 'scholar', 'github')
            **kwargs: 搜索引擎特定的参数

        Returns:
            搜索引擎实例
        """
        engine_name = engine_name.lower()

        if engine_name == "google":
            # 这里的 api_key 对应 serper.dev 的 X-API-KEY
            assert api_key is not None, "api_key is required for GoogleSearchEngine"
            return GoogleSearchEngine(api_key=api_key)
        elif engine_name == "scholar":
            # 这里的 api_key 对应 serper.dev 的 X-API-KEY（scholar endpoint）
            assert api_key is not None, "api_key is required for ScholarSearchEngine"
            return ScholarSearchEngine(api_key=api_key)
        elif engine_name == "github":
            # 这里的 api_key 对应 GitHub Personal Access Token
            assert api_key is not None, "api_key is required for GitHubSearchEngine"
            return GitHubSearchEngine(api_key=api_key)
        else:
            raise ValueError(f"Unknown search engine: {engine_name}")


