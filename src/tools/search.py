"""Tavily API wrapper for live documentation search.

Queries Tavily's Search API to find current documentation
for libraries and frameworks used in tasks. Returns structured results
with titles, URLs, and relevant snippets.

The Associate uses this to counter the Knowledge Cutoff failure mode
by injecting live documentation into the Builder's context.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

from src.core.config import SearchConfig
from src.core.exceptions import SearchError

logger = logging.getLogger("associate.tools.search")


class SearchResult:
    """A single search result."""

    def __init__(self, title: str, url: str, snippet: str):
        self.title = title
        self.url = url
        self.snippet = snippet

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}

    def __repr__(self) -> str:
        return f"SearchResult(title={self.title!r}, url={self.url!r})"


class TavilyClient:
    """Client for Tavily Search API.

    Requires TAVILY_API_KEY environment variable.
    """

    API_URL = "https://api.tavily.com/search"

    def __init__(self, config: Optional[SearchConfig] = None, api_key: Optional[str] = None):
        self.config = config or SearchConfig()
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.max_results = self.config.max_results
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(30.0))
        return self._client

    def search(self, query: str, num_results: Optional[int] = None) -> list[SearchResult]:
        """Execute a web search query.

        Args:
            query: Search query string.
            num_results: Number of results to return (default from config).

        Returns:
            List of SearchResult objects.

        Raises:
            SearchError: If API call fails or API key missing.
        """
        if not self.api_key:
            raise SearchError("TAVILY_API_KEY not set")

        n = num_results or self.max_results

        payload = {
            "query": query,
            "max_results": n,
            "search_depth": "basic",
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self.client.post(self.API_URL, json=payload, headers=headers)

            if response.status_code == 401:
                raise SearchError("Invalid Tavily API key")
            if response.status_code == 429:
                raise SearchError("Tavily rate limit exceeded")
            if response.status_code >= 400:
                raise SearchError(f"Tavily API error: HTTP {response.status_code}")

            response.raise_for_status()
            data = response.json()

            return self._parse_results(data, n)

        except SearchError:
            raise
        except httpx.TimeoutException:
            raise SearchError("Tavily API request timed out")
        except httpx.ConnectError:
            raise SearchError("Failed to connect to Tavily API")
        except Exception as e:
            raise SearchError(f"Tavily search failed: {e}") from e

    def search_documentation(
        self, library_name: str, topic: Optional[str] = None
    ) -> list[SearchResult]:
        """Search for library documentation specifically.

        Constructs a documentation-focused query and filters results
        to prioritize official docs, GitHub, and Stack Overflow.

        Args:
            library_name: Name of the library (e.g., "psycopg3", "pydantic").
            topic: Optional specific topic (e.g., "connection pooling").

        Returns:
            List of SearchResult objects focused on documentation.
        """
        query = f"{library_name} documentation"
        if topic:
            query += f" {topic}"
        query += " python"

        results = self.search(query)

        # Sort: official docs first, then GitHub, then others
        def _doc_priority(r: SearchResult) -> int:
            url = r.url.lower()
            if "readthedocs" in url or ".readthedocs.io" in url:
                return 0
            if "docs.python.org" in url:
                return 1
            if library_name.lower() in url and ("doc" in url or "guide" in url):
                return 2
            if "github.com" in url:
                return 3
            if "stackoverflow.com" in url:
                return 4
            return 5

        return sorted(results, key=_doc_priority)

    def _parse_results(self, data: dict[str, Any], limit: int) -> list[SearchResult]:
        """Parse Tavily API response into SearchResult objects."""
        results = []

        for item in data.get("results", [])[:limit]:
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("content", "")
            if title and url:
                results.append(SearchResult(title=title, url=url, snippet=snippet))

        return results[:limit]

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
