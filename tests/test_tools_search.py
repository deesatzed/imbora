"""Tests for src/tools/search.py — Tavily API wrapper."""

import json
import os

import httpx
import pytest

from src.core.config import SearchConfig
from src.core.exceptions import SearchError
from src.tools.search import SearchResult, TavilyClient


class TestSearchResult:
    def test_to_dict(self):
        result = SearchResult(title="Test", url="https://example.com", snippet="A snippet")
        d = result.to_dict()
        assert d["title"] == "Test"
        assert d["url"] == "https://example.com"
        assert d["snippet"] == "A snippet"

    def test_repr(self):
        result = SearchResult(title="Test", url="https://example.com", snippet="A snippet")
        assert "Test" in repr(result)
        assert "example.com" in repr(result)

    def test_empty_fields(self):
        result = SearchResult(title="", url="", snippet="")
        d = result.to_dict()
        assert d["title"] == ""


class TestTavilyClient:
    def test_init_defaults(self):
        client = TavilyClient()
        assert client.max_results == 3
        assert client.API_URL == "https://api.tavily.com/search"

    def test_init_custom_config(self):
        config = SearchConfig(max_results=5)
        client = TavilyClient(config=config)
        assert client.max_results == 5

    def test_init_custom_api_key(self):
        client = TavilyClient(api_key="test-key-123")
        assert client.api_key == "test-key-123"

    def test_search_no_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        client = TavilyClient(api_key="")
        with pytest.raises(SearchError, match="TAVILY_API_KEY not set"):
            client.search("test query")

    def test_search_documentation_no_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        client = TavilyClient(api_key="")
        with pytest.raises(SearchError, match="TAVILY_API_KEY not set"):
            client.search_documentation("pydantic")

    def test_parse_results(self):
        client = TavilyClient(api_key="test")
        data = {
            "results": [
                {"title": "Doc 1", "url": "https://doc1.com", "content": "Snippet 1"},
                {"title": "Doc 2", "url": "https://doc2.com", "content": "Snippet 2"},
                {"title": "Doc 3", "url": "https://doc3.com", "content": "Snippet 3"},
            ]
        }
        results = client._parse_results(data, limit=2)
        assert len(results) == 2
        assert results[0].title == "Doc 1"
        assert results[1].url == "https://doc2.com"

    def test_parse_results_content_as_snippet(self):
        client = TavilyClient(api_key="test")
        data = {
            "results": [
                {"title": "Doc", "url": "https://doc.com", "content": "The content text"},
            ]
        }
        results = client._parse_results(data, limit=5)
        assert results[0].snippet == "The content text"

    def test_parse_results_empty(self):
        client = TavilyClient(api_key="test")
        results = client._parse_results({}, limit=5)
        assert results == []

    def test_parse_results_missing_fields(self):
        client = TavilyClient(api_key="test")
        data = {"results": [{"title": "", "url": "", "content": "test"}]}
        results = client._parse_results(data, limit=5)
        # Empty title/url entries are skipped
        assert len(results) == 0

    def test_close(self):
        client = TavilyClient(api_key="test")
        # Access the client to create it
        _ = client.client
        assert client._client is not None
        client.close()
        assert client._client is None

    def test_close_without_init(self):
        client = TavilyClient(api_key="test")
        # Should not raise
        client.close()


# --- Integration tests (require TAVILY_API_KEY) ---

requires_tavily = pytest.mark.skipif(
    not os.getenv("TAVILY_API_KEY"),
    reason="TAVILY_API_KEY not set",
)


@requires_tavily
class TestTavilyClientIntegration:
    def test_real_search(self):
        client = TavilyClient()
        results = client.search("python pydantic documentation", num_results=3)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.title for r in results)
        client.close()

    def test_real_search_documentation(self):
        client = TavilyClient()
        results = client.search_documentation("httpx", topic="async client")
        assert len(results) > 0
        client.close()


def _tavily_response(results: list[dict]) -> bytes:
    """Build a Tavily-shaped JSON response body."""
    return json.dumps({"results": results}).encode("utf-8")


def _make_transport_client(handler) -> TavilyClient:
    """Return a TavilyClient whose internal httpx.Client uses the given transport."""
    transport = httpx.MockTransport(handler)
    client = TavilyClient(api_key="test-key-for-transport")
    # Replace the lazy-created httpx.Client with one backed by MockTransport
    client._client = httpx.Client(transport=transport, timeout=5.0)
    return client


class TestSearchHTTPPaths:
    """Exercise the real search() method with httpx.MockTransport."""

    def test_search_200_returns_results(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = _tavily_response(
                [
                    {"title": "Doc A", "url": "https://a.com", "content": "Snippet A"},
                    {"title": "Doc B", "url": "https://b.com", "content": "Snippet B"},
                ]
            )
            return httpx.Response(200, content=body, headers={"content-type": "application/json"})

        client = _make_transport_client(handler)
        results = client.search("test query")
        assert len(results) == 2
        assert results[0].title == "Doc A"
        assert results[1].url == "https://b.com"
        client.close()

    def test_search_401_raises_auth_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, content=b'{"error": "unauthorized"}')

        client = _make_transport_client(handler)
        with pytest.raises(SearchError, match="Invalid Tavily API key"):
            client.search("test")
        client.close()

    def test_search_429_raises_rate_limit(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, content=b'{"error": "rate limited"}')

        client = _make_transport_client(handler)
        with pytest.raises(SearchError, match="Tavily rate limit exceeded"):
            client.search("test")
        client.close()

    def test_search_500_raises_api_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, content=b'{"error": "internal"}')

        client = _make_transport_client(handler)
        with pytest.raises(SearchError, match="Tavily API error: HTTP 500"):
            client.search("test")
        client.close()

    def test_search_timeout_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timed out")

        client = _make_transport_client(handler)
        with pytest.raises(SearchError, match="timed out"):
            client.search("test")
        client.close()

    def test_search_connect_error_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = _make_transport_client(handler)
        with pytest.raises(SearchError, match="Failed to connect"):
            client.search("test")
        client.close()

    def test_search_generic_exception_wraps(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise RuntimeError("unexpected")

        client = _make_transport_client(handler)
        with pytest.raises(SearchError, match="search failed"):
            client.search("test")
        client.close()


class TestSearchDocumentationSorting:
    """Verify documentation search applies priority sorting."""

    def test_doc_priority_sorting(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = _tavily_response(
                [
                    {"title": "SO Answer", "url": "https://stackoverflow.com/q/123", "content": "so"},
                    {"title": "GitHub", "url": "https://github.com/lib/lib", "content": "gh"},
                    {"title": "RTD", "url": "https://lib.readthedocs.io/en/latest/", "content": "rtd"},
                    {"title": "Blog", "url": "https://blog.example.com/lib", "content": "blog"},
                ]
            )
            return httpx.Response(200, content=body, headers={"content-type": "application/json"})

        client = _make_transport_client(handler)
        results = client.search_documentation("lib")

        urls = [r.url for r in results]
        # readthedocs should sort first (priority 0)
        assert urls[0] == "https://lib.readthedocs.io/en/latest/"
        # github (priority 3) should be before stackoverflow (priority 4)
        github_idx = urls.index("https://github.com/lib/lib")
        so_idx = urls.index("https://stackoverflow.com/q/123")
        assert github_idx < so_idx
        client.close()

    def test_doc_search_appends_topic(self):
        captured_bodies: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_bodies.append(json.loads(request.content))
            body = _tavily_response([])
            return httpx.Response(200, content=body, headers={"content-type": "application/json"})

        client = _make_transport_client(handler)
        client.search_documentation("pydantic", topic="validation")

        assert len(captured_bodies) == 1
        query = captured_bodies[0]["query"]
        assert "pydantic" in query
        assert "validation" in query
        assert "python" in query
        client.close()
