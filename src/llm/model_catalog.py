"""Live OpenRouter model catalog utilities."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx


class OpenRouterModelCatalog:
    """Fetch and rank current OpenRouter models by multiple criteria."""

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.timeout_seconds = timeout_seconds

    def list_models(
        self,
        *,
        criteria: str = "newness",
        limit: int = 20,
        provider: str | None = None,
        search: str | None = None,
        max_prompt_cost: float | None = None,
        max_completion_cost: float | None = None,
        include_free: bool = True,
        latency_probe_limit: int = 80,
    ) -> list[dict[str, Any]]:
        """Return ranked model records for UI/CLI consumption."""
        raw_models = self._fetch_models()
        entries = [self._to_entry(row) for row in raw_models]

        if provider:
            provider_lower = provider.lower().strip()
            entries = [e for e in entries if e["provider"].lower() == provider_lower]

        if search:
            q = search.lower().strip()
            entries = [
                e
                for e in entries
                if q in e["id"].lower()
                or q in (e.get("name") or "").lower()
                or q in (e.get("description") or "").lower()
            ]

        if max_prompt_cost is not None:
            entries = [
                e
                for e in entries
                if e["prompt_cost"] is not None and e["prompt_cost"] <= max_prompt_cost
            ]

        if max_completion_cost is not None:
            entries = [
                e
                for e in entries
                if e["completion_cost"] is not None and e["completion_cost"] <= max_completion_cost
            ]

        if not include_free:
            entries = [
                e for e in entries if (e["blended_cost"] is not None and e["blended_cost"] > 0)
            ]

        if criteria == "latency":
            self._attach_latency_metrics(
                entries=entries,
                probe_limit=max(1, latency_probe_limit),
            )
            entries.sort(key=_latency_sort_key)
        elif criteria == "cost":
            entries.sort(key=_cost_sort_key)
        else:
            entries.sort(key=_newness_sort_key)

        trimmed = entries[: max(limit, 1)]
        for row in trimmed:
            row["created_iso"] = _to_iso(row.get("created_at"))
            row.pop("created_at", None)
        return trimmed

    def _fetch_models(self) -> list[dict[str, Any]]:
        url = f"{self.base_url}/models"
        headers = {
            "Accept": "application/json",
            "HTTP-Referer": "https://github.com/the-associate",
            "X-Title": "The Associate",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return []
        return [row for row in data if isinstance(row, dict)]

    def _fetch_latency_metrics(self, model_id: str) -> dict[str, float | None]:
        url = f"{self.base_url}/models/{model_id}/endpoints"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
        except Exception:
            return {"latency_ms": None, "throughput": None, "uptime": None}

        data = payload.get("data") if isinstance(payload, dict) else None
        endpoints = data.get("endpoints") if isinstance(data, dict) else None
        if not isinstance(endpoints, list):
            return {"latency_ms": None, "throughput": None, "uptime": None}

        latency_candidates: list[float] = []
        throughput_candidates: list[float] = []
        uptime_candidates: list[float] = []
        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                continue
            if endpoint.get("status") not in (0, None):
                continue
            latency = _as_float(endpoint.get("latency_last_30m"))
            throughput = _as_float(endpoint.get("throughput_last_30m"))
            uptime = _as_float(endpoint.get("uptime_last_30m"))
            if latency is not None:
                latency_candidates.append(latency)
            if throughput is not None:
                throughput_candidates.append(throughput)
            if uptime is not None:
                uptime_candidates.append(uptime)

        return {
            "latency_ms": min(latency_candidates) if latency_candidates else None,
            "throughput": max(throughput_candidates) if throughput_candidates else None,
            "uptime": max(uptime_candidates) if uptime_candidates else None,
        }

    def _attach_latency_metrics(self, entries: list[dict[str, Any]], probe_limit: int) -> None:
        probe_candidates = sorted(entries, key=_newness_sort_key)[:probe_limit]
        probe_ids = {row["id"] for row in probe_candidates}

        for row in entries:
            row["latency_ms"] = None
            row["throughput"] = None
            row["uptime"] = None
            row["latency_source"] = "heuristic"

        for row in entries:
            if row["id"] not in probe_ids:
                continue
            metrics = self._fetch_latency_metrics(row["id"])
            row["latency_ms"] = metrics["latency_ms"]
            row["throughput"] = metrics["throughput"]
            row["uptime"] = metrics["uptime"]
            if row["latency_ms"] is not None:
                row["latency_source"] = "openrouter_last_30m"

    def _to_entry(self, row: dict[str, Any]) -> dict[str, Any]:
        model_id = str(row.get("id") or "")
        provider = model_id.split("/", 1)[0] if "/" in model_id else "unknown"
        pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}

        prompt_cost = _as_float(pricing.get("prompt"))
        completion_cost = _as_float(pricing.get("completion"))
        blended_cost = None
        if prompt_cost is not None and completion_cost is not None:
            blended_cost = prompt_cost + completion_cost

        created_dt = _to_datetime(row.get("created"))
        top_provider = row.get("top_provider") if isinstance(row.get("top_provider"), dict) else {}

        return {
            "id": model_id,
            "name": row.get("name"),
            "provider": provider,
            "description": row.get("description"),
            "created_at": created_dt,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "blended_cost": blended_cost,
            "context_length": _as_int(row.get("context_length")),
            "max_completion_tokens": _as_int(top_provider.get("max_completion_tokens")),
            "latency_ms": None,
            "throughput": None,
            "uptime": None,
            "latency_source": None,
        }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
        if parsed < 0:
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_datetime(value: Any) -> datetime | None:
    as_int = _as_int(value)
    if as_int is None or as_int <= 0:
        return None

    # OpenRouter uses unix seconds. Guard for occasional ms payloads.
    if as_int > 10_000_000_000:
        as_int = as_int // 1000
    try:
        return datetime.fromtimestamp(as_int, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _to_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _cost_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    blended = row.get("blended_cost")
    return (
        blended is None,
        blended if blended is not None else float("inf"),
        -int((row.get("created_at") or datetime(1970, 1, 1, tzinfo=UTC)).timestamp()),
        row.get("id") or "",
    )


def _newness_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    created = row.get("created_at") or datetime(1970, 1, 1, tzinfo=UTC)
    blended = row.get("blended_cost")
    return (
        -int(created.timestamp()),
        blended is None,
        blended if blended is not None else float("inf"),
        row.get("id") or "",
    )


def _latency_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    latency = row.get("latency_ms")
    throughput = row.get("throughput")
    blended = row.get("blended_cost")
    return (
        latency is None,
        latency if latency is not None else float("inf"),
        -(throughput if throughput is not None else 0.0),
        blended is None,
        blended if blended is not None else float("inf"),
        row.get("id") or "",
    )
