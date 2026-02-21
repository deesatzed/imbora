"""Schema + model validation helpers for APC/1.0 packets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.core.models import AgentPacket


class PacketValidationError(ValueError):
    """Raised when packet validation fails."""


def _default_schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "contracts" / "agent_packet_v1.schema.json"


def _validate_with_jsonschema(packet: dict[str, Any], schema_path: Path) -> None:
    """Run JSON Schema validation when jsonschema is available.

    This is optional to avoid hard runtime dependency in environments where
    only the Pydantic models are installed.
    """
    try:
        import jsonschema  # type: ignore[import-not-found]
    except Exception:
        return

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(instance=packet, schema=schema)
    except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
        path = ".".join(str(part) for part in exc.path)
        raise PacketValidationError(f"JSON schema validation failed at '{path}': {exc.message}") from exc


def validate_agent_packet(
    packet_data: dict[str, Any],
    *,
    schema_path: Path | None = None,
) -> AgentPacket:
    """Validate a packet against APC/1.0 schema and Pydantic contracts."""
    resolved_schema = schema_path or _default_schema_path()
    if resolved_schema.exists():
        _validate_with_jsonschema(packet_data, resolved_schema)

    try:
        return AgentPacket.model_validate(packet_data)
    except ValidationError as exc:
        raise PacketValidationError(f"Packet model validation failed: {exc}") from exc

