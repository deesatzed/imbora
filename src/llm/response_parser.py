"""Response parsing utilities for LLM output.

Extracts code blocks, JSON, and structured data from raw LLM responses.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


def extract_code_blocks(text: str, language: Optional[str] = None) -> list[str]:
    """Extract fenced code blocks from LLM output.

    Args:
        text: Raw LLM response.
        language: If specified, only return blocks with this language tag.

    Returns:
        List of code block contents (without fences).
    """
    if language:
        pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
    else:
        pattern = r"```(?:\w+)?\s*\n(.*?)```"

    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]


def extract_json_block(text: str) -> Optional[dict[str, Any]]:
    """Extract and parse the first JSON block from LLM output."""
    blocks = extract_code_blocks(text, "json")
    if blocks:
        try:
            return json.loads(blocks[0])
        except json.JSONDecodeError:
            pass

    # Try parsing the whole response as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def normalize_error_signature(error_text: str) -> str:
    """Normalize an error message for deduplication.

    Strips file paths, line numbers, and timestamps so that
    the same logical error produces the same signature.
    """
    sig = error_text.strip()

    # Remove timestamps (before line numbers, since :HH:MM:SS overlaps with :N:N)
    sig = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*\w*', '<TIMESTAMP>', sig)

    # Remove file paths (Unix and Windows)
    sig = re.sub(r'(/[\w./-]+|\w:\\[\w.\\-]+)', '<PATH>', sig)

    # Remove line numbers
    sig = re.sub(r'line \d+', 'line <N>', sig, flags=re.IGNORECASE)
    sig = re.sub(r':\d+:\d+', ':<N>:<N>', sig)

    # Remove memory addresses
    sig = re.sub(r'0x[0-9a-fA-F]+', '<ADDR>', sig)

    # Collapse whitespace
    sig = re.sub(r'\s+', ' ', sig).strip()

    return sig
