"""Tests for src/llm/response_parser.py — code block extraction, JSON, error normalization."""

from src.llm.response_parser import (
    extract_code_blocks,
    extract_json_block,
    normalize_error_signature,
)


class TestExtractCodeBlocks:
    def test_single_python_block(self):
        text = """Here's the code:

```python
def hello():
    print("hi")
```

That should work."""
        blocks = extract_code_blocks(text, "python")
        assert len(blocks) == 1
        assert 'def hello():' in blocks[0]

    def test_multiple_blocks(self):
        text = """```python
code1
```

Some text.

```python
code2
```"""
        blocks = extract_code_blocks(text, "python")
        assert len(blocks) == 2
        assert blocks[0] == "code1"
        assert blocks[1] == "code2"

    def test_any_language(self):
        text = """```javascript
const x = 1;
```

```python
x = 1
```"""
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2

    def test_no_language_tag(self):
        text = """```
plain code
```"""
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0] == "plain code"

    def test_no_code_blocks(self):
        text = "Just some text without code blocks"
        blocks = extract_code_blocks(text)
        assert blocks == []

    def test_language_filter_excludes_others(self):
        text = """```javascript
const x = 1;
```"""
        blocks = extract_code_blocks(text, "python")
        assert blocks == []


class TestExtractJsonBlock:
    def test_json_code_block(self):
        text = """Here's the result:

```json
{"name": "test", "value": 42}
```"""
        result = extract_json_block(text)
        assert result == {"name": "test", "value": 42}

    def test_raw_json(self):
        text = '{"status": "ok"}'
        result = extract_json_block(text)
        assert result == {"status": "ok"}

    def test_raw_json_with_whitespace(self):
        text = '  \n  {"status": "ok"}  \n  '
        result = extract_json_block(text)
        assert result == {"status": "ok"}

    def test_invalid_json(self):
        text = "This is not JSON at all"
        result = extract_json_block(text)
        assert result is None

    def test_json_block_priority(self):
        text = """```json
{"from": "block"}
```

{"from": "raw"}"""
        result = extract_json_block(text)
        assert result["from"] == "block"

    def test_nested_json(self):
        text = """```json
{
    "task": {
        "title": "Fix bug",
        "files": ["a.py", "b.py"]
    },
    "approved": true
}
```"""
        result = extract_json_block(text)
        assert result["task"]["title"] == "Fix bug"
        assert result["approved"] is True
        assert len(result["task"]["files"]) == 2


class TestNormalizeErrorSignature:
    def test_strips_unix_paths(self):
        text = "Error in /home/user/project/src/main.py: SyntaxError"
        sig = normalize_error_signature(text)
        assert "/home" not in sig
        assert "<PATH>" in sig
        assert "SyntaxError" in sig

    def test_strips_windows_paths(self):
        text = r"Error in C:\Users\dev\project\main.py: SyntaxError"
        sig = normalize_error_signature(text)
        assert "C:" not in sig
        assert "<PATH>" in sig

    def test_strips_line_numbers(self):
        text = "Error at line 42: unexpected token"
        sig = normalize_error_signature(text)
        assert "42" not in sig
        assert "line <N>" in sig

    def test_strips_colon_line_col(self):
        text = "file.py:123:45: error"
        sig = normalize_error_signature(text)
        assert "123" not in sig
        assert ":<N>:<N>" in sig

    def test_strips_timestamps(self):
        text = "2024-01-15T10:30:45.123Z Error occurred"
        sig = normalize_error_signature(text)
        assert "2024" not in sig
        assert "<TIMESTAMP>" in sig

    def test_strips_memory_addresses(self):
        text = "Object at 0x7fff5b2a3c00 is not callable"
        sig = normalize_error_signature(text)
        assert "0x7fff" not in sig
        assert "<ADDR>" in sig

    def test_collapses_whitespace(self):
        text = "Error:   too   many   spaces"
        sig = normalize_error_signature(text)
        assert "  " not in sig

    def test_same_error_same_signature(self):
        err1 = "Error in /home/user1/project/main.py at line 42: TypeError"
        err2 = "Error in /home/user2/other/main.py at line 99: TypeError"
        assert normalize_error_signature(err1) == normalize_error_signature(err2)

    def test_different_error_different_signature(self):
        err1 = "TypeError: expected str"
        err2 = "ValueError: invalid literal"
        assert normalize_error_signature(err1) != normalize_error_signature(err2)
