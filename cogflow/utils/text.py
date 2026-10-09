"""
CogFlow Text Utilities

Helper functions for text processing and manipulation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union


def truncate_text(
    text: str,
    max_length: int = 1000,
    suffix: str = "...",
) -> str:
    """Truncate text to a maximum length.
    
    Args:
        text: Input text
        max_length: Maximum characters
        suffix: Suffix to add when truncating
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def clean_text(
    text: str,
    remove_extra_whitespace: bool = True,
    remove_special_chars: bool = False,
    lowercase: bool = False,
) -> str:
    """Clean and normalize text.
    
    Args:
        text: Input text
        remove_extra_whitespace: Collapse multiple spaces
        remove_special_chars: Remove non-alphanumeric chars
        lowercase: Convert to lowercase
    
    Returns:
        Cleaned text
    """
    result = text
    
    if lowercase:
        result = result.lower()
    
    if remove_special_chars:
        result = re.sub(r'[^a-zA-Z0-9\s]', '', result)
    
    if remove_extra_whitespace:
        result = re.sub(r'\s+', ' ', result).strip()
    
    return result


def extract_json(
    text: str,
    return_first: bool = True,
) -> Union[Dict, List[Dict], None]:
    """Extract JSON from text.
    
    Finds and parses JSON objects or arrays embedded in text.
    
    Args:
        text: Text containing JSON
        return_first: Return only first JSON found
    
    Returns:
        Parsed JSON or None if not found
    """
    # Find JSON patterns
    json_objects = []
    
    # Try to find JSON objects
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    bracket_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
    
    for pattern in [brace_pattern, bracket_pattern]:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                json_objects.append(parsed)
            except json.JSONDecodeError:
                continue
    
    if not json_objects:
        return None
    
    if return_first:
        return json_objects[0]
    
    return json_objects


def count_tokens_approx(
    text: str,
    chars_per_token: float = 4.0,
) -> int:
    """Approximate token count.
    
    Uses a simple character-based estimation.
    For accurate counts, use the LLM provider's count_tokens.
    
    Args:
        text: Input text
        chars_per_token: Average characters per token
    
    Returns:
        Estimated token count
    """
    return int(len(text) / chars_per_token)


def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    separator: str = "\n",
) -> List[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks
        separator: Preferred split point
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Find separator near end
        if end < len(text):
            sep_pos = text.rfind(separator, start, end)
            if sep_pos > start:
                end = sep_pos + len(separator)
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def format_messages(
    messages: List[Dict[str, str]],
    style: str = "openai",
) -> str:
    """Format messages list as string.
    
    Args:
        messages: List of message dicts
        style: Output style ('openai', 'anthropic', 'simple')
    
    Returns:
        Formatted string
    """
    lines = []
    
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if style == "simple":
            lines.append(f"{role}: {content}")
        elif style == "anthropic":
            if role == "user":
                lines.append(f"\n\nHuman: {content}")
            elif role == "assistant":
                lines.append(f"\n\nAssistant: {content}")
            else:
                lines.append(f"\n\n{role.title()}: {content}")
        else:  # openai
            lines.append(f"<|{role}|>\n{content}")
    
    return "\n".join(lines)


__all__ = [
    "truncate_text",
    "clean_text",
    "extract_json",
    "count_tokens_approx",
    "split_text",
    "format_messages",
]
