"""
utils/clean.py
---------------
Utility functions for cleaning and normalizing text extracted from PDFs.

Responsibilities:
- Remove unwanted text patterns (page numbers, citations, figure captions).
- Normalize whitespace, line breaks, and encoding.
- Optionally apply heuristics to split sections (abstract, introduction, etc.).
- Provide helper functions used across the ingestion and embedding pipeline.

Example:
    clean_text(raw_text: str) -> str
"""

from __future__ import annotations

import re
import sys


# Precompiled regular expressions for filtering artifacts.
PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*[\-\u2013\u2014]{1,3}\s*\d+\s*[\-\u2013\u2014]{1,3}\s*$"),
    re.compile(r"^\s*\(\s*\d+\s*\)\s*$"),
]
HEADER_FOOTER_PATTERNS = [
    re.compile(r"^\s*(references|acknowledgements?|footnotes?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(page|figure|table)\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*arxiv.*$", re.IGNORECASE),
    re.compile(r"^\s*copyright.*$", re.IGNORECASE),
]
CITATION_LINE_PATTERN = re.compile(r"^\s*\[\d+\]\s+")
WHITESPACE_RE = re.compile(r"[ \t]+")
BLANK_LINE_RE = re.compile(r"\n{2,}")


def clean_text(raw: str) -> str:
    """
    Clean and normalize raw text extracted from PDFs.

    Tasks may include:
        - Removing headers, footers, page numbers
        - Collapsing repeated whitespace
        - Removing non-ASCII artifacts
        - Normalizing punctuation or unwanted tokens

    Args:
        raw (str): Raw extracted text.

    Returns:
        str: Cleaned, simplified text suitable for embedding.
    """
    if not raw:
        return ""

    text = raw
    if not isinstance(text, str):
        text = str(text or "")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", " ")

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        if any(pattern.match(stripped) for pattern in PAGE_NUMBER_PATTERNS):
            continue
        if any(pattern.match(stripped) for pattern in HEADER_FOOTER_PATTERNS):
            continue
        if CITATION_LINE_PATTERN.match(stripped):
            continue

        normalized = WHITESPACE_RE.sub(" ", stripped)
        cleaned_lines.append(normalized)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = BLANK_LINE_RE.sub("\n\n", cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text


def _demo() -> None:
    """Read stdin, clean text, and print result (manual smoke-test helper)."""
    raw_input = sys.stdin.read()
    cleaned = clean_text(raw_input)
    print(cleaned)


if __name__ == "__main__":
    _demo()
