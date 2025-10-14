"""
ingest_stream.py
-----------------
Streams, downloads, and parses research paper PDFs retrieved from arXiv.

Responsibilities:
- Download PDFs using URLs from the metadata file (from fetch_arxiv.py).
- Extract full text content using `pypdf`.
- Chunk long text into overlapping segments suitable for embedding.
- Clean and normalize extracted text (remove headers, references, etc.).
- Output clean, structured text chunks for embedding.

Example:
    $ python src/ingest_stream.py --input ./data/arxiv_results.json --output ./data/chunks.jsonl

Output:
    JSONL file where each line contains:
        {
            "paper_id": "...",
            "title": "...",
            "chunk_id": "...",
            "text": "chunk of text"
        }
"""

from __future__ import annotations

import argparse
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from utils.clean import clean_text


logger = logging.getLogger(__name__)


def pdf_text_from_url(url: str, timeout: int = 90) -> str:
    """
    Download a PDF from a URL and extract its full text.

    Args:
        url (str): Direct link to a PDF file.
        timeout (int): HTTP request timeout in seconds.

    Returns:
        str: Extracted text from all pages of the PDF.
    """
    if not url or not isinstance(url, str):
        raise ValueError("A valid PDF URL must be provided.")

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download PDF from {url}: {exc}") from exc

    buffer = BytesIO()
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            buffer.write(chunk)
    buffer.seek(0)

    try:
        reader = PdfReader(buffer)
    except PdfReadError as exc:
        raise RuntimeError(f"Unable to read PDF from {url}: {exc}") from exc

    pages_text: List[str] = []
    for idx, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive logging for extraction issues
            logger.warning("Failed to extract text from page %s of %s: %s", idx, url, exc)
            page_text = ""
        if page_text:
            pages_text.append(page_text.strip())

    return "\n\n".join(pages_text)


def chunk_text(text: str, source: str, chunk_size: int = 800, chunk_overlap: int = 200) -> list[dict]:
    """
    Split a large document text into overlapping text chunks for embedding.

    Args:
        text (str): The full document text.
        source (str): Identifier or URL of the original document (metadata).
        chunk_size (int): Number of characters (or tokens) per chunk.
        chunk_overlap (int): Overlap size between adjacent chunks.

    Returns:
        list[dict]: Each chunk dict contains:
            {
              "text": str,
              "source": str,
              "chunk_id": int
            }
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or positive.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    normalized = (text or "").strip()
    if not normalized:
        return []

    step = chunk_size - chunk_overlap
    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_id = 0
    length = len(normalized)

    while start < length:
        end = min(start + chunk_size, length)
        segment = normalized[start:end].strip()
        if segment:
            chunks.append({"text": segment, "source": source, "chunk_id": chunk_id})
            chunk_id += 1
        start += step

    return chunks


def stream_and_chunk(papers_meta: list[dict], chunk_size: int = 800, chunk_overlap: int = 200) -> list[dict]:
    """
    Given a list of paper metadata (with PDF URLs), stream each PDF, extract text,
    chunk it, and combine into a flattened list of text chunks.

    Args:
        papers_meta (list[dict]): Metadata of papers (with keys like "pdf_url", "title", etc.).

    Args:
        papers_meta (list[dict]): Metadata of papers (with keys like "pdf_url", "title", etc.).
        chunk_size (int): Number of characters per chunk.
        chunk_overlap (int): Overlap size between adjacent chunks.

    Returns:
        list[dict]: All chunks combined; each chunk dict should have:
            {
              "text": str,
              "source": str,       # e.g. paper URL or title
              "chunk_id": int,
              "title": str,        # optional
              "arxiv_id": str,     # optional
              "updated": str       # optional
            }
    """
    if not papers_meta:
        return []

    all_chunks: List[Dict[str, Any]] = []

    for paper in papers_meta:
        if not isinstance(paper, dict):
            logger.debug("Skipping non-dict paper metadata: %r", paper)
            continue

        pdf_url = paper.get("pdf_url") or paper.get("pdf") or paper.get("url")
        if not pdf_url:
            logger.debug("Missing PDF URL in paper metadata: %s", paper.get("title", "unknown"))
            continue

        source = paper.get("source") or pdf_url or paper.get("title") or paper.get("id") or "unknown"

        try:
            raw_text = pdf_text_from_url(pdf_url)
        except Exception as exc:  # pragma: no cover - defensive guard for downstream errors
            logger.warning("Failed to download or parse PDF %s: %s", pdf_url, exc)
            continue

        if not raw_text:
            logger.debug("Empty text extracted from %s", pdf_url)
            continue

        cleaned_text = raw_text
        try:
            cleaned_candidate = clean_text(raw_text)
            if isinstance(cleaned_candidate, str) and cleaned_candidate:
                cleaned_text = cleaned_candidate
        except Exception as exc:  # pragma: no cover
            logger.warning("Clean step failed for %s: %s", pdf_url, exc)

        try:
            chunk_records = chunk_text(
                cleaned_text,
                source,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Chunking failed for %s: %s", pdf_url, exc)
            continue

        if not chunk_records:
            logger.debug("No chunks produced for %s", pdf_url)
            continue

        for idx, chunk in enumerate(chunk_records):
            if not isinstance(chunk, dict):
                logger.debug("Skipping non-dict chunk from %s: %r", pdf_url, chunk)
                continue

            text = chunk.get("text", "").strip()
            if not text:
                continue

            chunk_id = chunk.get("chunk_id")
            if chunk_id is None:
                chunk_id = idx

            chunk_payload: Dict[str, Any] = {
                "text": text,
                "source": chunk.get("source") or source,
                "chunk_id": chunk_id,
            }

            if paper.get("title"):
                chunk_payload["title"] = paper["title"]

            arxiv_identifier = paper.get("arxiv_id") or paper.get("id")
            if arxiv_identifier:
                chunk_payload["arxiv_id"] = arxiv_identifier

            if paper.get("updated"):
                chunk_payload["updated"] = paper["updated"]

            all_chunks.append(chunk_payload)

    return all_chunks


def _read_metadata(input_path: Path, limit: int | None = None) -> list[dict]:
    """Load paper metadata from JSON or JSONL file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {input_path}")

    data: List[dict] = []
    suffix = input_path.suffix.lower()

    if suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON line: %s", exc)
                    continue
                if isinstance(record, dict):
                    data.append(record)
    elif suffix == ".json":
        with input_path.open("r", encoding="utf-8") as handle:
            content = json.load(handle)
            if isinstance(content, list):
                data.extend(entry for entry in content if isinstance(entry, dict))
            elif isinstance(content, dict):
                data.append(content)
    else:
        raise ValueError(f"Unsupported metadata file format: {input_path.suffix}")

    if limit is not None and limit >= 0:
        data = data[:limit]
    return data


def _write_chunks(chunks: List[dict], output_path: Path) -> None:
    """Persist generated chunks as JSONL."""
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream PDFs defined in metadata and produce cleaned text chunks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to arXiv metadata (JSON or JSONL).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of papers to process.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Character length of each chunk (default: %(default)s).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between consecutive chunks (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chunks.jsonl"),
        help="Destination JSONL file for text chunks (default: %(default)s).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for streaming PDFs and producing text chunks."""
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    try:
        metadata = _read_metadata(args.input, limit=args.limit)
    except Exception as exc:
        logger.error("Failed to read metadata: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Processing %s papers from %s", len(metadata), args.input)

    chunks = stream_and_chunk(
        metadata,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    logger.info("Writing %s chunks to %s", len(chunks), args.output)
    try:
        _write_chunks(chunks, args.output)
    except Exception as exc:
        logger.error("Failed to write chunks: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
