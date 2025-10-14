"""
fetch_arxiv.py
----------------
Fetches the latest LLM-related research papers from arXiv using its public API.

Responsibilities:
- Query arXiv for new papers related to Large Language Models (LLMs).
- Filter and retrieve relevant metadata (title, authors, abstract, PDF URL, published date).
- Save metadata to local cache (e.g., JSON) or pass directly to the ingestion pipeline.

Example:
    $ python src/fetch_arxiv.py --query "large language models" --max_results 20

Output:
    arxiv_results.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List
from xml.etree import ElementTree as ET

import requests


ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
XML_NAMESPACES = {"atom": ATOM_NS, "arxiv": ARXIV_NS}


def _parse_atom_datetime(value: str) -> datetime | None:
    """Parse arXiv Atom timestamp into a timezone-aware datetime."""
    if not value:
        return None
    value = value.strip()
    # arXiv uses Zulu suffix; convert to ISO 8601 the stdlib understands.
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def fetch_llm_papers(query: str, days_back: int = 3, max_results: int = 20) -> list[dict]:
    """
    Query the arXiv API for the most recent LLM-related papers.

    Args:
        query (str): Search keywords (e.g. "large language model", "RAG").
        days_back (int): Number of past days to consider for “new” papers.
        max_results (int): Maximum number of papers to return.

    Returns:
        list[dict]: A list of paper metadata dicts, each containing:
            {
              "id": str,
              "title": str,
              "authors": list[str],
              "pdf_url": str,
              "updated": str,  # ISO date
              "summary": str,
              "primary_category": str,
            }
    """
    if max_results <= 0:
        return []

    search_query = f"all:{query.strip()}" if query else "all:LLM"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
    }

    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch arXiv papers: {exc}") from exc

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        raise RuntimeError("Unable to parse arXiv API response") from exc

    threshold = datetime.now(timezone.utc) - timedelta(days=max(days_back, 0))
    papers: List[dict] = []

    for entry in root.findall("atom:entry", XML_NAMESPACES):
        updated_text = entry.findtext("atom:updated", default="", namespaces=XML_NAMESPACES).strip()
        updated_dt = _parse_atom_datetime(updated_text)
        if updated_dt and updated_dt < threshold:
            continue

        authors = [
            author.findtext("atom:name", default="", namespaces=XML_NAMESPACES).strip()
            for author in entry.findall("atom:author", XML_NAMESPACES)
        ]
        authors = [name for name in authors if name]

        pdf_url = ""
        for link in entry.findall("atom:link", XML_NAMESPACES):
            link_type = link.attrib.get("type")
            title = link.attrib.get("title", "")
            if link_type == "application/pdf" or title.lower() == "pdf":
                pdf_url = link.attrib.get("href", "")
                break

        primary_category = ""
        primary_elem = entry.find("arxiv:primary_category", XML_NAMESPACES)
        if primary_elem is not None:
            primary_category = primary_elem.attrib.get("term", "")
        if not primary_category:
            category_elem = entry.find("atom:category", XML_NAMESPACES)
            if category_elem is not None:
                primary_category = category_elem.attrib.get("term", "")

        paper = {
            "id": entry.findtext("atom:id", default="", namespaces=XML_NAMESPACES).strip(),
            "title": entry.findtext("atom:title", default="", namespaces=XML_NAMESPACES).strip(),
            "authors": authors,
            "pdf_url": pdf_url,
            "updated": updated_text,
            "summary": entry.findtext("atom:summary", default="", namespaces=XML_NAMESPACES).strip(),
            "primary_category": primary_category,
        }
        papers.append(paper)

    if days_back > 0:
        papers = [
            paper
            for paper in papers
            if not paper["updated"]
            or (_parse_atom_datetime(paper["updated"]) or threshold) >= threshold
        ]

    return papers


def _parse_args() -> argparse.Namespace:
    """Configure and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch recent LLM-related arXiv papers and cache their metadata."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="large language model",
        help="arXiv search query (default: %(default)s)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=3,
        help="Only include papers updated within the last N days (default: %(default)s)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of papers to fetch (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("arxiv_results.json"),
        help="Destination JSON file for metadata (default: %(default)s)",
    )
    return parser.parse_args()


def _write_results(papers: List[dict], output_path: Path) -> None:
    """Persist fetched metadata to disk."""
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(papers, fp, indent=2)


def main() -> None:
    """CLI entrypoint for fetching and saving arXiv metadata."""
    args = _parse_args()
    papers = fetch_llm_papers(args.query, days_back=args.days_back, max_results=args.max_results)
    _write_results(papers, args.output)
    print(f"Fetched {len(papers)} papers → {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
