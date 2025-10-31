"""
rag_answer.py
---------------
Retrieves relevant paper sections and generates context-aware answers or summaries using an LLM.

Responsibilities:
- Load FAISS (or Pinecone) vector index.
- Accept a natural-language query from the user (e.g., "What are recent methods for long-context training?").
- Retrieve the top-K most relevant text chunks.
- Construct a context-augmented prompt and send it to the LLM (OpenAI GPT or Hugging Face model).
- Return the generated response (answer or summary).

Example:
    $ python src/rag_answer.py --query "What are key contributions of recent LLM papers?"

Output:
    Contextualized answer or summary printed to stdout or saved in reports.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
import requests

from embed_index import (
    DEFAULT_INDEX_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_NAME,
    load_faiss_index,
    search_index,
)


logger = logging.getLogger(__name__)

load_dotenv()


# Configure your hosted Hugging Face model here (edit directly in this file or via env vars)
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "google/flan-t5-base").strip()
HF_API_BASE = (os.getenv("HF_API_BASE") or "").strip() or None
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "90"))
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.2"))
CONTEXT_CHAR_LIMIT = int(os.getenv("CONTEXT_CHAR_LIMIT", "1200"))
MAX_CONTEXT_SECTIONS = int(os.getenv("MAX_CONTEXT_SECTIONS", "5"))

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 5
OPENAI_SYSTEM_PROMPT = (
    "You summarize research papers from retrieved context.\n"
    "Strict rules:\n"
    "- Only summarize the retrieved papers.\n"
    f"- Summarize at most {DEFAULT_TOP_K} papers.\n"
    "- Exclude irrelevant content.\n"
    "- No external knowledge.\n"
    "- One bullet per paper.\n"
    "- Each bullet MUST be an HTML <li><b><a href=\"URL\" target=\"_blank\" rel=\"noopener noreferrer\">Paper Title</a></b>: 1–2 sentence summary grounded in context.</li>\n"
    "- Do not invent details not in context.\n"
    "- No duplicates. Treat items as the same paper if ANY of these match:\n"
    "  (a) same arXiv id root (e.g., 2501.12345v1 and 2501.12345v3),\n"
    "  (b) same DOI,\n"
    "  (c) highly similar titles (case-insensitive, ignore punctuation/parenthetical suffixes).\n"
    "- If duplicates appear in context, summarize the paper only once. Prefer the newest version link (latest arXiv version or publisher DOI) and the clearest title.\n"
    "- If deduplication reduces the number of bullets below the requested cap, DO NOT invent extra items; output fewer bullets."
)


def _clamp_text(text: str, max_length: int) -> str:
    """Collapse whitespace and trim text to a maximum character length."""
    normalized = " ".join((text or "").split())
    if len(normalized) > max_length:
        cutoff = max(0, max_length - 3)
        normalized = normalized[:cutoff].rstrip() + "..."
    return normalized


def format_context(hits: list[dict], max_sections: int, char_limit: int) -> str:
    """
    Format the retrieved chunks into a prompt-ready context string.

    Args:
        hits (list[dict]): Retrieved chunk dictionaries with text, source, etc.
        max_sections (int): Maximum number of chunks to include (<= len(hits), <= 0 means all).
        char_limit (int): Maximum number of characters per chunk excerpt.

    Returns:
        str: A concatenated string of context passages with identifiers.
    """
    if not hits:
        return ""

    if max_sections > 0:
        hits = hits[:max_sections]

    blocks: List[str] = []
    for idx, hit in enumerate(hits, start=1):
        heading = hit.get("title") or hit.get("source") or f"Chunk {hit.get('chunk_id', idx)}"
        raw_text = hit.get("text") or hit.get("snippet") or ""
        excerpt = _clamp_text(raw_text, char_limit)
        blocks.append(f"[{idx}] {heading}\n{excerpt}")
    return "\n\n".join(blocks)


def _call_openai(context: str, query: str, model_id: str, max_papers: int) -> str:
    """Call OpenAI Chat Completions API and return the generated answer."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("openai package is required to use the OpenAI provider.") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set to use the OpenAI provider.")

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context (ordered as retrieved):\n{context}\n\n"
                    f"Task: {query}\n\n"
                    f"Summarize up to {max_papers} papers as grounded bullet points. Each bullet must be formatted exactly as '<li><b><a href=\"URL\" target=\"_blank\" rel=\"noopener noreferrer\">Paper Title</a></b>: 1–2 sentence summary grounded in the retrieved context</li>' using only retrieved details.\n"
                    "Deduplicate strictly by arXiv id root / DOI / near-identical titles; include each paper at most once using its most canonical, newest link."
                ),
            },
        ],
        max_completion_tokens=800,
    )
    if not completion or not completion.choices:
        print("⚠️ Model returned an empty response. Try a larger model or reduce context size.")
        return ""
    message = completion.choices[0].message
    content = getattr(message, "content", None)
    if not content or not content.strip():
        print("⚠️ Model returned an empty response. Try a larger model or reduce context size.")
        return ""
    return content.strip()


def _call_huggingface(prompt: str, token: str) -> str:
    """Call the Hugging Face Inference API (REST) and return the generated answer."""
    if not token:
        raise RuntimeError("HF_TOKEN must be set to use the Hugging Face provider.")

    url = HF_API_BASE.rstrip("/") if HF_API_BASE else f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
    if not HF_API_BASE and not HF_MODEL_ID:
        raise RuntimeError("HF_MODEL_ID must be configured when using the Hugging Face provider.")

    headers = {
        "Authorization": f"Bearer {token.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": HF_MAX_NEW_TOKENS,
            "temperature": HF_TEMPERATURE,
        },
        "options": {"wait_for_model": True, "use_cache": True},
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=HF_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Hugging Face network error: {exc}") from exc

    if response.status_code == 404 and not HF_API_BASE:
        raise RuntimeError(
            f"Hugging Face model '{HF_MODEL_ID}' was not found. Confirm the model ID and your access rights."
        )
    if response.status_code == 404 and HF_API_BASE:
        raise RuntimeError(
            "The Hugging Face endpoint URL returned 404. Verify HF_API_BASE points to your inference endpoint."
        )
    if response.status_code == 403:
        raise RuntimeError(
            f"Access to Hugging Face model '{HF_MODEL_ID or HF_API_BASE}' is forbidden for this token. "
            "Visit the model page and ensure you've accepted the terms or use a different account/token."
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        details: str
        try:
            details = response.json()
        except ValueError:
            details = response.text
        raise RuntimeError(
            f"Hugging Face Inference API error (status {response.status_code}): {details}"
        ) from exc

    data = response.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Hugging Face Inference API error: {data['error']}")

    texts: List[str] = []
    if isinstance(data, list):
        for item in data:
            text = (item or {}).get("generated_text") or (item or {}).get("summary_text")
            if isinstance(text, str):
                texts.append(text.strip())
    elif isinstance(data, dict):
        text = data.get("generated_text") or data.get("summary_text")
        if isinstance(text, str):
            texts.append(text.strip())

    if not texts:
        raise RuntimeError(f"Unexpected response format from Hugging Face Inference API: {data!r}")

    return "\n".join(texts).strip()


def answer_question(
    question: str,
    use_openai: bool = False,
    *,
    k: int = DEFAULT_TOP_K,
    provider: Optional[str] = None,
    index_path: Path = DEFAULT_INDEX_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    embed_model_name: str = DEFAULT_MODEL_NAME,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    include_context: bool = False,
) -> str | tuple[str, str]:
    """
    Given a textual question, retrieve relevant context chunks and invoke the LLM to generate an answer.

    Args:
        question (str): The user’s query.
        use_openai (bool): Whether to prefer the OpenAI API (if available).
        k (int): Number of top retrieval results to include.
        provider (Optional[str]): Force a provider ("openai" or "hf"); overrides `use_openai`.
        index_path (Path): Location of the FAISS index file.
        metadata_path (Path): Location of the metadata JSON file.
        embed_model_name (str): Embedding model used when the index was built.
        openai_model (str): Chat completion model for OpenAI provider.
        include_context (bool): When True, return a tuple of (answer, retrieved_context_text).

    Returns:
        str | tuple[str, str]: Answer text, optionally paired with the retrieval context string.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    question = question.strip()
    provider_choice = provider.lower() if provider else None
    context: str = ""

    def _finish(value: str) -> str | tuple[str, str]:
        return (value, context) if include_context else value

    try:
        index, metadata, embed_model = load_faiss_index(
            str(index_path),
            str(metadata_path),
            embed_model_name,
        )
    except FileNotFoundError as exc:
        return _finish(
            f"Vector index is missing ({exc}). Please build the index first "
            "with `python src/embed_index.py --build --input data/chunks.jsonl`."
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Failed to load FAISS index: %s", exc)
        return _finish(
            "Sorry, I ran into an unexpected error while loading the vector index. Please try rebuilding it."
        )

    try:
        hits = search_index(index, embed_model, metadata, question, k=k)
    except Exception as exc:  # pragma: no cover - unexpected retrieval failures
        logger.error("Search failed: %s", exc)
        return _finish("Sorry, something went wrong while searching the index. Please rebuild and try again.")

    if not hits:
        return _finish(
            "Sorry, I couldn't find any matching context for that question. "
            "Try re-building the embeddings or fetching newer papers."
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Resolve provider preference.
    resolved_provider: Optional[str] = None
    if provider_choice == "openai":
        if not openai_key:
            return _finish("OpenAI provider selected, but OPENAI_API_KEY is not set.")
        resolved_provider = "openai"
    elif provider_choice == "hf":
        if not hf_token:
            return _finish("Hugging Face provider selected, but HF_TOKEN is not set.")
        resolved_provider = "hf"
    else:
        if openai_key:
            resolved_provider = "openai"
        elif hf_token:
            resolved_provider = "hf"
        else:
            return _finish(
                "No LLM provider configured. Set OPENAI_API_KEY for OpenAI or HF_TOKEN for Hugging Face "
                "before running the assistant."
            )

    docs_sorted = sorted(hits, key=lambda h: h.get("score", 0.0), reverse=True)
    if not docs_sorted:
        return _finish("No relevant research found in the last 24 hours.")

    def _dedupe_key(doc: dict) -> str:
        arxiv_id = (doc.get("arxiv_id") or doc.get("paper_id") or "").strip().lower()
        if arxiv_id:
            arxiv_id = arxiv_id.replace("arxiv:", "")
            return f"arxiv:{arxiv_id.split('v')[0]}"

        source = (doc.get("source") or doc.get("id") or doc.get("paper_url") or "").strip().lower()
        match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", source)
        if match:
            identifier = match.group(1).replace("arxiv:", "")
            return f"arxiv:{identifier.split('v')[0]}"

        doi = (doc.get("doi") or doc.get("paper_doi") or "").strip().lower()
        if doi:
            return f"doi:{doi}"

        title = (doc.get("title") or "").strip().lower()
        title = re.sub(r"[\s\-_:;,.()\[\]{}]+", " ", title)
        return f"title:{title}"

    unique_docs: list[dict] = []
    seen_keys: set[str] = set()
    for doc in docs_sorted:
        key = _dedupe_key(doc)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_docs.append(doc)

    if not unique_docs:
        return _finish("No relevant research found in the last 24 hours.")

    summary_docs = unique_docs[:DEFAULT_TOP_K]
    if k is None or k <= 0:
        context_docs = summary_docs
    else:
        context_docs = unique_docs[: min(k, len(unique_docs))]

    max_summary = len(summary_docs)
    if max_summary <= 0:
        return _finish("No relevant research found in the last 24 hours.")

    retrieved_sections: list[str] = []
    for idx, doc in enumerate(context_docs, 1):
        title = doc.get("title") or doc.get("source") or f"Document {idx}"
        url = doc.get("source") or ""
        snippet = _clamp_text(doc.get("text") or "", CONTEXT_CHAR_LIMIT)
        section_lines = [f"{idx}. Title: {title}"]
        if url:
            section_lines.append(f"   URL: {url}")
        section_lines.append(f"   Snippet: {snippet}")
        retrieved_sections.append("\n".join(section_lines))

    context = "Retrieved papers:\n" + "\n".join(retrieved_sections)

    hf_prompt = (
        "You are a research summarization assistant. You MUST summarize ONLY the information found in the retrieved context below.\n"
        "STRICT RULES:\n"
        "- Do NOT use external knowledge.\n"
        "- Do NOT hallucinate missing details.\n"
        "- If context does not support a claim, say so.\n"
        "- Cover each retrieved document at least once.\n"
        "- Cite content using simple markers like [1], [2], [3] based on order retrieved.\n"
        "- If multiple topics appear, summarize each separately.\n"
        "- If context is irrelevant to the query, say “The retrieved documents do not contain enough information.”\n\n"
        "Context (ordered as retrieved):\n"
        f"{context}\n\n"
        "Task:\n"
        f"{question}\n\n"
        f"Summarize up to {max_summary} papers as grounded bullet points. Each bullet must be formatted exactly as '<li><b><a href=\"URL\" target=\"_blank\" rel=\"noopener noreferrer\">Paper Title</a></b>: 1–2 sentence summary grounded in the retrieved context</li>'."
    )

    try:
        if resolved_provider == "openai":
            answer = _call_openai(context, question, openai_model, max_summary)
        else:
            answer = _call_huggingface(hf_prompt, hf_token or "")
    except requests.RequestException as exc:
        logger.error("Provider network request failed: %s", exc)
        return _finish(
            "I hit a network issue while contacting the language model provider. "
            "Please check your connection and try again."
        )
    except Exception as exc:  # pragma: no cover - catch-all for provider failures
        logger.exception("Provider call failed")
        return _finish("Sorry, the language model provider returned an error. Please try again later.")

    answer = answer.strip() or "The language model returned an empty response."
    if resolved_provider == "hf":
        lowered = answer.lower()
        if (
            "you are a helpful research assistant" in lowered
            or "using only the information in the context" in lowered
            or "weekly newsquiz" in lowered
        ):
            answer = "I do not know."
    return _finish(answer)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions against the RAG assistant index.")
    parser.add_argument("--query", type=str, required=True, help="User query to answer.")
    parser.add_argument("--provider", choices=["openai", "hf"], help="Force a specific LLM provider.")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Number of results to retrieve.")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH, help="Path to FAISS index file.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH, help="Path to metadata JSON file.")
    parser.add_argument(
        "--embed-model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Embedding model name used when building/loading the index.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI chat model to use (default: %(default)s).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    response = answer_question(
        args.query,
        use_openai=(args.provider == "openai"),
        k=args.k,
        provider=args.provider,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        embed_model_name=args.embed_model,
        openai_model=args.model,
        include_context=False,
    )
    if isinstance(response, tuple):
        response_text, _ = response
    else:
        response_text = response
    print(response_text)  # noqa: T201


if __name__ == "__main__":
    main()
