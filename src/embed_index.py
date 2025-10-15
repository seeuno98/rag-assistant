"""
embed_index.py
----------------
Generates semantic embeddings for text chunks and builds a vector index.

Responsibilities:
- Load text chunks from ingest_stream.py output.
- Use `sentence-transformers` or another embedding model to generate vector representations.
- Store embeddings and metadata in FAISS (or Pinecone) index.
- Save both the index and metadata for later retrieval.

Example:
    $ python src/embed_index.py --input ./data/chunks.jsonl --output ./indexes/faiss_index

Output:
    ./indexes/faiss_index
    ./indexes/metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


DEFAULT_INDEX_PATH = Path("indexes/faiss.index")
DEFAULT_METADATA_PATH = Path("indexes/docs_meta.json")
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _validate_chunks(chunks: Iterable[dict]) -> List[dict]:
    """Filter out invalid chunk entries and ensure text is present."""
    valid_chunks: List[dict] = []
    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            logger.debug("Skipping non-dict chunk at position %s", idx)
            continue
        text = (chunk.get("text") or "").strip()
        if not text:
            logger.debug("Skipping empty-text chunk at position %s", idx)
            continue
        valid_chunks.append({**chunk, "text": text})
    return valid_chunks


def build_faiss_index(chunks: list[dict], embed_model_name: str) -> tuple:
    """
    Build a FAISS (inner-product) index from text chunks.

    Args:
        chunks (list[dict]): List of chunk dicts with "text" keys.
        embed_model_name (str): Model identifier for sentence-transformers.

    Returns:
        (faiss.Index, list[dict]): A tuple of (index object, metadata list) where metadata list
        aligns with embeddings (same order).
    """
    valid_chunks = _validate_chunks(chunks)
    if not valid_chunks:
        raise ValueError("No valid chunks provided to build the index.")

    model = SentenceTransformer(embed_model_name)
    texts = [chunk["text"] for chunk in valid_chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise RuntimeError("Embeddings array must be 2-dimensional.")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, valid_chunks


def load_faiss_index(index_path: str, meta_path: str, embed_model_name: str):
    """
    Load an existing FAISS index and metadata file and return loaded model and metadata.

    Args:
        index_path (str): Path to saved FAISS index file.
        meta_path (str): Path to saved metadata JSON.
        embed_model_name (str): Embedding model name to re-init the embedding model.

    Returns:
        (faiss.Index, list[dict], SentenceTransformer): loaded index, metadata, embedding model.
    """
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not Path(meta_path).exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    index = faiss.read_index(str(index_path))
    with Path(meta_path).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, list):
        raise ValueError("Metadata file must contain a list of chunk dictionaries.")

    model = SentenceTransformer(embed_model_name)
    return index, metadata, model


def search_index(index, model, metadata: list[dict], query: str, k: int = 5) -> list[dict]:
    """
    Query the vector index for the most relevant chunks given a user query.

    Args:
        index (faiss.Index): The FAISS index built earlier.
        model (SentenceTransformer): The embedding model instance.
        metadata (list[dict]): Chunk metadata aligned with embeddings.
        query (str): Text query from the user.
        k (int): Number of top matches to return.

    Returns:
        list[dict]: Top k matches. Each dict includes:
            {
              "text": str,
              "source": str,
              "chunk_id": int,
              "score": float,
              "idx": int
            }
    """
    if index is None or metadata is None or model is None:
        raise ValueError("Index, metadata, and model must be provided.")
    if not query:
        return []

    normalized_query = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    if normalized_query.ndim == 1:
        normalized_query = normalized_query.reshape(1, -1)

    k = max(1, k)
    scores, idxs = index.search(normalized_query, k)
    if scores.size == 0:
        return []

    matches: List[dict] = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0])):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        snippet = (meta.get("text") or "")[:200].replace("\n", " ")
        matches.append(
            {
                "text": meta.get("text"),
                "source": meta.get("source"),
                "chunk_id": meta.get("chunk_id"),
                "score": float(score),
                "idx": int(idx),
                "snippet": snippet,
                "rank": rank,
            }
        )
    return matches


def _load_chunks_from_file(path: Path) -> List[dict]:
    """Load chunk metadata from JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    chunks: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %s: %s", line_no, exc)
                continue
            if isinstance(record, dict):
                chunks.append(record)
    return chunks


def _write_metadata(metadata: List[dict], path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query a FAISS index over text chunks.")
    parser.add_argument("--input", type=Path, help="Path to chunks JSONL file.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Sentence Transformer model.")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH, help="Path to FAISS index file.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH, help="Path to metadata JSON.")
    parser.add_argument("--build", action="store_true", help="Build (and save) the FAISS index from chunks.")
    parser.add_argument("--search", type=str, help="Query string to search against the index.")
    parser.add_argument("--k", type=int, default=5, help="Number of top matches to return when searching.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.build:
        if not args.input:
            raise SystemExit("--build requires --input pointing to a chunks JSONL file.")
        logger.info("Loading chunks from %s", args.input)
        chunks = _load_chunks_from_file(args.input)
        logger.info("Loaded %s chunks", len(chunks))
        index, metadata = build_faiss_index(chunks, args.model)
        logger.info("Saving FAISS index to %s", args.index_path)
        if not args.index_path.parent.exists():
            args.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(args.index_path))
        logger.info("Saving metadata to %s", args.metadata_path)
        _write_metadata(metadata, args.metadata_path)
        print(f"Wrote index with {index.ntotal} vectors to {args.index_path}")  # noqa: T201
        print(f"Wrote metadata with {len(metadata)} records to {args.metadata_path}")  # noqa: T201

    if args.search:
        index, metadata, model = load_faiss_index(str(args.index_path), str(args.metadata_path), args.model)
        logger.info("Searching for query: %s", args.search)
        matches = search_index(index, model, metadata, args.search, k=args.k)
        if not matches:
            print("No matches found.")  # noqa: T201
            return
        for match in matches:
            score = match["score"]
            source = match.get("source") or "unknown"
            snippet = match.get("snippet") or ""
            print(f"[{score:.3f}] {source} :: {snippet}")  # noqa: T201
    elif not args.build:
        print("No action taken. Use --build and/or --search to perform work.")  # noqa: T201


if __name__ == "__main__":
    main()
