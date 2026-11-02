"""Citation parsing and evidence formatting."""

from __future__ import annotations

import re

_CITATION = re.compile(r"\[(\d{1,3})\]")


def cited_indices(answer: str) -> list[int]:
    """Return the distinct 1-based citation numbers in order of first appearance."""
    seen: list[int] = []
    for match in _CITATION.finditer(answer):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def location_label(metadata: dict) -> str:
    source = metadata.get("source", "unknown")
    if "page" in metadata:
        return f"{source}, page {metadata['page']}"
    if "row_start" in metadata:
        return f"{source}, rows {metadata['row_start']}-{metadata['row_end']}"
    return source


def format_document_evidence(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] ({location_label(chunk['metadata'])})\n{chunk['text']}")
    return "\n\n".join(blocks)


def build_citation(index: int, chunk: dict) -> dict:
    meta = chunk["metadata"]
    score = chunk.get("rerank_score")
    if score is None:
        score = chunk.get("rrf_score")
    return {
        "id": index,
        "source": meta.get("source", "unknown"),
        "location": location_label(meta),
        "page": meta.get("page"),
        "chunk_id": chunk["chunk_id"],
        "snippet": chunk["text"][:300],
        "score": score,
    }
