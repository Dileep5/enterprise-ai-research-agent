"""Thin wrapper around a persistent ChromaDB collection.

Embeddings are always computed by our own ``Embedder`` and passed explicitly, so
Chroma never downloads its default embedding model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.errors import VectorStoreError
from app.ingestion.chunking import Chunk

logger = logging.getLogger(__name__)


@dataclass
class DenseHit:
    chunk: Chunk
    similarity: float  # cosine similarity in [-1, 1]


class ChromaVectorStore:
    def __init__(self, path: Path | None, collection: str):
        try:
            if path is None:
                self._client = chromadb.EphemeralClient()
            else:
                path.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(path))
            self._collection = self._client.get_or_create_collection(
                name=collection,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not open vector store: {exc}") from exc
        # Incremented on every write so dependent indexes (BM25) know to rebuild.
        self.version = 0

    @staticmethod
    def _clean_metadata(metadata: dict) -> dict:
        # Chroma only accepts str/int/float/bool values (no None).
        return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunk and embedding counts differ.")
        try:
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[self._clean_metadata(c.metadata) for c in chunks],
                embeddings=embeddings,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to write to vector store: {exc}") from exc
        self.version += 1

    def query(self, embedding: list[float], n_results: int) -> list[DenseHit]:
        total = self._collection.count()
        if total == 0:
            return []
        try:
            res = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results, total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {exc}") from exc
        hits = []
        for cid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
        ):
            hits.append(DenseHit(Chunk(cid, text, dict(meta or {})), similarity=1.0 - float(dist)))
        return hits

    def all_chunks(self) -> list[Chunk]:
        res = self._collection.get(include=["documents", "metadatas"])
        return [
            Chunk(cid, text, dict(meta or {}))
            for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]

    def has_document(self, doc_id: str) -> bool:
        return bool(self._collection.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def count_chunks(self, doc_id: str | None = None) -> int:
        if doc_id is None:
            return self._collection.count()
        return len(self._collection.get(where={"doc_id": doc_id})["ids"])

    def list_documents(self) -> list[dict]:
        res = self._collection.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for meta in res["metadatas"]:
            meta = meta or {}
            doc = docs.setdefault(
                meta.get("doc_id", "unknown"),
                {"doc_id": meta.get("doc_id"), "source": meta.get("source"),
                 "file_type": meta.get("file_type"), "chunks": 0},
            )
            doc["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["source"] or "")

    def delete_document(self, doc_id: str) -> int:
        ids = self._collection.get(where={"doc_id": doc_id})["ids"]
        if ids:
            self._collection.delete(ids=ids)
            self.version += 1
        return len(ids)
