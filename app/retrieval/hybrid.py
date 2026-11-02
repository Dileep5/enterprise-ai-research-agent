"""Hybrid retrieval: dense (Chroma) + sparse (BM25) -> RRF -> cross-encoder rerank."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from app.config import Settings
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import Embedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    dense_similarity: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk]
    sufficient: bool
    reason: str


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: ChromaVectorStore,
        reranker: Reranker | None = None,
    ):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self._bm25: BM25Index | None = None
        self._bm25_version = -1

    def _bm25_index(self) -> BM25Index:
        if self._bm25 is None or self._bm25_version != self.store.version:
            self._bm25 = BM25Index(self.store.all_chunks())
            self._bm25_version = self.store.version
        return self._bm25

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or self.settings.retrieval_top_k
        n = self.settings.retrieval_candidates

        dense_hits = self.store.query(self.embedder.embed_query(query), n)
        sparse_hits = self._bm25_index().search(query, n)
        if not dense_hits and not sparse_hits:
            return RetrievalResult([], False, "No documents have been ingested.")

        candidates: dict[str, ScoredChunk] = {}
        for hit in dense_hits:
            candidates[hit.chunk.chunk_id] = ScoredChunk(
                hit.chunk.chunk_id, hit.chunk.text, hit.chunk.metadata, dense_similarity=hit.similarity
            )
        for chunk, score in sparse_hits:
            sc = candidates.setdefault(chunk.chunk_id, ScoredChunk(chunk.chunk_id, chunk.text, chunk.metadata))
            sc.bm25_score = score

        fused = reciprocal_rank_fusion(
            [[h.chunk.chunk_id for h in dense_hits], [c.chunk_id for c, _ in sparse_hits]],
            k=self.settings.rrf_k,
        )
        ranked = []
        for cid, rrf in fused[:n]:
            candidates[cid].rrf_score = rrf
            ranked.append(candidates[cid])

        if self.reranker is not None:
            scores = self.reranker.score(query, [c.text for c in ranked])
            for chunk, score in zip(ranked, scores, strict=True):
                chunk.rerank_score = score
            ranked.sort(key=lambda c: c.rerank_score, reverse=True)

        top = ranked[:top_k]
        sufficient, reason = self._assess_evidence(top)
        return RetrievalResult(top, sufficient, reason)

    def _assess_evidence(self, chunks: list[ScoredChunk]) -> tuple[bool, str]:
        if not chunks:
            return False, "No candidate passages were found."
        if self.reranker is not None:
            best = max(c.rerank_score for c in chunks)
            ok = best >= self.settings.min_rerank_score
            return ok, f"best rerank score {best:.2f} (threshold {self.settings.min_rerank_score})"
        dense = [c.dense_similarity for c in chunks if c.dense_similarity is not None]
        best = max(dense) if dense else 0.0
        ok = best >= self.settings.min_dense_similarity
        return ok, f"best dense similarity {best:.2f} (threshold {self.settings.min_dense_similarity})"
