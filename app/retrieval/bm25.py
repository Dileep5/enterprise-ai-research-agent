"""In-memory BM25 keyword index rebuilt from the vector store's chunks."""

from __future__ import annotations

import math
import re

from rank_bm25 import BM25Okapi

from app.ingestion.chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "has", "have",
    "how", "i", "in", "is", "it", "its", "of", "on", "or", "our", "the", "their", "this", "to",
    "was", "we", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
})


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class _LuceneBM25(BM25Okapi):
    """BM25Okapi with Lucene's IDF, log(1 + (N - n + 0.5) / (n + 0.5)).

    The classic Okapi IDF is zero or negative for terms present in at least half of
    the chunks, which hides relevant matches in small corpora (e.g. two uploaded files).
    """

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        corpus = [tokenize(c.text) for c in chunks]
        # BM25 divides by corpus size; guard the empty case.
        self._bm25 = _LuceneBM25(corpus) if chunks else None

    def search(self, query: str, n_results: int) -> list[tuple[Chunk, float]]:
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.chunks[i], float(scores[i])) for i in ranked[:n_results] if scores[i] > 0]
