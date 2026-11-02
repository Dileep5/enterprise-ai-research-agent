import pytest

from app.ingestion.chunking import Chunk
from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.hybrid import HybridRetriever
from tests.conftest import OverlapReranker, make_settings


def test_rrf_matches_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion([["x", "shared"], ["y", "shared"]], k=1)
    assert fused[0][0] == "shared"


def test_rrf_ignores_duplicates_within_a_list_and_rejects_bad_k():
    assert dict(reciprocal_rank_fusion([["a", "a"]], k=60))["a"] == pytest.approx(1 / 61)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)


def test_bm25_prefers_exact_terms():
    chunks = [Chunk("1", "restocking fee for opened tents"), Chunk("2", "loyalty points expire"),
              Chunk("3", "store opening hours")]
    hits = BM25Index(chunks).search("restocking fee", 3)
    assert hits[0][0].chunk_id == "1"


def test_bm25_empty_index_and_stopword_query():
    assert BM25Index([]).search("anything", 5) == []
    assert BM25Index([Chunk("1", "text")]).search("the of and", 5) == []
    assert tokenize("The Gold-tier's 60-day window") == ["gold", "tier's", "60", "day", "window"]


def test_hashing_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder()
    v1, v2 = e.embed_query("return window"), e.embed_query("return window")
    assert v1 == v2
    assert sum(x * x for x in v1) == pytest.approx(1.0)


def test_ingestion_is_idempotent(make_services):
    services = make_services()
    first = services.ingestion.ingest_bytes(b"Gold members get a 60 day return window.", "policy.txt")
    second = services.ingestion.ingest_bytes(b"Gold members get a 60 day return window.", "policy.txt")
    assert first.status == "ingested" and second.status == "duplicate"
    assert services.store.count_chunks() == first.chunks
    assert services.store.list_documents()[0]["source"] == "policy.txt"


def test_hybrid_retrieval_finds_relevant_document(ingested_services):
    services = ingested_services()
    result = services.retriever.retrieve("What restocking fee applies to opened tents?")
    assert result.sufficient
    assert result.chunks[0].metadata["source"] == "return_policy.txt"
    assert result.chunks[0].rerank_score is not None
    assert all(c.rrf_score > 0 for c in result.chunks)


def test_hybrid_retrieval_cites_pdf_pages(ingested_services):
    result = ingested_services().retriever.retrieve("What was the Net Promoter Score?")
    top = result.chunks[0]
    assert top.metadata["source"] == "annual_report_2025.pdf"
    assert top.metadata["page"] == 2


def test_insufficient_evidence_for_unrelated_question(ingested_services):
    result = ingested_services().retriever.retrieve("Who won the 1998 FIFA World Cup final?")
    assert not result.sufficient


def test_empty_store_returns_insufficient(make_services):
    result = make_services().retriever.retrieve("anything at all")
    assert result.chunks == [] and not result.sufficient


def test_dense_threshold_used_without_reranker(tmp_path):
    from app.ingestion.pipeline import IngestionPipeline
    from app.retrieval.vector_store import ChromaVectorStore

    settings = make_settings(tmp_path, min_dense_similarity=0.99)
    store = ChromaVectorStore(settings.chroma_path, settings.chroma_collection)
    IngestionPipeline(settings, HashingEmbedder(), store).ingest_bytes(b"alpha beta gamma delta", "a.txt")
    result = HybridRetriever(settings, HashingEmbedder(), store, None).retrieve("alpha zeta")
    assert result.chunks and not result.sufficient
    assert "dense similarity" in result.reason


def test_bm25_index_rebuilds_after_new_ingestion(make_services):
    services = make_services(reranker=OverlapReranker())
    services.ingestion.ingest_bytes(b"first document about kayaks", "a.txt")
    services.retriever.retrieve("kayaks")
    services.ingestion.ingest_bytes(b"second document about snowshoes", "b.txt")
    result = services.retriever.retrieve("snowshoes")
    assert result.chunks[0].metadata["source"] == "b.txt"
    assert result.chunks[0].bm25_score is not None
