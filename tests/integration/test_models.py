"""Tests with the real Sentence Transformers embedder and cross-encoder.

Downloads ~180 MB of models on first run. Enable with RUN_MODEL_TESTS=1.
"""

import pytest

from app.retrieval.embeddings import SentenceTransformerEmbedder
from app.retrieval.reranker import CrossEncoderReranker
from app.services import build_services
from tests.conftest import SAMPLE_DOCS, FakeDatabase, make_settings

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def services(tmp_path_factory):
    # Use the production thresholds from Settings defaults.
    settings = make_settings(tmp_path_factory.mktemp("models"), chunk_size=800, chunk_overlap=120,
                             min_rerank_score=-3.0, min_dense_similarity=0.30)
    svc = build_services(
        settings,
        embedder=SentenceTransformerEmbedder(settings.embedding_model),
        reranker=CrossEncoderReranker(settings.reranker_model),
        database=FakeDatabase(),
    )
    for path in sorted(SAMPLE_DOCS.iterdir()):
        svc.ingestion.ingest_path(path)
    return svc


def test_embeddings_are_normalised():
    vec = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2").embed_query("tents")
    assert len(vec) == 384
    assert sum(v * v for v in vec) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(
    "question, source",
    [
        ("How long do Gold members have to return items?", "return_policy.txt"),
        ("How many points per dollar do Gold members earn?", "loyalty_program.docx"),
        ("What was net revenue in fiscal 2025?", "annual_report_2025.pdf"),
        ("When must stock-outs of top SKUs be escalated?", "store_operations_handbook.md"),
        ("Which supplier has the longest lead time?", "supplier_directory.csv"),
    ],
)
def test_real_models_retrieve_correct_source(services, question, source):
    result = services.retriever.retrieve(question)
    assert result.sufficient, result.reason
    assert result.chunks[0].metadata["source"] == source


def test_reranker_rejects_off_topic_question(services):
    result = services.retriever.retrieve("What is the boiling point of mercury in Kelvin?")
    assert not result.sufficient, result.reason
