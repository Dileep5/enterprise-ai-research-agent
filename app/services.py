"""Composition root: wires settings into concrete components.

Both the FastAPI app and the MCP server build their dependencies here, and tests
construct ``Services`` directly with fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.agent.graph import ResearchAgent
from app.config import Settings, get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.llm.client import LLM, OpenAIChatLLM
from app.observability.tracing import build_callbacks
from app.retrieval.embeddings import Embedder, SentenceTransformerEmbedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker, Reranker
from app.retrieval.vector_store import ChromaVectorStore
from app.sql.database import RetailDatabase


@dataclass
class Services:
    settings: Settings
    store: ChromaVectorStore
    ingestion: IngestionPipeline
    retriever: HybridRetriever
    database: RetailDatabase
    llm: LLM
    agent: ResearchAgent


def build_services(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    llm: LLM | None = None,
    database: RetailDatabase | None = None,
    store: ChromaVectorStore | None = None,
    use_reranker: bool | None = None,
) -> Services:
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    use_reranker = settings.enable_reranker if use_reranker is None else use_reranker
    if reranker is None and use_reranker:
        reranker = CrossEncoderReranker(settings.reranker_model)
    store = store or ChromaVectorStore(settings.chroma_path, settings.chroma_collection)
    database = database or RetailDatabase(
        settings.database_url,
        row_limit=settings.sql_row_limit,
        statement_timeout_ms=settings.sql_statement_timeout_ms,
    )
    llm = llm or OpenAIChatLLM(settings)
    retriever = HybridRetriever(settings, embedder, store, reranker if use_reranker else None)
    agent = ResearchAgent(
        llm=llm,
        retriever=retriever,
        database=database,
        sql_max_attempts=settings.sql_max_attempts,
        callbacks=build_callbacks(settings),
    )
    return Services(settings, store, IngestionPipeline(settings, embedder, store), retriever, database, llm, agent)


@lru_cache
def get_services() -> Services:
    return build_services(get_settings())
