"""Shared fixtures. Unit tests use deterministic fakes: no network, no model downloads."""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from app.config import Settings
from app.errors import DatabaseUnavailableError, LLMError
from app.llm.prompts import ROUTER_SYSTEM
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.vector_store import ChromaVectorStore
from app.services import Services, build_services
from app.sql.database import SQLResult
from app.sql.guard import validate_sql

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = ROOT / "data" / "documents"


class FakeLLM:
    """Scripted LLM. ``handler(kind, system, user)`` returns the completion text,
    where kind is 'route', 'sql' or 'answer'."""

    def __init__(self, handler: Callable[[str, str, str], str], available: bool = True):
        self.handler = handler
        self.available = available
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def kind(system: str) -> str:
        if system == ROUTER_SYSTEM:
            return "route"
        if system.startswith("You write a single PostgreSQL"):
            return "sql"
        return "answer"

    def complete(self, system: str, user: str, *, config=None) -> str:
        kind = self.kind(system)
        self.calls.append((kind, user))
        return self.handler(kind, system, user)


class FailingLLM:
    available = True

    def complete(self, system, user, *, config=None):
        raise LLMError("Could not reach the OpenAI API (timeout or network error).")


class OverlapReranker:
    """Scores passages by query-term overlap; enough to exercise reranking logic."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        return [float(len(q & set(re.findall(r"[a-z0-9]+", p.lower())))) - 2.0 for p in passages]


class FakeDatabase:
    """In-memory stand-in for RetailDatabase. Applies the real SQL guard."""

    def __init__(self, results: dict[str, SQLResult] | None = None, *, unavailable: bool = False,
                 fail_first: bool = False):
        self.results = results or {}
        self.unavailable = unavailable
        self.fail_first = fail_first
        self.executed: list[str] = []

    def schema_description(self) -> str:
        if self.unavailable:
            raise DatabaseUnavailableError("The retail database is unavailable.")
        return "orders(order_id integer, order_date date, status text)"

    def execute(self, sql: str) -> SQLResult:
        from app.errors import SQLExecutionError

        if self.unavailable:
            raise DatabaseUnavailableError("The retail database is unavailable.")
        safe = validate_sql(sql, row_limit=200)
        self.executed.append(safe)
        if self.fail_first and len(self.executed) == 1:
            raise SQLExecutionError('Database rejected the query: column "revenue" does not exist')
        return self.results.get("default", SQLResult(safe, ["value"], [[42]], 1, False))

    def list_tables(self):
        return ["orders"]

    def describe_table(self, table):
        return [{"column": "order_id", "type": "integer", "nullable": False}]

    def ping(self) -> bool:
        return not self.unavailable


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        _env_file=None,
        openai_api_key=None,
        langfuse_public_key=None,
        langfuse_secret_key=None,
        chroma_path=tmp_path / "chroma",
        chroma_collection=f"test-{uuid.uuid4().hex[:8]}",
        chunk_size=400,
        chunk_overlap=60,
        enable_reranker=True,
        min_rerank_score=0.0,
        min_dense_similarity=0.2,
        database_url=None,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def store(settings) -> ChromaVectorStore:
    return ChromaVectorStore(settings.chroma_path, settings.chroma_collection)


@pytest.fixture
def make_services(settings, store):
    def _make(llm=None, database=None, reranker=None, use_reranker=True) -> Services:
        return build_services(
            settings,
            embedder=HashingEmbedder(),
            reranker=reranker or (OverlapReranker() if use_reranker else None),
            llm=llm or FakeLLM(lambda *_: "INSUFFICIENT_EVIDENCE", available=False),
            database=database or FakeDatabase(),
            store=store,
            use_reranker=use_reranker,
        )

    return _make


@pytest.fixture
def ingested_services(make_services):
    """Factory returning services with all sample documents ingested."""

    def _make(**kwargs) -> Services:
        services = make_services(**kwargs)
        for path in sorted(SAMPLE_DOCS.iterdir()):
            services.ingestion.ingest_path(path)
        return services

    return _make


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch):
    # Guarantee unit tests never call paid APIs, even if the developer's shell has keys.
    for var in ("OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield


def pytest_collection_modifyitems(config, items):
    if not os.getenv("TEST_DATABASE_URL"):
        skip = pytest.mark.skip(reason="TEST_DATABASE_URL not set")
        for item in items:
            if "postgres" in item.keywords:
                item.add_marker(skip)
    if not os.getenv("RUN_MODEL_TESTS"):
        skip = pytest.mark.skip(reason="RUN_MODEL_TESTS not set (downloads HF models)")
        for item in items:
            if "models" in item.keywords:
                item.add_marker(skip)
