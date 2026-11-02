"""Request/response models for the HTTP API (Pydantic v2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=3, max_length=2000, examples=["What is the return window for Gold members?"])
    mode: Literal["auto", "documents", "sql", "both"] = "auto"

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return v


class Citation(BaseModel):
    id: int
    source: str
    location: str
    page: int | None = None
    chunk_id: str
    snippet: str
    score: float | None = None


class SQLInfo(BaseModel):
    query: str | None = None
    columns: list[str] = []
    rows: list[list] = []
    row_count: int = 0
    truncated: bool = False
    error: str | None = None
    attempts: int = 0


class Grounding(BaseModel):
    checked: bool = False
    numbers_unsupported: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    route: Literal["documents", "sql", "both"]
    route_reason: str
    citations: list[Citation] = []
    sql: SQLInfo | None = None
    insufficient_evidence: bool = False
    grounding: Grounding = Grounding()
    warnings: list[str] = []
    steps: list[str] = []
    latency_ms: float


class IngestFileResult(BaseModel):
    filename: str
    status: Literal["ingested", "duplicate", "error"]
    doc_id: str | None = None
    chunks: int = 0
    error: str | None = None


class IngestResponse(BaseModel):
    results: list[IngestFileResult]


class DocumentInfo(BaseModel):
    doc_id: str | None
    source: str | None
    file_type: str | None
    chunks: int


class SQLRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sql: str = Field(..., min_length=6, max_length=10_000)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    llm_configured: bool
    database_reachable: bool
    indexed_chunks: int
    tracing_enabled: bool


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict = {}
