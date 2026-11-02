"""Application configuration loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Enterprise AI Research & Data Intelligence Agent"
    log_level: str = "INFO"

    # --- LLM -----------------------------------------------------------------
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    openai_timeout_s: float = 30.0
    openai_max_retries: int = 2
    llm_temperature: float = 0.0

    # --- Retrieval -----------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_reranker: bool = True
    chroma_path: Path = Path("./storage/chroma")
    chroma_collection: str = "documents"
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_candidates: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    # Evidence gate (stage 1 of refusal; the LLM's INSUFFICIENT_EVIDENCE is stage 2).
    # Calibrated for recall on eval/retrieval_paraphrase.jsonl + eval/calibration_unanswerable.jsonl:
    # at -9.0 the gate passes 16/20 answerable paraphrases (vs 6/20 at -3.0) and still
    # rejects 15/20 unanswerable questions before any LLM call. See IMPLEMENTATION_DETAILS.md.
    min_rerank_score: float = -9.0
    min_dense_similarity: float = 0.30

    # --- SQL -----------------------------------------------------------------
    # Read-only role used by the agent. The admin URL is only used by seed scripts.
    database_url: str | None = None
    database_admin_url: str | None = None
    readonly_db_user: str = "agent_readonly"
    readonly_db_password: SecretStr | None = None  # required by scripts/seed_db.py; no default
    sql_row_limit: int = Field(default=200, ge=1, le=10_000)
    sql_statement_timeout_ms: int = Field(default=5_000, ge=100)
    sql_max_attempts: int = Field(default=2, ge=1, le=5)

    # --- API -----------------------------------------------------------------
    max_upload_mb: int = Field(default=20, ge=1)
    max_question_chars: int = Field(default=2_000, ge=10)

    # --- Observability -------------------------------------------------------
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_chunk(cls, v: int, info):
        size = info.data.get("chunk_size", 800)
        if v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value().strip())

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
