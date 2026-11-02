"""Domain exceptions. Each carries an HTTP status and a stable error code."""

from __future__ import annotations


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# --- Ingestion ---------------------------------------------------------------
class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class DocumentParseError(AppError):
    status_code = 422
    code = "document_parse_error"


class EmptyDocumentError(AppError):
    status_code = 422
    code = "empty_document"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


# --- LLM ---------------------------------------------------------------------
class LLMUnavailableError(AppError):
    status_code = 503
    code = "llm_not_configured"


class LLMError(AppError):
    status_code = 502
    code = "llm_error"


# --- SQL ---------------------------------------------------------------------
class UnsafeSQLError(AppError):
    status_code = 400
    code = "unsafe_sql"


class SQLExecutionError(AppError):
    status_code = 400
    code = "sql_execution_error"


class DatabaseUnavailableError(AppError):
    status_code = 503
    code = "database_unavailable"


# --- Retrieval ---------------------------------------------------------------
class VectorStoreError(AppError):
    status_code = 503
    code = "vector_store_error"
