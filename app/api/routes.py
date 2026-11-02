"""HTTP routes. Handlers are sync functions so FastAPI runs them in a thread pool
(embedding, reranking and database calls are blocking)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.api.schemas import (
    DocumentInfo,
    HealthResponse,
    IngestFileResult,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SQLInfo,
    SQLRequest,
)
from app.errors import AppError, FileTooLargeError
from app.services import Services

logger = logging.getLogger(__name__)
router = APIRouter()


def services_dep(request: Request) -> Services:
    return request.app.state.get_services()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(services: Services = Depends(services_dep)) -> HealthResponse:
    db_ok = services.database.ping()
    try:
        chunks = services.store.count_chunks()
    except Exception:
        chunks = 0
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        llm_configured=services.llm.available,
        database_reachable=db_ok,
        indexed_chunks=chunks,
        tracing_enabled=bool(services.agent.callbacks),
    )


@router.post("/documents", response_model=IngestResponse, tags=["documents"])
def upload_documents(
    files: list[UploadFile] = File(..., description="PDF, DOCX, TXT, MD or CSV files"),
    services: Services = Depends(services_dep),
) -> IngestResponse:
    max_bytes = services.settings.max_upload_mb * 1024 * 1024
    results: list[IngestFileResult] = []
    for upload in files:
        name = upload.filename or "unnamed"
        try:
            data = upload.file.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise FileTooLargeError(f"'{name}' exceeds the {services.settings.max_upload_mb} MB limit.")
            outcome = services.ingestion.ingest_bytes(data, name)
            results.append(IngestFileResult(filename=outcome.filename, status=outcome.status,
                                            doc_id=outcome.doc_id, chunks=outcome.chunks))
        except AppError as exc:
            logger.info("Rejected upload %s: %s", name, exc.message)
            results.append(IngestFileResult(filename=name, status="error", error=exc.message))
            if len(files) == 1:
                raise
    if results and all(r.status == "error" for r in results):
        raise HTTPException(status_code=400, detail=[r.model_dump() for r in results])
    return IngestResponse(results=results)


@router.get("/documents", response_model=list[DocumentInfo], tags=["documents"])
def list_documents(services: Services = Depends(services_dep)) -> list[DocumentInfo]:
    return [DocumentInfo(**d) for d in services.store.list_documents()]


@router.delete("/documents/{doc_id}", tags=["documents"])
def delete_document(doc_id: str, services: Services = Depends(services_dep)) -> dict:
    removed = services.store.delete_document(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {"doc_id": doc_id, "deleted_chunks": removed}


@router.post("/query", response_model=QueryResponse, tags=["agent"])
def query(body: QueryRequest, services: Services = Depends(services_dep)) -> QueryResponse:
    state = services.agent.run(body.question, mode=body.mode)
    sql = None
    if state.get("route") in ("sql", "both"):
        result = state.get("sql_result") or {}
        sql = SQLInfo(
            query=state.get("sql_query"),
            columns=result.get("columns", []),
            rows=result.get("rows", []),
            row_count=result.get("row_count", 0),
            truncated=result.get("truncated", False),
            error=state.get("sql_error"),
            attempts=state.get("sql_attempts", 0),
        )
    return QueryResponse(
        answer=state["answer"],
        route=state["route"],
        route_reason=state.get("route_reason", ""),
        citations=state.get("citations", []),
        sql=sql,
        insufficient_evidence=state.get("insufficient_evidence", False),
        grounding=state.get("grounding") or {},
        warnings=state.get("warnings", []),
        steps=state.get("steps", []),
        latency_ms=state["latency_ms"],
    )


@router.post("/sql", response_model=SQLInfo, tags=["sql"])
def run_sql(body: SQLRequest, services: Services = Depends(services_dep)) -> SQLInfo:
    """Run a hand-written read-only query through the same guard the agent uses."""
    result = services.database.execute(body.sql)
    return SQLInfo(**result.to_dict(), attempts=1)


@router.get("/sql/schema", tags=["sql"])
def sql_schema(services: Services = Depends(services_dep)) -> dict:
    return {t: services.database.describe_table(t) for t in services.database.list_tables()}
