"""MCP server exposing the agent's read-only capabilities to MCP clients
(Claude Desktop, IDEs, other agents).

Run over stdio:   python -m app.mcp_server

Every tool is read-only: document search never writes to the vector store and
SQL passes through the same AST guard + read-only transaction as the API.
"""

from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP

from app.errors import AppError
from app.logging_config import configure_logging
from app.services import Services, get_services

mcp = FastMCP("enterprise-research-agent")

_services_factory = get_services


def set_services_factory(factory) -> None:
    """Override the services factory (used by tests)."""
    global _services_factory
    _services_factory = factory


def _services() -> Services:
    return _services_factory()


def _error(exc: AppError) -> dict:
    return {"error": exc.code, "message": exc.message}


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> dict:
    """Hybrid (vector + BM25 + rerank) search over ingested documents. Returns passages with sources."""
    top_k = max(1, min(int(top_k), 20))
    try:
        result = _services().retriever.retrieve(query, top_k=top_k)
    except AppError as exc:
        return _error(exc)
    return {
        "sufficient_evidence": result.sufficient,
        "reason": result.reason,
        "passages": [
            {"source": c.metadata.get("source"), "page": c.metadata.get("page"),
             "chunk_id": c.chunk_id, "text": c.text, "rerank_score": c.rerank_score}
            for c in result.chunks
        ],
    }


@mcp.tool()
def list_documents() -> list[dict]:
    """List documents currently indexed in the vector store."""
    return _services().store.list_documents()


@mcp.tool()
def list_tables() -> list[str]:
    """List the retail database tables the agent may query."""
    return _services().database.list_tables()


@mcp.tool()
def describe_table(table: str) -> dict:
    """Describe the columns of one retail table."""
    try:
        return {"table": table, "columns": _services().database.describe_table(table)}
    except AppError as exc:
        return _error(exc)


@mcp.tool()
def run_readonly_sql(sql: str) -> dict:
    """Run a single SELECT against the retail database. Writes, DDL and system tables are rejected."""
    try:
        return _services().database.execute(sql).to_dict()
    except AppError as exc:
        return _error(exc)


@mcp.tool()
def ask_research_agent(question: str) -> dict:
    """Ask the full LangGraph agent a question; returns a cited answer."""
    state = _services().agent.run(question)
    return {
        "answer": state["answer"],
        "route": state["route"],
        "citations": state.get("citations", []),
        "sql_query": state.get("sql_query"),
        "insufficient_evidence": state.get("insufficient_evidence", False),
        "warnings": state.get("warnings", []),
    }


def main() -> None:
    # stdout carries the MCP protocol, so logs must go to stderr.
    configure_logging("WARNING", stream=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
