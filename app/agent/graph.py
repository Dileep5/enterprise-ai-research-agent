r"""LangGraph workflow that routes questions to document retrieval, SQL, or both.

    START -> route --(documents)--> retrieve -----------------------------> synthesize -> END
                   \--(both)------> retrieve -> generate_sql -> execute_sql -/
                    \--(sql)------------------> generate_sql -> execute_sql -/
                                                     ^            |
                                                     +-- retry ---+  (validation / DB error)

If an SQL-routed question yields no usable rows, or the SQL evidence does not answer it
(the LLM refuses), the graph falls back to ``retrieve`` once before refusing (the router
may have mis-classified a document question).

Every node converts domain errors into state (``warnings`` / ``sql_error``) instead of
raising, so a failing dependency degrades the answer rather than failing the request.
"""

from __future__ import annotations

import logging
import operator
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.agent.citations import build_citation, cited_indices, format_document_evidence, location_label
from app.agent.grounding import unsupported_numbers
from app.agent.router import route_question
from app.agent.sql_generation import generate_sql
from app.errors import (
    AppError,
    DatabaseUnavailableError,
    LLMError,
    LLMUnavailableError,
    SQLExecutionError,
    UnsafeSQLError,
)
from app.llm.client import LLM
from app.llm.prompts import ANSWER_SYSTEM, ANSWER_USER, INSUFFICIENT_EVIDENCE_TOKEN

logger = logging.getLogger(__name__)

Route = Literal["documents", "sql", "both"]

INSUFFICIENT_MESSAGE = (
    "I could not find sufficient evidence in the available sources to answer this question."
)
LLM_NOT_CONFIGURED = "The language model is not configured (set OPENAI_API_KEY)."


class AgentState(TypedDict, total=False):
    question: str
    mode: str
    route: Route
    route_reason: str
    retrieval: dict
    sql_query: str | None
    sql_result: dict | None
    sql_error: str | None
    sql_retryable: bool
    sql_attempts: int
    answer: str
    citations: list[dict]
    insufficient_evidence: bool
    grounding: dict
    warnings: Annotated[list[str], operator.add]
    steps: Annotated[list[str], operator.add]


class ResearchAgent:
    def __init__(self, *, llm: LLM, retriever, database, sql_max_attempts: int = 2, callbacks=None):
        self.llm = llm
        self.retriever = retriever
        self.database = database
        self.sql_max_attempts = sql_max_attempts
        self.callbacks = callbacks or []
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def _route(self, state: AgentState, config: RunnableConfig) -> dict:
        mode = state.get("mode", "auto")
        if mode in ("documents", "sql", "both"):
            return {"route": mode, "route_reason": "forced by request", "steps": ["route"]}
        route, reason = route_question(state["question"], self.llm, config)
        return {"route": route, "route_reason": reason, "steps": ["route"]}

    def _retrieve(self, state: AgentState) -> dict:
        try:
            result = self.retriever.retrieve(state["question"])
        except AppError as exc:
            logger.warning("Retrieval failed: %s", exc.message)
            return {
                "retrieval": {"chunks": [], "sufficient": False, "reason": exc.message},
                "warnings": [f"Document search unavailable: {exc.message}"],
                "steps": ["retrieve"],
            }
        return {
            "retrieval": {
                "chunks": [c.to_dict() for c in result.chunks],
                "sufficient": result.sufficient,
                "reason": result.reason,
            },
            "steps": ["retrieve"],
        }

    def _generate_sql(self, state: AgentState, config: RunnableConfig) -> dict:
        attempts = state.get("sql_attempts", 0) + 1
        if not self.llm.available:
            return {"sql_query": None, "sql_error": LLM_NOT_CONFIGURED + " SQL generation needs a model.",
                    "sql_retryable": False, "sql_attempts": attempts, "steps": ["generate_sql"]}
        retrieval = state.get("retrieval") or {}
        context = None
        if retrieval.get("sufficient"):
            # Multi-source questions: let document definitions (e.g. "below the reorder
            # point") inform the query.
            context = "\n---\n".join(c["text"] for c in retrieval["chunks"][:3])[:2000]
        try:
            schema = self.database.schema_description()
            sql = generate_sql(
                self.llm,
                state["question"],
                schema,
                previous_sql=state.get("sql_query") if state.get("sql_error") else None,
                error=state.get("sql_error"),
                context=context,
                config=config,
            )
        except (LLMUnavailableError, LLMError, DatabaseUnavailableError) as exc:
            return {"sql_query": None, "sql_error": exc.message, "sql_retryable": False,
                    "sql_attempts": attempts, "steps": ["generate_sql"]}
        return {"sql_query": sql, "sql_error": None, "sql_attempts": attempts, "steps": ["generate_sql"]}

    def _execute_sql(self, state: AgentState) -> dict:
        if state.get("sql_error") or not state.get("sql_query"):
            return {"steps": ["execute_sql"]}
        try:
            result = self.database.execute(state["sql_query"])
        except (UnsafeSQLError, SQLExecutionError) as exc:
            return {"sql_result": None, "sql_error": exc.message, "sql_retryable": True,
                    "steps": ["execute_sql"]}
        except DatabaseUnavailableError as exc:
            return {"sql_result": None, "sql_error": exc.message, "sql_retryable": False,
                    "steps": ["execute_sql"]}
        return {"sql_result": result.to_dict(), "sql_query": result.query, "sql_error": None,
                "steps": ["execute_sql"]}

    def _synthesize(self, state: AgentState, config: RunnableConfig) -> dict:
        warnings: list[str] = []
        retrieval = state.get("retrieval") or {}
        chunks = retrieval.get("chunks", []) if retrieval.get("sufficient") else []
        sql_result = state.get("sql_result")

        if retrieval and not retrieval.get("sufficient"):
            warnings.append(f"Document evidence insufficient ({retrieval.get('reason')}).")
        if state.get("sql_error"):
            warnings.append(f"SQL step failed: {state['sql_error']}")

        if not chunks and not sql_result:
            answer = INSUFFICIENT_MESSAGE
            if state.get("sql_error"):
                answer += f" The database query could not be completed: {state['sql_error']}"
            return {"answer": answer, "citations": [], "insufficient_evidence": True,
                    "warnings": warnings, "steps": ["synthesize"]}

        evidence = []
        if chunks:
            evidence.append(format_document_evidence(chunks))
        if sql_result:
            evidence.append(self._format_sql_evidence(sql_result))

        try:
            if not self.llm.available:
                raise LLMUnavailableError(LLM_NOT_CONFIGURED)
            answer = self.llm.complete(
                ANSWER_SYSTEM,
                ANSWER_USER.format(question=state["question"], evidence="\n\n".join(evidence)),
                config=config,
            )
        except (LLMUnavailableError, LLMError) as exc:
            warnings.append(f"Answer generation unavailable: {exc.message}")
            return {**self._extractive_fallback(chunks, sql_result), "warnings": warnings,
                    "steps": ["synthesize"]}

        if INSUFFICIENT_EVIDENCE_TOKEN in answer:
            return {"answer": INSUFFICIENT_MESSAGE, "citations": [], "insufficient_evidence": True,
                    "warnings": warnings, "steps": ["synthesize"]}

        indices = cited_indices(answer)
        valid = [i for i in indices if 1 <= i <= len(chunks)]
        if len(valid) != len(indices):
            warnings.append("Answer referenced citation numbers that do not exist; they were ignored.")
        if chunks and not valid and not sql_result:
            warnings.append("Answer did not cite any document evidence.")
        citations = [build_citation(i, chunks[i - 1]) for i in valid]
        sources = {str(i): c["text"] for i, c in enumerate(chunks, start=1)}
        if sql_result:
            sources["SQL"] = self._format_sql_evidence(sql_result)
        unsupported = unsupported_numbers(answer, "\n\n".join(evidence), state["question"], sources)
        if unsupported:
            warnings.append("Answer contains numbers not found in the evidence (verify or treat as derived): "
                            + ", ".join(unsupported))
        return {"answer": answer, "citations": citations, "insufficient_evidence": False,
                "grounding": {"numbers_unsupported": unsupported, "checked": True},
                "warnings": warnings, "steps": ["synthesize"]}

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _format_sql_evidence(sql_result: dict) -> str:
        from app.sql.database import SQLResult

        table = SQLResult(**sql_result).to_markdown()
        return f"[SQL] Result of query:\n{sql_result['query']}\n\n{table}"

    @staticmethod
    def _extractive_fallback(chunks: list[dict], sql_result: dict | None) -> dict:
        parts = ["The language model is unavailable, so the retrieved evidence is shown directly."]
        if sql_result:
            from app.sql.database import SQLResult

            parts.append("Database result [SQL]:\n" + SQLResult(**sql_result).to_markdown())
        for i, chunk in enumerate(chunks[:3], start=1):
            parts.append(f"[{i}] {location_label(chunk['metadata'])}: {chunk['text'][:400]}")
        citations = [build_citation(i, c) for i, c in enumerate(chunks[:3], start=1)]
        return {"answer": "\n\n".join(parts), "citations": citations, "insufficient_evidence": False}

    # ----------------------------------------------------------------- edges
    @staticmethod
    def _after_route(state: AgentState) -> str:
        return "generate_sql" if state["route"] == "sql" else "retrieve"

    @staticmethod
    def _after_retrieve(state: AgentState) -> str:
        if state["route"] == "both" and not state.get("sql_attempts"):
            return "generate_sql"
        return "synthesize"

    @staticmethod
    def _sql_usable(state: AgentState) -> bool:
        result = state.get("sql_result")
        return bool(result and result["row_count"] and any(v is not None for row in result["rows"] for v in row))

    def _after_execute(self, state: AgentState) -> str:
        if (
            state.get("sql_error")
            and state.get("sql_retryable")
            and state.get("sql_attempts", 0) < self.sql_max_attempts
        ):
            return "generate_sql"
        if state["route"] == "sql" and not self._sql_usable(state) and "retrieve" not in state.get("steps", []):
            # Mis-routed or unanswerable in SQL: try the documents before giving up.
            return "retrieve"
        return "synthesize"

    @staticmethod
    def _after_synthesize(state: AgentState) -> str:
        # The SQL result did not answer the question (the LLM refused): the router may have
        # sent a document question to SQL, so consult the documents before refusing.
        if (
            state.get("insufficient_evidence")
            and state["route"] == "sql"
            and "retrieve" not in state.get("steps", [])
        ):
            return "retrieve"
        return END

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("route", self._route)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate_sql", self._generate_sql)
        graph.add_node("execute_sql", self._execute_sql)
        graph.add_node("synthesize", self._synthesize)

        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", self._after_route, ["retrieve", "generate_sql"])
        graph.add_conditional_edges("retrieve", self._after_retrieve, ["generate_sql", "synthesize"])
        graph.add_edge("generate_sql", "execute_sql")
        graph.add_conditional_edges("execute_sql", self._after_execute, ["generate_sql", "retrieve", "synthesize"])
        graph.add_conditional_edges("synthesize", self._after_synthesize, ["retrieve", END])
        return graph.compile()

    # ------------------------------------------------------------------- run
    def run(self, question: str, mode: str = "auto") -> dict:
        started = time.perf_counter()
        initial: AgentState = {"question": question, "mode": mode, "sql_attempts": 0,
                               "warnings": [], "steps": []}
        config: RunnableConfig = {"callbacks": self.callbacks, "run_name": "research-agent",
                                  "recursion_limit": 25}
        final = self.graph.invoke(initial, config=config)
        final["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return final
