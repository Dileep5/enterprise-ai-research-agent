"""Question routing: LLM classification with a deterministic keyword fallback."""

from __future__ import annotations

import json
import logging
import re

from app.errors import AppError
from app.llm.client import LLM
from app.llm.prompts import ROUTER_SYSTEM

logger = logging.getLogger(__name__)

ROUTES = ("documents", "sql", "both")

# Weighted cue phrases. Matching is on word boundaries, case-insensitive.
_SQL_CUES = {
    "revenue": 3, "sales": 2, "sold": 2, "orders": 2, "order count": 3, "units": 2,
    "inventory": 2, "in stock": 2, "stock levels": 3, "on hand": 3, "below their reorder point": 3,
    "below the reorder point": 3, "best-selling": 3, "best selling": 3, "top": 1, "average order value": 3,
    "how many customers": 3, "how many orders": 3, "number of": 1, "total": 1, "per store": 2,
    "by store": 2, "each store": 2, "which store": 2, "by month": 2, "monthly": 2, "by category": 2,
    "channel": 1, "return rate": 2, "returned orders": 3, "cancelled": 2, "customers": 1,
    "margin": 2, "database": 3, "in 2024": 1, "in 2025": 1,
}
_DOC_CUES = {
    "policy": 3, "handbook": 3, "procedure": 2, "annual report": 3, "report": 1, "program": 2,
    "according to": 2, "return window": 3, "refund": 2, "points": 2, "supplier": 3, "suppliers": 3,
    "lead time": 3, "ceo": 3, "strategy": 2, "strategic": 2, "target": 2, "sustainability": 3,
    "guideline": 2, "document": 2, "priorities": 2, "restocking fee": 3, "cycle count": 3,
    "escalat": 2, "threshold": 1, "expire": 2, "benefits": 2, "opening hours": 3, "rating": 1,
}


def _score(question: str, cues: dict[str, int]) -> int:
    text = question.lower()
    return sum(w for cue, w in cues.items() if re.search(r"\b" + re.escape(cue), text))


def heuristic_route(question: str) -> tuple[str, str]:
    sql_score, doc_score = _score(question, _SQL_CUES), _score(question, _DOC_CUES)
    if sql_score >= 3 and doc_score >= 3:
        route = "both"
    elif sql_score > doc_score:
        route = "sql"
    else:
        route = "documents"
    return route, f"keyword heuristic (sql={sql_score}, documents={doc_score})"


def parse_route(text: str) -> tuple[str, str] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    route = str(payload.get("route", "")).strip().lower()
    if route not in ROUTES:
        return None
    return route, str(payload.get("reason", "")).strip() or "llm"


def route_question(question: str, llm: LLM | None, config: dict | None = None) -> tuple[str, str]:
    if llm is not None and llm.available:
        try:
            parsed = parse_route(llm.complete(ROUTER_SYSTEM, question, config=config))
            if parsed:
                return parsed[0], f"llm: {parsed[1]}"
            logger.warning("Router returned unparseable output; using heuristic.")
        except AppError as exc:
            logger.warning("LLM router failed (%s); using heuristic.", exc.message)
    return heuristic_route(question)
