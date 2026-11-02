"""Text-to-SQL helpers."""

from __future__ import annotations

import re

from app.llm.client import LLM
from app.llm.prompts import SQL_CONTEXT_SUFFIX, SQL_RETRY_SUFFIX, SQL_SYSTEM

_FENCED = re.compile(r"```(?:sql|postgresql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    match = _FENCED.search(text)
    sql = match.group(1) if match else text
    return sql.strip().rstrip(";").strip()


def generate_sql(
    llm: LLM,
    question: str,
    schema: str,
    *,
    previous_sql: str | None = None,
    error: str | None = None,
    context: str | None = None,
    config: dict | None = None,
) -> str:
    user = question
    if context:
        user += SQL_CONTEXT_SUFFIX.format(context=context)
    if previous_sql and error:
        user += SQL_RETRY_SUFFIX.format(previous_sql=previous_sql, error=error)
    return extract_sql(llm.complete(SQL_SYSTEM.format(schema=schema), user, config=config))
