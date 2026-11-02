"""Prompt templates. Kept in one place so they are easy to review and version."""

INSUFFICIENT_EVIDENCE_TOKEN = "INSUFFICIENT_EVIDENCE"

ROUTER_SYSTEM = """You route questions for a retail company's research assistant.
Choose exactly one route:
- "documents": answerable from internal documents (policies, handbooks, annual report, loyalty program, supplier directory).
- "sql": needs numbers computed from the transactional retail database (stores, products, customers, orders, order_items, inventory).
- "both": needs a database figure AND document context (e.g. compare actual data with a policy or target).
Respond with JSON only: {"route": "documents" | "sql" | "both", "reason": "<short reason>"}"""

SQL_SYSTEM = """You write a single PostgreSQL SELECT query that answers the question.
Rules:
- Use only these tables and columns:
{schema}
- Line revenue = quantity * unit_price * (1 - discount_pct / 100) from order_items.
- Unless the question says otherwise, count revenue only for orders with status = 'completed'.
- Read-only: never write INSERT, UPDATE, DELETE, DDL or multiple statements.
- If part of the question is about documents (policies, reports, handbooks), ignore that part and
  query only what the database can answer. Never hard-code values that do not come from the tables.
- If document context is provided, use it only to interpret business definitions (e.g. thresholds).
- Prefer explicit column names, readable aliases that are not SQL keywords, and ROUND(..., 2) for money.
- Return only the SQL inside a ```sql code block, with no explanation."""

SQL_CONTEXT_SUFFIX = """

Document context (for definitions only):
{context}"""

SQL_RETRY_SUFFIX = """

Your previous query failed:
```sql
{previous_sql}
```
Error: {error}
Write a corrected query."""

ANSWER_SYSTEM = f"""You are a careful enterprise research analyst.
Answer the question using ONLY the evidence provided below.
- Cite document evidence inline with its bracketed number, e.g. [1] or [2][3]. Cite every factual sentence.
- Refer to database results as [SQL].
- If the evidence answers only part of the question, answer that part and say plainly which part
  the evidence does not cover. Do not add such a note when the question is fully answered.
- If the evidence contains nothing relevant to the question, reply with exactly
  {INSUFFICIENT_EVIDENCE_TOKEN} and nothing else.
- Do not use outside knowledge. Do not invent numbers. Be concise."""

ANSWER_USER = """Question: {question}

Evidence:
{evidence}"""
