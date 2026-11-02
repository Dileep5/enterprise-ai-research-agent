# Verification Report: Enterprise AI Research Agent

Date: 2026-11-02. This report lists exactly what was executed, with results. Nothing here is estimated.

## Environment

| Item | Value |
|---|---|
| Machine | Apple M3, 8 GB RAM, macOS (Darwin 27) |
| Python | 3.12.16 (uv-managed) |
| PostgreSQL | 16.15 (portable binaries, local, trust auth). Homebrew refused to run on this macOS version |
| Docker | **not available** |
| LLM | OpenAI `gpt-4o-mini` (via `OPENAI_API_KEY` from the shell) |
| Key packages | see `requirements.lock`: fastapi 0.162, langgraph 1.2.19, langchain-core 1.7.4, chromadb 1.5.9, sentence-transformers 5.7.0, sqlglot 29.0.1, psycopg 3.3.6, mcp 1.30.0, langfuse 3.15.0 |

## Commands executed

| # | Command | Result |
|---|---|---|
| 1 | `uv pip install -r requirements-dev.txt` | PASS (after adding `langchain`, which `langfuse.langchain` requires) |
| 2 | `python -m scripts.make_sample_docs` | PASS: generated PDF (3 pages) + DOCX |
| 3 | `python -m scripts.seed_db --admin-url postgresql://postgres@localhost:5432/retail` | PASS: 8 stores, 28 products, 400 customers, 2,500 orders, 6,219 items, 224 inventory rows |
| 4 | Manual check: `DELETE FROM orders` as `agent_readonly` | Rejected (`ReadOnlySqlTransaction`) as intended |
| 5 | `ruff check app scripts tests` | PASS |
| 6 | `pytest tests/unit -q` | **118 passed** |
| 7 | `TEST_DATABASE_URL=postgresql://postgres@localhost:5432/retail_test pytest tests/integration -m postgres -q` | **12 passed** |
| 8 | `RUN_MODEL_TESTS=1 pytest tests/integration -m models -q` | **7 passed** (real MiniLM embedder + cross-encoder) |
| 9 | `pytest -q` (no env vars) | 118 passed, 19 skipped (integration tests skip cleanly) |
| 10 | `pytest tests/unit --cov=app` | 86% line coverage of `app/` |
| 11 | `python -m scripts.evaluate --mode retrieval` | PASS: see `eval/results/retrieval-20261102-175217.json` |
| 12 | `python -m scripts.evaluate --mode retrieval --no-reranker` | PASS: `eval/results/retrieval-norerank-20261102-175223.json` |
| 13 | `python -m scripts.evaluate --mode full` (3 runs, see below) | PASS (0 errors in 34 questions each run) |
| 14 | `uvicorn app.main:app --port 8765` + curl: `/health`, `/documents` (2 valid files + 1 `.exe`), re-upload, `/query` (documents, SQL, out-of-scope), `/sql` with DROP, `/query` with empty question | All behaved as expected: per-file error for `.exe`, `duplicate` on re-upload, cited answer, SQL answer with query + rows, refusal, 400 `unsafe_sql`, 422 |
| 15 | MCP stdio client (`mcp.client.stdio`) → `python -m app.mcp_server`: list tools, `run_readonly_sql` (SELECT and DELETE), `search_documents` | PASS: 6 tools listed; SELECT returned rows; DELETE returned `unsafe_sql` |
| 16 | YAML parse of `docker-compose.yml` and `.github/workflows/ci.yml` | Valid YAML |

## Not executed

| Item | Reason |
|---|---|
| `docker compose up --build` / `docker build` | Docker not installed on this machine |
| GitHub Actions workflow | Repository not pushed (awaiting your approval) |
| Langfuse tracing against a live server | No Langfuse credentials; code path is wired in, but only the "disabled" path ran |

## Evaluation history (full mode, gpt-4o-mini)

The benchmark was also used to find and fix agent weaknesses, so these numbers are development results, not held-out results.

| Run file | Change before run | Routing | SQL exec | SQL match | Keyword recall | Refusal (unanswerable) | False refusal |
|---|---|---|---|---|---|---|---|
| `full-20261102-175410.json` | initial version | 0.97 | 0.93 | 0.86 | 0.84 | 1.00 | 0.10 |
| `full-20261102-175654.json` | partial-answer prompt; doc context passed to SQL generation on `both` route; SQL→documents fallback | 0.97 | 1.00 | 0.93 | 1.00 | 1.00 | 0.00 |
| `full-20261102-180035.json` (final) | prompt wording only (no "not covered" note on full answers) | 0.97 | 1.00 | 0.93 | 1.00 | 1.00 | 0.00 |

Remaining misses in the final run: `d15` (LLM routed a supplier question to SQL; recovered via the documents fallback, so the answer is correct but the route metric counts a miss) and `b01` (computed a revenue share instead of an order-count share).

## Bugs found and fixed while testing

1. `BM25Okapi` gives zero/negative IDF to terms present in ≥50% of chunks, which hides matches in small corpora. Switched to Lucene IDF.
2. The graph relied on the LLM client to raise when unconfigured. It now checks `llm.available` explicitly.
3. The `truncated` flag could never be true because the guard capped LIMIT at exactly `row_limit`. The executor now caps at `row_limit + 1`.
4. The MCP server did not apply the app's logging config, and the app's logger wrote to stdout, which would corrupt the stdio protocol. MCP now logs to stderr.
5. CI set `HF_HOME: ~/...`, which GitHub Actions does not expand. Removed.
