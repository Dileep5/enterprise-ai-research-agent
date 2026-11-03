# Troubleshooting

## Setup

| Symptom | Cause | Fix |
|---|---|---|
| `seed_db.py: error: Set READONLY_DB_PASSWORD` | No password, or the `.env.example` placeholder | Set `READONLY_DB_PASSWORD` to 8+ characters in `.env` |
| `connection to server ... failed` during `make seed` | PostgreSQL not running or wrong `DATABASE_ADMIN_URL` | Start PostgreSQL; check host, port, user. The database itself is created automatically |
| `/health` shows `"database_reachable": false` | `DATABASE_URL` wrong, or the read-only role not created | Run `make seed`; check the user and password match `READONLY_DB_USER` / `READONLY_DB_PASSWORD` |
| `/health` shows `"llm_configured": false` | `OPENAI_API_KEY` empty | Set it in `.env` or the environment. Document questions then fall back to showing passages; SQL questions return a configuration message |
| First query takes ~10–30 s | Hugging Face models downloading (~180 MB) | One-time cost. The Docker image bakes the models in. Set `HF_TOKEN` to avoid rate limits |
| `HF_TOKEN` warning in logs | Anonymous Hugging Face downloads | Harmless; set `HF_TOKEN` to silence it |
| `ModuleNotFoundError: langchain` from Langfuse | `langfuse.langchain` needs the `langchain` package | Included in `requirements.txt`; reinstall |

## Answers

| Symptom | Cause | Fix |
|---|---|---|
| "I could not find sufficient evidence…" for an answerable question | Evidence gate: the best cross-encoder score is below `MIN_RERANK_SCORE` (−9.0). Common for conversational paraphrases | Rephrase with the document's terms, or lower `MIN_RERANK_SCORE` (e.g. −10). The trade-off is measured in IMPLEMENTATION_DETAILS.md §4.6 |
| Warning "Answer contains numbers not found in the evidence" | The numeric grounding check found a number neither in the cited source nor in the question | Check the cited passage. Derived values (differences, ratios) are flagged by design |
| SQL answer looks plausible but wrong | Text-to-SQL semantic error (e.g. wrong denominator) | `sql.query` is returned in every response; inspect it. Use `mode: "documents"` if the question is about reports or policies |
| Question about a report figure went to SQL | Router misclassification | The graph falls back to the documents when SQL yields nothing usable or the LLM refuses. Force `mode` if needed |
| `400 unsafe_sql` from `/sql` | Statement rejected by `app/sql/guard.py` | Only single SELECTs on the six retail tables are allowed |
| `Query exceeded the 5000 ms statement timeout` | Expensive query (e.g. a cartesian join) | Add filters, or raise `SQL_STATEMENT_TIMEOUT_MS` |
| Changed a document but the old content still appears | IDs are content hashes; the old version still exists | `DELETE /documents/{doc_id}` for the old version, or delete `storage/chroma` and re-ingest |

## MCP

| Symptom | Fix |
|---|---|
| Client shows "server disconnected" | Run `python -m app.mcp_server` manually from the repo root; it must be started with the repo as `cwd` and the venv's Python |
| Protocol errors / garbage on stdout | Nothing may print to stdout; the server logs to stderr. Don't add `print()` to tools |
| Tools return `database_unavailable` | Set `DATABASE_URL` in the MCP client config's `env` block |

## Tests

| Symptom | Fix |
|---|---|
| 19 tests skipped | Integration tests need `TEST_DATABASE_URL` (PostgreSQL) and `RUN_MODEL_TESTS=1` (models). This is expected without them |
| `test_postgres.py` fails on role creation | `TEST_DATABASE_URL` must be an admin connection; the test database is created and re-seeded |

## Docker

| Symptom | Fix |
|---|---|
| `POSTGRES_PASSWORD` error at `docker compose up` | Copy `.env.example` to `.env` and set the passwords |
| `api` never starts | It waits for `init` (seed + ingest) to finish successfully: `docker compose logs init` |
| Port 5432 already in use | Stop the local PostgreSQL, or change the published port in `docker-compose.yml` |
| Reset everything | `docker compose down -v` (removes the database and Chroma volumes) |

## Known dependency advisories

`pip-audit` reports 4 advisories for `chromadb 1.5.9` (PYSEC-2026-311, -3813, -3814, -3815) with **no fixed release** (1.5.9 is the latest). Chroma runs embedded in-process (`PersistentClient`); no Chroma HTTP server or port is exposed. CI ignores only these IDs, so any new advisory fails the build. Re-check when chromadb releases a fix.
