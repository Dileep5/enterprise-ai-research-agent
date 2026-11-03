# Demo

A 10-minute walkthrough of the main features. It assumes the local setup from the README is done (`make install`, `.env`, `make seed`, `make ingest`) and the API is running (`make run`). The outputs shown come from real runs; LLM wording varies slightly between runs.

## 1. Health and indexed corpus

```bash
curl -s localhost:8000/health
# {"status":"ok","llm_configured":true,"database_reachable":true,"indexed_chunks":13,"tracing_enabled":false}
curl -s localhost:8000/documents
```

## 2. Document question with citations

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "How long do Gold members have to return items, and what about electronics?"}'
```
Expected:
- `"route": "documents"`;
- an answer stating **60 days** for Gold members and **30 days** for electronics, with `[1]`/`[2]` citations to `return_policy.txt`;
- `"grounding": {"checked": true, "numbers_unsupported": []}`.

## 3. SQL question (guarded text-to-SQL)

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "Which product category had the highest completed revenue in October 2025, and how much?"}'
```
Expected: `"route": "sql"`, **Tents, 29,462.09** `[SQL]`. The generated query is in `sql.query`, ending in `LIMIT 201`, the cap the guard injects.

## 4. Multi-source question

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "How many Gold tier customers do we have, and what extra return time does the policy give them?"}'
```
Expected:
- `"route": "both"`, with steps `route → retrieve → generate_sql → execute_sql → synthesize`;
- **44** customers `[SQL]` and **60 days** `[n]`.

## 5. Self-correction: mis-routed question

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "Which supplier has the longest lead time, and how long is it?"}'
```
Expected: steps similar to `route → generate_sql → execute_sql → … → retrieve → synthesize`. The router picks SQL, but there is no supplier table, so the graph falls back to the documents. The answer is **TrailTech Electronics, 60 days**, citing `supplier_directory.csv, rows 1-10`.

## 6. Refusal

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "What is the parental leave policy for store employees?"}'
```
Expected: `"insufficient_evidence": true`, "I could not find sufficient evidence…".

## 7. SQL security

```bash
curl -s localhost:8000/sql -H 'content-type: application/json' -d '{"sql": "DROP TABLE orders"}'
# 400 {"error":"unsafe_sql","message":"Only SELECT queries are allowed (got DROP)."}
curl -s localhost:8000/sql -H 'content-type: application/json' -d "{\"sql\": \"SELECT 'pg_authid'::regclass\"}"
# 400 {"error":"unsafe_sql","message":"Cast to catalog type 'REGCLASS' is not allowed."}
.venv/bin/python -m pytest tests/unit/test_sql_guard.py -q                     # 42 passed
TEST_DATABASE_URL=postgresql://postgres:pw@localhost:5432/retail_test \
  .venv/bin/python -m pytest tests/integration/test_postgres.py -q -k readonly  # role-level checks
```

## 8. Upload validation

```bash
curl -s -F files=@data/documents/return_policy.txt localhost:8000/documents   # "status":"duplicate"
curl -s -F "files=@Makefile;filename=bad.exe" localhost:8000/documents         # 415 unsupported_file_type
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/query -H 'content-type: application/json' -d '{"question":""}'  # 422
```

## 9. Retrieval ablation (offline, no API key)

```bash
.venv/bin/python -m scripts.evaluate --mode ablation
```
Expected for the paraphrase set (hit@1): dense 0.75, BM25 0.60, RRF 0.80, **RRF + rerank 0.85**.

## 10. MCP tool discovery

```bash
.venv/bin/python -m pytest tests/integration/test_mcp_stdio.py -q   # spawns the server, lists the 6 tools
```

## Full agent evaluation (costs a few cents of OpenAI usage)

```bash
make eval-full        # 34 questions; writes eval/results/full-*.json
```
