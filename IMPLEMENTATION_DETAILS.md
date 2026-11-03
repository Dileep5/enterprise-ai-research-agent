# Implementation Details

This document walks through every major module, the important functions, the decisions behind them, how failures are handled, and the trade-offs that were accepted. File references are relative to the repository root.

## 1. Configuration: `app/config.py`

- `Settings` is a Pydantic v2 `BaseSettings` that reads environment variables and `.env`. Field constraints (`ge`/`le`) reject nonsense values at startup, and a validator enforces `chunk_overlap < chunk_size`.
- Secrets (`openai_api_key`, `readonly_db_password`, `langfuse_secret_key`) are `SecretStr`, so they never appear in reprs or logs.
- `readonly_db_password` has **no default**. `scripts/seed_db.py` refuses to create the role with an empty or placeholder password (`change-me…`).
- `min_rerank_score = -9.0` is a calibrated value; see section 4.6.

## 2. Ingestion: `app/ingestion/`

| Function | What it does |
|---|---|
| `loaders.file_type_for` | Maps an extension to a loader; unknown types raise `UnsupportedFileTypeError` (HTTP 415). |
| `loaders.load_bytes` | Dispatches to `_load_pdf` / `_load_docx` / `_load_text` / `_load_csv`, wraps parser exceptions in `DocumentParseError` (422), and raises `EmptyDocumentError` when no text was extracted. |
| `_load_pdf` | One `Section` per page with `{"page": n}` metadata, which makes page-level citations possible. Encrypted PDFs are rejected. |
| `_load_docx` | Paragraphs plus table rows joined with `|`, so tabular facts (e.g. tier thresholds) are searchable. |
| `_load_text` | Decodes UTF-8 with a latin-1 fallback; a NUL byte means a binary file disguised as text. |
| `_load_csv` | Renders each row as `col: value; …` in groups of 25 rows with `row_start`/`row_end` metadata. Column names give both BM25 and the embedder the context a raw CSV line lacks. |
| `chunking.chunk_sections` | `RecursiveCharacterTextSplitter` (800 characters, 120 overlap). Chunk IDs are `{doc_id}:{index}`. |
| `pipeline.IngestionPipeline.ingest_bytes` | `doc_id = sha256(content)[:16]`; already-stored content returns `status="duplicate"` without re-embedding (idempotent uploads). The client-supplied filename is reduced to its basename. |

**Trade-off:** content addressing means renaming a file and uploading it again is still a duplicate. Changing the content creates a new document, but the old version is not removed automatically. `DELETE /documents/{doc_id}` exists for that.

## 3. Vector store and BM25: `app/retrieval/vector_store.py`, `bm25.py`

- `ChromaVectorStore` wraps a persistent Chroma collection created with `hnsw:space=cosine` and `embedding_function=None`. Embeddings always come from our own `Embedder`, so Chroma never downloads its default model. Distances are converted to cosine similarity (`1 - distance`).
- Chroma metadata must be scalar, so `_clean_metadata` drops `None` values.
- `version` increments on every write. `HybridRetriever._bm25_index` compares it to the BM25 index's version and rebuilds lazily.
- `_LuceneBM25` overrides `rank_bm25`'s IDF with Lucene's `log(1 + (N - n + 0.5)/(n + 0.5))`. The Okapi IDF is zero or negative for terms in at least half the chunks. With two uploaded files that hid real matches; a unit test (`test_bm25_index_rebuilds_after_new_ingestion`) caught it.
- **Trade-off:** BM25 lives in memory and is rebuilt from Chroma. That is fine for thousands of chunks. At larger scale, use a search engine (OpenSearch, or Postgres full-text search).

## 4. Hybrid retrieval: `app/retrieval/hybrid.py`

`HybridRetriever.retrieve(query)`:
1. Gets the dense top-`retrieval_candidates` (20) from Chroma and the BM25 top-20.
2. Fuses them with `reciprocal_rank_fusion` (`fusion.py`): `score(d) = Σ 1/(k + rank)` with `k=60`. It uses ranks rather than scores, so it needs no score normalisation between cosine values and BM25 values. Duplicates within one list are ignored.
3. Reranks the fused candidates with the cross-encoder (`reranker.py`, raw logits).
4. Returns the top-5 plus `sufficient`/`reason` from `_assess_evidence`.

### 4.1 Measured value of each stage
Measured with `python -m scripts.evaluate --mode ablation` (`eval/results/ablation-20261102-185341.json`): chunk-level answer retrieval on 20 paraphrased questions (`eval/retrieval_paraphrase.jsonl`):

| Variant | hit@1 | hit@5 | MRR |
|---|---|---|---|
| Dense only | 0.75 | 0.95 | 0.838 |
| BM25 only | 0.60 | 0.80 | 0.679 |
| Dense + BM25 (RRF) | 0.80 | 1.00 | 0.888 |
| RRF + cross-encoder | **0.85** | **1.00** | **0.917** |

On the main benchmark's document questions every variant scores 0.95 to 1.0, because those questions reuse the documents' wording. The paraphrase set exists to make the comparison informative.

### 4.2 Negative result: contextual chunk headers
I tried prefixing each chunk with its document title for embedding and BM25. On the paraphrase set, dense hit@1 *dropped* from 0.75 to 0.70, and hybrid+rerank was unchanged at 0.85. The change was reverted rather than kept for show.

### 4.3–4.6 Evidence gate calibration
The original threshold (`min_rerank_score = -3.0`) passed only **30%** of answerable paraphrased questions; MS-MARCO cross-encoder scores are low for conversational phrasing. I measured score distributions on two sets:
- the 20 paraphrases (answerable);
- 20 in-domain unanswerable questions (`eval/calibration_unanswerable.jsonl`).

| Rule | Answerable passed | Unanswerable rejected |
|---|---|---|
| rerank ≥ −3 | 6/20 | 18/20 |
| rerank ≥ −9 | 16/20 | 15/20 |
| rerank ≥ −10 | 18/20 | 10/20 |
| best "rerank OR dense similarity" combination | 16/20 | 15/20 |

The distributions overlap. For example, "What are the store hours on Black Friday?" scores high because it sits next to the real opening hours. Adding dense similarity did not beat a single threshold, so the simpler rule won: **−9.0**. Final precision comes from stage 2, the LLM's `INSUFFICIENT_EVIDENCE` instruction. False refusals cannot be recovered, while a false pass costs one LLM call. On the untouched 34-question benchmark the gate rejects only 1 of 4 unanswerable questions, and the LLM refused all 4 in all 3 repeated runs.

## 5. SQL layer: `app/sql/`

### `guard.validate_sql(sql, row_limit)`
1. Parses with sqlglot (postgres dialect). A parse failure raises `UnsafeSQLError`.
2. Requires exactly one statement of type `Select`, `Union`, `Intersect` or `Except`. CTEs parse as `Select`.
3. Walks every AST node (`find_all(exp.Expression)`) and rejects:
   - DML/DDL/`Command`/`Copy`/`Set`/transaction-control nodes, including DML hidden inside CTEs (`WITH d AS (DELETE …)`);
   - `SELECT … INTO` and `FOR UPDATE`/`FOR SHARE` locks;
   - functions named in, or prefixed by, the deny-list (`pg_*`, `lo_*`, `dblink*`, `set_config`, `current_setting`, `txid_current`, `nextval`, …);
   - casts to catalog types (`regclass`, `oid`, …). `'pg_authid'::regclass` would otherwise confirm that a catalog table exists.
4. Allows tables only from `ALLOWED_TABLES`, only in schema `public`. CTE names are exempt; table functions are rejected.
5. Re-renders the SQL from the AST with comments stripped and the LIMIT capped. The executor passes `row_limit + 1` so truncation can be detected.

### `database.RetailDatabase.execute`
- Runs `SET TRANSACTION READ ONLY` and then `SET LOCAL statement_timeout` (a literal composed with `psycopg.sql`, never string-interpolated).
- Executes without parameters, so psycopg does not interpret `%` in `LIKE '%x%'`.
- Always rolls back.
- Maps errors to domain types: `QueryCanceled` becomes an "exceeded timeout" message; `InsufficientPrivilege` becomes "role is read-only"; `OperationalError` becomes `DatabaseUnavailableError` (non-retryable); other `psycopg.Error`s become `SQLExecutionError` carrying the server's primary message (retryable, fed back to the LLM).
- `_connect` retries once (0.5 s backoff) on transient connection failure, with a 5 s connect timeout.
- Values are converted to JSON-safe types (`Decimal` → float, dates → ISO strings).

### Read-only role: `scripts/seed_db._ensure_readonly_role`
LOGIN with no superuser/createdb/createrole; `default_transaction_read_only = on`; a role-level `statement_timeout`; `REVOKE ALL ON SCHEMA public` followed by `GRANT USAGE` and `GRANT SELECT ON ALL TABLES`; and `REVOKE CREATE ON SCHEMA public FROM PUBLIC`. The last one matters on PostgreSQL < 15, where `PUBLIC` can create objects in `public` by default.

**Bounded by timeout, not by the guard:** recursive CTEs, huge `OFFSET`s and cartesian joins are allowed through the guard and stopped by `statement_timeout` (tested with a triple self-join of `order_items`).

## 6. LLM client and prompts: `app/llm/`

- `OpenAIChatLLM` wraps LangChain `ChatOpenAI` with a timeout (30 s), `max_retries=2` (the OpenAI SDK's exponential backoff) and temperature 0. Each failure mode maps to a user-safe `LLMError`: authentication, rate limit/quota, timeout/connection, and other API errors.
- `available` is false without an API key. The graph checks it explicitly instead of relying on the client to raise.
- Prompts live in `prompts.py`. The answer prompt requires `[n]`/`[SQL]` citations and allows partial answers. It asks for the exact token `INSUFFICIENT_EVIDENCE` only when nothing in the evidence is relevant.

## 7. Agent: `app/agent/`

| Node / function | Behaviour |
|---|---|
| `_route` | Uses `mode` when the request forces one; otherwise `route_question`: LLM JSON classification, parsed by `parse_route`, which tolerates surrounding prose. Unparseable output or an `AppError` falls back to `heuristic_route` (weighted keyword cues). |
| `_retrieve` | Calls the retriever; an `AppError` becomes an empty, insufficient retrieval plus a warning. |
| `_generate_sql` | Uses the schema from `information_schema` (cached). For `both` routes with sufficient evidence, the top-3 chunks go in as "document context" so business definitions (e.g. "below the reorder point" → `<`) shape the query. On retry, the previous SQL and the database error are appended. |
| `_execute_sql` | `UnsafeSQLError` and `SQLExecutionError` are retryable; `DatabaseUnavailableError` is not. |
| `_after_execute` | Retries while attempts < `sql_max_attempts` (2). If an SQL route produced no usable rows, it routes to `retrieve`. |
| `_synthesize` | Short-circuits to refusal when there is no sufficient evidence (no LLM call). If the LLM is unavailable, it falls back to the evidence verbatim (`_extractive_fallback`). It drops citation numbers that don't exist (with a warning) and runs `unsupported_numbers`. |
| `_after_synthesize` | If the LLM refused on an SQL-only route, it consults the documents once before refusing. This fixed a measured case where the router sent an annual-report question to SQL. |

### Numeric grounding: `app/agent/grounding.py`
`unsupported_numbers(answer, evidence, question, sources)` splits the answer into sentences. A sentence citing `[n]` must find each of its numbers (compared by value, so `180,895.90` equals `180895.9`) in source *n* or in the question. Uncited sentences are checked against all evidence. Small integers (<10) are ignored because they are list markers and ordinals. Results appear in `QueryResponse.grounding` and `warnings`.
- **Why citation-aware:** the first version checked against *all* evidence and missed a test case in which "25%" was cited to the return policy but actually came from the handbook.
- **Limitation:** derived values such as a computed difference ("0.03 percentage points") are flagged even when the arithmetic is right. That is intentional; they need a human check.

## 8. API: `app/api/`, `app/main.py`

- Handlers are synchronous `def`s, so FastAPI runs them in a thread pool; the model, Chroma and database calls block.
- Pydantic models use `extra="forbid"` on requests. `question` is stripped and must contain at least 3 non-whitespace characters, up to 2,000 characters.
- Uploads are read with `max_bytes + 1` to detect oversize files without reading everything. Multi-file uploads return per-file status. A single failing file returns its specific error code; all files failing returns 400.
- Exception handlers map `AppError` to `{error, message, details}` with the error's status code. Any other exception returns a generic 500 without internal details (tested).
- Services are built lazily on first request (`app.state.get_services`), so the app boots even before the models or the database are reachable. `/health` reports `degraded` when the database is down.
- The lifespan hook flushes Langfuse on shutdown.

## 9. MCP server: `app/mcp_server.py`

FastMCP with six read-only tools. Tool errors are returned as `{error, message}` payloads rather than exceptions, so the agent on the other end can reason about them. Logging is forced to **stderr** because stdout carries the stdio protocol. `top_k` is clamped to 1–20. Discovery is tested by spawning the real server as a subprocess (`tests/integration/test_mcp_stdio.py`).

## 10. Observability: `app/observability/tracing.py`

When both Langfuse keys are set, `build_callbacks` creates the Langfuse v3 client and a LangChain `CallbackHandler`. The agent passes `callbacks` in the LangGraph `RunnableConfig`, so graph nodes and `ChatOpenAI` calls are traced as one tree (`run_name="research-agent"`). Any import or initialisation error disables tracing with a warning, so the API keeps running. **Not verified against a live Langfuse server** (no credentials were available); tests cover the handler construction and the wiring.

## 11. Evaluation: `scripts/evaluate.py`

| Mode | Needs | Measures |
|---|---|---|
| `retrieval` | nothing | heuristic routing; document hit@k/MRR; chunk-level answer hit@k; evidence-gate pass/reject rates |
| `ablation` | nothing | dense / BM25 / RRF / RRF+rerank on the benchmark and paraphrase sets |
| `full` | OpenAI + DB | routing, SQL execution, SQL result match against a reference query, keyword recall, citation precision, **citation support** (do the cited chunks contain the expected facts?), **grounded numbers**, refusal accuracy, false refusals, latency |

- **Reproducibility:** every result file records the git commit, a dirty flag (ignoring `eval/results`), SHA-256 of every benchmark file and of the corpus, the Python and key package versions, the model, temperature, threshold and top-k.
- **Variance:** `--repeats N` aggregates the mean with min/max.
- **Fresh index:** the corpus is re-indexed into a temporary Chroma collection on every run, so results never depend on local state.
- **Caveat:** the prompts and the threshold were developed against these benchmarks, so the results are not held-out estimates.

## 12. Failure handling summary

| Failure | Behaviour |
|---|---|
| OpenAI down / rate-limited / bad key | Router → keyword heuristic; SQL → explicit error; answer → extractive fallback with citations |
| No API key | Same as above; `/health` shows `llm_configured: false` |
| PostgreSQL down | One connect retry; SQL step reports "unavailable" (no retry loop); `/health` degraded; documents still work |
| Bad generated SQL | Error text fed back to the LLM for one retry; unsafe SQL is never executed |
| Slow query | `statement_timeout` → readable error |
| Chroma failure | `VectorStoreError` → retrieval warning; SQL still works |
| Malformed upload | 415 / 422 / 413 with a stable error code |
| Unexpected exception | Generic 500; details only in server logs |
