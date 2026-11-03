# Interview Guide: Enterprise AI Research Agent

Answers point to the code (`file:function`) and to measured results in `TEST_RESULTS.md`.

## Architecture and orchestration

**1. Walk me through what happens when a question hits `POST /query`.**
1. `routes.query` validates the request with `QueryRequest` (Pydantic v2, `extra="forbid"`, stripped question of 3–2,000 characters) and calls `ResearchAgent.run`.
2. The LangGraph `StateGraph` starts at `_route`.
3. Document questions go through `_retrieve` (hybrid search plus the evidence gate) and then `_synthesize`.
4. SQL questions go through `_generate_sql` → `_execute_sql`, with up to one retry, then `_synthesize`. `both` does retrieval first, then SQL.
5. `_synthesize` builds numbered evidence, calls the LLM, validates citations and runs the numeric grounding check.
6. The final state is mapped to `QueryResponse`: answer, route, citations, SQL details, grounding, warnings, steps and latency.

**2. Why LangGraph instead of a plain function or a LangChain agent?**
The control flow has real branches and loops: routing, an SQL retry loop that feeds the database error back, a fallback from SQL to documents, and a second fallback after an LLM refusal. LangGraph makes those edges explicit (`_after_route`, `_after_execute`, `_after_synthesize`), testable (tests assert the exact `steps` sequence) and bounded (`recursion_limit=25`, `sql_max_attempts=2`, a fallback runs at most once). A free-form ReAct agent would decide these paths at runtime, which is harder to test and to bound.

**3. How is state managed?**
`AgentState` is a `TypedDict`. Scalar keys are overwritten by node outputs. `warnings` and `steps` are `Annotated[list, operator.add]`, so each node appends. Nodes return partial dicts, never the whole state.

**4. What happens if a node fails?**
Nodes catch `AppError` subclasses and write them into state rather than raising:
- `_retrieve` → empty retrieval plus a warning;
- `_execute_sql` → `sql_error` plus `sql_retryable`;
- `_synthesize`, when the LLM fails → `_extractive_fallback` returns the passages verbatim.

Only truly unexpected exceptions escape. The API then returns a generic 500 without internals (`test_unhandled_errors_return_generic_500`).

**5. How does routing work and what if the router is wrong?**
- **The router:** `route_question` asks the LLM for JSON `{route, reason}`, and `parse_route` extracts it even from surrounding prose. If the LLM fails or returns garbage, `heuristic_route` scores weighted keyword cues.
- **Recovering from wrong routes:**
  1. An SQL route with no usable rows (`_sql_usable`) goes to `retrieve`.
  2. If the LLM refuses on an SQL-only route, `_after_synthesize` consults the documents once.
- **Measured:** d15 ("Which supplier has the longest lead time?") is routed to SQL in every run, yet answered correctly with a citation to `supplier_directory.csv`.

**6. Why is the router accuracy 0.97 and not higher?**
One benchmark question (d15) is consistently routed to SQL by `gpt-4o-mini`. I chose to recover from it in the graph rather than tune the router prompt to one question. The keyword fallback alone scores 0.91.

## Retrieval

**7. Why hybrid retrieval?**
Dense embeddings handle paraphrases; BM25 handles exact tokens such as SKUs, percentages and names. Measured on 20 paraphrased questions (chunk-level hit@1): dense 0.75, BM25 0.60, RRF 0.80, RRF+rerank 0.85; MRR 0.838 → 0.917 (`eval/results/ablation-*.json`).

**8. Explain Reciprocal Rank Fusion. Why k=60?**
`score(d) = Σ_lists 1/(k + rank_d)` (`fusion.reciprocal_rank_fusion`). It uses ranks, not raw scores, so cosine similarities and BM25 scores never need normalising onto one scale. k=60 is the constant from Cormack et al. A larger k flattens the advantage of top ranks, so documents that appear in both lists win. Duplicates within a single list are counted once.

**9. Why rerank after fusion?**
Bi-encoders and BM25 score the query and the passage independently. A cross-encoder reads them together and is more accurate, but too slow to run over the whole corpus. So it reranks only the fused top-20. On paraphrases it raised hit@1 from 0.80 to 0.85.

**10. What was wrong with `rank_bm25`'s BM25Okapi?**
Its IDF, `log((N-n+0.5)/(n+0.5))`, is zero or negative when a term appears in at least half the chunks. With two uploaded documents, a query term present in one of them scored 0, and the match vanished. `_LuceneBM25` overrides `_calc_idf` with `log(1 + …)`, which is always positive. It was caught by `test_bm25_index_rebuilds_after_new_ingestion`.

**11. How do BM25 and Chroma stay in sync?**
`ChromaVectorStore.version` increments on every write or delete. `HybridRetriever._bm25_index` rebuilds the in-memory BM25 index when the version it was built from is stale.

**12. How do you decide there is insufficient evidence?**
Refusal has two stages:
1. **The gate.** `_assess_evidence` passes only if the best cross-encoder logit is ≥ `min_rerank_score` (≥ `min_dense_similarity` when the reranker is off). Below that, `_synthesize` refuses **without calling the LLM**.
2. **The LLM.** The answer prompt requires the exact token `INSUFFICIENT_EVIDENCE` when nothing relevant was found.

**13. Why is the threshold −9.0, and how did you choose it?**
The original −3.0 looked fine on the main benchmark but passed only 30% of answerable *paraphrased* questions. I measured score distributions on 20 paraphrases and 20 in-domain unanswerable questions. They overlap; dense similarity added nothing a single threshold didn't. At −9.0 the gate passes 16/20 answerable questions and rejects 15/20 unanswerable ones before any LLM call; the LLM handles the rest. Result: 4/4 benchmark unanswerables refused in 3/3 runs, 20/20 in the robustness set, and 0 false refusals on the benchmark.

**14. You tried contextual chunk headers. What happened?**
Prefixing chunks with the document title for embedding and BM25 *lowered* dense hit@1 (0.75 → 0.70) and left hybrid+rerank unchanged, so I reverted it. A negative result is still a result, and the ablation harness is what made it visible.

**15. How are citations produced and validated?**
Evidence is numbered `[1]..[n]` with location labels (`citations.location_label`: page for PDFs, row range for CSVs). The prompt requires inline `[n]` citations. `cited_indices` extracts them; numbers outside `1..n` are dropped with a warning, and an uncited document answer gets a warning too. Each `Citation` carries the source, location, chunk ID, snippet and score.

**16. How do you check that the evidence actually supports the answer?**
At runtime, `grounding.unsupported_numbers` checks every number in a sentence that cites `[n]` against source *n* (or the question). Numbers are compared by value, so `$180,895.90` matches `180895.9`.
- **Why citation-aware:** checking against all evidence was too lenient. A test caught "25%" cited to the return policy when it actually came from the handbook.
- **In evaluation:** `citation_supports_answer` checks that the cited chunks contain the expected facts. It scored 1.0 across 3 runs, and 0.989 of answers had every number grounded. The miss was a correctly derived difference, "0.03", which the check flags by design.

## Text-to-SQL and security

**17. Describe the three SQL security layers.**
1. **`guard.validate_sql`:** an AST check with sqlglot covering:
   - exactly one SELECT, CTE or set operation;
   - no DML/DDL/COPY/SET anywhere in the tree, including inside CTEs;
   - no `SELECT INTO` and no row locks;
   - a function deny-list (`pg_*`, `lo_*`, `dblink*`, `set_config`, …);
   - no casts to catalog types (`regclass`, `oid`);
   - allow-listed tables in `public` only;
   - LIMIT capped and comments stripped.
2. **`RetailDatabase.execute`:** `SET TRANSACTION READ ONLY` plus `SET LOCAL statement_timeout`, always rolled back.
3. **The `agent_readonly` role:** SELECT-only grants, `default_transaction_read_only=on`, and `CREATE` on `public` revoked from `PUBLIC`.

Each layer assumes the others failed. Integration tests attack layer 3 directly.

**18. Why parse SQL instead of using regex?**
Regex can't tell `DELETE` in a string literal from `DELETE` in a CTE. It also misses `SELECT … INTO`, and comment tricks like `SELECT 1 /*; DROP */` defeat it. An AST check sees exactly what PostgreSQL would execute. The SQL is also re-rendered from the AST, so what runs is exactly what was validated.

**19. What did your adversarial probing find?**
Of 25 probe statements, `CAST('x' AS regclass)` got through. It resolves names against the system catalogs and so leaks whether tables exist. It is now blocked, and all 25 probes are covered by tests.

**20. What isn't stopped by the guard?**
Expensive but read-only queries: recursive CTEs, huge `OFFSET`s, cartesian joins. Those are bounded by `statement_timeout`, which is tested with a triple self-join. `SELECT current_user` is allowed; it is harmless.

**21. How does the retry loop work?**
- **Retryable errors:** `UnsafeSQLError` and `SQLExecutionError` (e.g. `column "revenue" does not exist`) set `sql_retryable=True`. `_generate_sql` then appends the previous SQL and the error (`SQL_RETRY_SUFFIX`), up to `sql_max_attempts` (2).
- **Non-retryable errors:** an unavailable database, or a missing LLM, end the loop immediately.
- **Tested:** `test_sql_error_triggers_one_retry_with_error_feedback`.

**22. How do multi-source ("both") questions work?**
Retrieval runs first. If the evidence is sufficient, the top-3 chunks go into SQL generation as *definitions only* (`SQL_CONTEXT_SUFFIX`). This came from a real failure: the handbook says "below the reorder point" (`<`), but without the context the model wrote `<=`. The answer prompt then combines `[SQL]` and `[n]` evidence and allows explicitly partial answers.

**23. What's a remaining text-to-SQL weakness?**
Semantic errors that execute successfully. In the robustness run, "What fraction of goods sold were sent back?" produced `completed/completed = 1.0`. No guard or grounding check catches a valid query that answers a different question. Mitigations: the SQL is always returned to the user; mitigations I'd add are a semantic-layer or metric definitions, or few-shot examples.

**24. Why does `truncated` work?**
The executor validates with `row_limit + 1` and fetches `row_limit + 1` rows. If more than `row_limit` come back, the result is truncated, and the response says so.

## API, reliability and operations

**25. Which timeouts and retries exist?**
| Component | Timeout / retry |
|---|---|
| OpenAI | 30 s timeout, `max_retries=2` (SDK exponential backoff) |
| PostgreSQL | 5 s connect timeout plus one reconnect with backoff; 5 s `statement_timeout` (10 s at role level) |
| SQL generation | one semantic retry |
| Graph | `recursion_limit=25` |

Worst-case latency is therefore bounded by these limits.

**26. How are malformed uploads handled?**
Each case returns a stable error code: unsupported extension → 415; empty file, header-only CSV, corrupt PDF/DOCX or binary data posing as text → 422; oversize → 413 (read with `max+1` bytes). Multi-file uploads return per-file status, and the request fails only if every file fails.

**27. Why are handlers `def` and not `async def`?**
Embedding, reranking, Chroma and psycopg calls all block. A sync `def` runs in FastAPI's thread pool. The same calls inside `async def` would block the event loop.

**28. Why build services lazily?**
`app.state.get_services` builds services on first use, so the API starts and `/health` can report `degraded` even when the models aren't downloaded yet or the database is down.

**29. How does the MCP server work and what did you have to be careful about?**
It is FastMCP with six read-only tools sharing the same services. Errors come back as `{error, message}` payloads so the calling agent can reason about them. Stdout carries JSON-RPC, so logging goes to **stderr**: the app's logger originally wrote to stdout, which would have corrupted the protocol. Discovery is tested by spawning the real subprocess (`test_mcp_stdio.py`).

**30. How is observability done?**
- **Setup:** with both Langfuse keys set, `build_callbacks` creates a LangChain `CallbackHandler`, and `run` passes it in `RunnableConfig`, so graph nodes and `ChatOpenAI` calls appear as one trace tree.
- **Failure handling:** an init failure disables tracing rather than the API, and the lifespan hook flushes buffered events on shutdown.
- **Verification:** the wiring is unit-tested; it was not run against a live Langfuse server.

## Evaluation and testing

**31. How do you test an LLM application deterministically?**
- **Fakes:** unit tests inject `FakeLLM` (scripted per prompt kind: route, SQL or answer), `HashingEmbedder`, `OverlapReranker` and `FakeDatabase`, which still applies the real SQL guard. An autouse fixture deletes API keys from the environment, so unit tests can never call a paid API.
- **Real components:** separate test marks (`postgres`, `models`) run against the real database and the real models.

**32. What does the evaluation measure, and how is it reproducible?**
- **What it measures:**
  - routing;
  - document- and chunk-level hit@k and MRR;
  - evidence-gate rates;
  - SQL execution, and result match against reference SQL (every reference value must appear in the generated result, within 0.5%);
  - keyword recall;
  - citation precision and citation support;
  - grounded numbers;
  - refusal and false-refusal rates.
- **Reproducibility:** each result file records:
  - the git commit and a dirty flag;
  - SHA-256 hashes of the benchmark files and the corpus;
  - the package versions;
  - the model, temperature, threshold and top-k.
- **Fresh index:** every run re-indexes into a temporary Chroma collection.
- **Variance:** `--repeats 3` showed zero variance on accuracy metrics at temperature 0, and a small one on grounded numbers (0.967–1.0).

**33. Are your eval numbers trustworthy?**
They are real but optimistic:
- **Small and in-sample:** the corpus has 13 chunks, the benchmark was used while developing prompts, and the robustness sets were used to calibrate the gate.
- **Honest choices:** I report each of these, keep the full history of runs in `eval/results/`, and report misses (b01, d15, p12) rather than tuning prompts to individual questions.

**34. What would you do with more time or at production scale?**
- **Search:** move BM25 to a search engine, and use a vector database with metadata filters and document ACLs.
- **Gate:** calibrate it on a larger labelled set, or train a small answerability classifier.
- **SQL:** add a semantic layer for metric definitions.
- **API:** add authentication and rate limiting.
- **Evaluation:** a larger held-out benchmark, plus nightly evals in CI with a budget.

**35. What was the hardest bug?**
The evidence gate. Every test passed and the benchmark showed zero false refusals, yet real users phrasing questions conversationally would have been refused 70% of the time. Only by building the paraphrase ablation did the problem become visible. The lesson: a benchmark written by the person who wrote the documents shares their vocabulary.
