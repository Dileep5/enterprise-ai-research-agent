# Enterprise AI Research & Data Intelligence Agent

A LangGraph agent that answers business questions from **documents** (hybrid RAG with citations) and from a **PostgreSQL** retail database (guarded text-to-SQL), or from both at once. It refuses when the evidence is insufficient, flags numbers the evidence does not support, and recovers from mis-routing, bad SQL and dependency outages. It is served through FastAPI and an MCP server.

```mermaid
flowchart LR
    Q[Question] --> R{LangGraph router}
    R -->|documents| H[Hybrid retrieval<br/>Chroma + BM25 → RRF → cross-encoder]
    R -->|sql| S[Text-to-SQL<br/>sqlglot guard → read-only txn → SELECT-only role]
    R -->|both| H --> S
    S -->|error: retry once| S
    S -->|no usable rows / LLM refused| H
    H --> G{evidence gate}
    G -->|pass| A[LLM answer with citations]
    S --> A
    A --> V[citation + numeric grounding checks]
    G -->|fail| X[refuse]
```

More detail: [ARCHITECTURE.md](ARCHITECTURE.md) · [IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md) · [TEST_RESULTS.md](TEST_RESULTS.md) · [TROUBLESHOOTING.md](TROUBLESHOOTING.md) · [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md)

## Highlights (all measured; see TEST_RESULTS.md)

- **Hybrid retrieval that earns its complexity.** On 20 paraphrased questions, chunk-level hit@1 is 0.75 dense-only, 0.60 BM25-only, 0.80 with RRF and **0.85** with RRF plus cross-encoder; MRR is 0.838 → **0.917**.
- **Two-stage refusal.**
  - **Stage 1, the evidence gate:** a cross-encoder threshold calibrated for recall. The original value refused 70% of answerable paraphrases; the calibrated one refuses 20%.
  - **Stage 2, the LLM:** asked to refuse explicitly when the evidence doesn't answer the question.
  - **Results:** 4/4 unanswerable benchmark questions refused in 3/3 runs, 20/20 in the robustness set, and 0 false refusals on the benchmark.
- **Three-layer SQL security.**
  - **Layer 1:** a sqlglot AST guard. It survived a 25-statement adversarial probe once a `regclass` gap the probe found was closed.
  - **Layer 2:** read-only transactions with a statement timeout.
  - **Layer 3:** a SELECT-only database role.
- **Grounding checks.**
  - **Citation support:** the cited chunks contain the expected facts in 100% of benchmark answers.
  - **Numbers:** a citation-aware check confirms every number in a sentence appears in the source that sentence cites.
- **Self-correcting graph:** SQL errors are fed back to the LLM for one retry, and the graph falls back from SQL to documents when the router guessed wrong.

## Prerequisites

- Python 3.11 or 3.12
- PostgreSQL 14+ (local), **or** Docker with Compose
- An OpenAI API key (needed for SQL generation and answer synthesis; retrieval and tests run without one)
- About 2 GB of disk for dependencies and ~180 MB for the Hugging Face models (downloaded on first use)

## Run locally

```bash
git clone <repo-url> enterprise-ai-research-agent && cd enterprise-ai-research-agent
make install                    # python3.12 -m venv .venv + pip install -r requirements-dev.txt
cp .env.example .env            # set OPENAI_API_KEY, DATABASE_URL, DATABASE_ADMIN_URL, READONLY_DB_PASSWORD
make seed                       # creates the DB if missing, schema, 2,500 demo orders, read-only role
make ingest                     # indexes data/documents/ (5 files: PDF, DOCX, TXT, MD, CSV)
make run                        # http://localhost:8000/docs
```

**Stop:** Ctrl-C. **Clean up:** `rm -rf storage/` (vector index), `dropdb retail` (demo database).

## Run with Docker

```bash
cp .env.example .env            # set OPENAI_API_KEY, POSTGRES_PASSWORD, READONLY_DB_PASSWORD
docker compose up --build       # postgres → init (seed + ingest, one-shot) → api on :8000
docker compose down             # stop
docker compose down -v          # stop and delete database + vector-index volumes
```

> The Compose stack and Dockerfile were **not executed** during development; Docker was unavailable. The base images and the Linux dependency resolution were verified instead. See TEST_RESULTS.md.

## Example requests

```bash
curl -s localhost:8000/health

curl -F files=@data/documents/return_policy.txt -F files=@data/documents/annual_report_2025.pdf \
     localhost:8000/documents

curl -s localhost:8000/query -H 'content-type: application/json' \
     -d '{"question": "How many Gold tier customers do we have, and what extra return time does the policy give them?"}'
```

Response from a real run (abridged):

```json
{
  "answer": "There are 44 Gold tier customers [SQL]. Gold tier loyalty members receive an extended return window of 60 days [1][4].",
  "route": "both",
  "citations": [{"id": 1, "location": "return_policy.txt"}, {"id": 4, "location": "loyalty_program.docx"}],
  "sql": {"query": "SELECT COUNT(*) AS gold_tier_customer_count FROM customers WHERE loyalty_tier = 'Gold' LIMIT 201", "rows": [[44]]},
  "grounding": {"checked": true, "numbers_unsupported": []},
  "steps": ["route", "retrieve", "generate_sql", "execute_sql", "synthesize"]
}
```

| Try | Expected path |
|---|---|
| What restocking fee applies to opened tents returned after 14 days? | documents, cites return_policy.txt |
| Which product category had the highest completed revenue in October 2025? | sql |
| Which supplier has the longest lead time? | sql → no usable rows → documents (self-correction) |
| Compare our 2025 online order share with the e-commerce share in the annual report | both |
| What is the parental leave policy? | refused |
| `POST /sql {"sql": "DROP TABLE orders"}` | 400 `unsafe_sql` |

## MCP server

```bash
.venv/bin/python -m app.mcp_server     # stdio; tools: search_documents, list_documents, list_tables,
                                       # describe_table, run_readonly_sql, ask_research_agent
```

```json
{"mcpServers": {"research-agent": {
  "command": "/abs/path/.venv/bin/python", "args": ["-m", "app.mcp_server"],
  "cwd": "/abs/path/enterprise-ai-research-agent",
  "env": {"DATABASE_URL": "postgresql://agent_readonly:...@localhost:5432/retail"}}}}
```

## Tests and evaluation

```bash
make lint
make test                                                     # 140 unit tests, no network
TEST_DATABASE_URL=postgresql://postgres:pw@localhost:5432/retail_test make test-integration   # 13
make test-models                                              # 7, real Hugging Face models
make eval-retrieval                                           # offline retrieval metrics
.venv/bin/python -m scripts.evaluate --mode ablation          # retriever ablation
make eval-full                                                # full agent (OpenAI + DB)
.venv/bin/python -m scripts.evaluate --mode full --repeats 3  # variance across runs
```

Latest results (commit `9ec1f00`): **160 passed, 0 failed**. Full agent on the 34-question benchmark, 3 runs:

| Metric | Result |
|---|---|
| SQL executed | 14/14 |
| SQL result matches reference | 13/14 |
| Keyword recall | 1.00 |
| Citation support | 1.00 |
| Unanswerable refused | 4/4 |
| False refusals | 0 |
| Median latency | 1.9 s |

Caveats: the corpus is small (13 chunks), and the benchmarks were used during development. TEST_RESULTS.md has the details and the history.

## Project layout

```
app/agent/          graph.py (LangGraph), router.py, sql_generation.py, citations.py, grounding.py
app/retrieval/      embeddings, vector_store (Chroma), bm25, fusion (RRF), reranker, hybrid
app/sql/            guard.py (sqlglot), database.py (psycopg 3, read-only execution)
app/ingestion/      loaders (PDF/DOCX/TXT/MD/CSV), chunking, pipeline
app/api/            routes.py, schemas.py      app/mcp_server.py      app/observability/tracing.py
scripts/            seed_db.py, ingest_samples.py, evaluate.py, make_sample_docs.py
eval/               benchmark.jsonl (34), retrieval_paraphrase.jsonl (20), calibration_unanswerable.jsonl (20), results/
tests/              unit/ (fakes), integration/ (PostgreSQL, MCP stdio, real models)
```

## Limitations

- **Small, in-sample evaluation.** Five demo documents (13 chunks); the benchmarks were used during development.
- **Evidence gate recall.** Some conversational paraphrases are still refused (4/20 in the robustness set).
- **Silent SQL semantic errors.** Text-to-SQL can return a valid query that answers a different question. The query is always returned for inspection.
- **Not executed:** Docker, the GitHub Actions workflow and live Langfuse tracing.
- **No API authentication** and an in-memory BM25 index, both fine for a demo, not for production.
- **Unfixed advisories:** `chromadb` 1.5.9 has advisories with no fixed release yet; see TROUBLESHOOTING.md.

## License

MIT. See [LICENSE](LICENSE).
