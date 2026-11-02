# Enterprise AI Research & Data Intelligence Agent

A LangGraph agent that answers business questions from two kinds of sources:

- **Unstructured documents** (PDF, DOCX, TXT/MD, CSV) via hybrid retrieval: dense vectors (Sentence Transformers + ChromaDB) and BM25, fused with Reciprocal Rank Fusion and reranked by a cross-encoder. Answers carry citations down to the page or CSV row range.
- **Structured retail data** in PostgreSQL via guarded text-to-SQL. Every query passes an AST-level allow-list and runs in a read-only transaction under a SELECT-only role.

The agent routes each question to documents, SQL, or both. It refuses when the evidence is insufficient, and it degrades gracefully when the LLM, the vector store or the database fails. The same capabilities are exposed through a FastAPI service and an MCP server.

## Architecture

```mermaid
flowchart LR
    subgraph Clients
        U[HTTP client] --> API
        M[MCP client<br/>Claude Desktop / IDE] --> MCP
    end
    API[FastAPI<br/>app/main.py] --> AG
    MCP[MCP server<br/>app/mcp_server.py] --> AG
    MCP --> RET
    MCP --> DB

    subgraph AG[LangGraph agent - app/agent/graph.py]
        direction LR
        R{route} -->|documents| RT[retrieve]
        R -->|both| RT
        R -->|sql| GS[generate_sql]
        RT -->|both| GS
        GS --> EX[execute_sql]
        EX -->|error, retry once| GS
        EX -->|no usable rows| RT
        RT --> SY[synthesize<br/>+ citations]
        EX --> SY
    end

    RT --> RET
    subgraph RET[Hybrid retriever]
        D[Chroma dense<br/>all-MiniLM-L6-v2] --> F[RRF fusion]
        B[BM25 Lucene IDF] --> F
        F --> X[Cross-encoder rerank<br/>ms-marco-MiniLM-L-6-v2]
        X --> G{evidence<br/>threshold}
    end

    EX --> DB
    subgraph DB[SQL safety layers]
        V[sqlglot AST guard<br/>SELECT-only, table allow-list,<br/>function deny-list, LIMIT cap] --> TX[READ ONLY txn<br/>+ statement_timeout]
        TX --> PG[(PostgreSQL<br/>agent_readonly role)]
    end

    SY --> LLM[OpenAI via<br/>langchain-openai]
    GS --> LLM
    R --> LLM
    AG -.callbacks.-> LF[Langfuse<br/>optional]

    UP[POST /documents] --> ING[Loaders -> chunker -> embedder] --> D
```

### Ingestion pipeline

```mermaid
flowchart LR
    A[Upload] --> B{extension?}
    B -->|pdf| P[pypdf<br/>per-page text]
    B -->|docx| W[python-docx<br/>paragraphs + tables]
    B -->|txt/md| T[UTF-8, latin-1 fallback]
    B -->|csv| C[pandas<br/>rows as 'col: value']
    B -->|other| E[415 unsupported]
    P & W & T & C --> S[RecursiveCharacterTextSplitter]
    S --> H[sha256 doc_id<br/>duplicate check]
    H --> EM[embed] --> CH[(ChromaDB)]
```

## Features

| Area | Implementation |
|---|---|
| Orchestration | LangGraph `StateGraph` with conditional routing, an SQL retry loop that feeds the database error back to the LLM, and a fallback from SQL to documents |
| Routing | LLM classifier (JSON output) with a deterministic keyword fallback when the LLM is down or returns unparseable output |
| Retrieval | Chroma (cosine) + BM25 → RRF (k=60) → cross-encoder rerank; BM25 index rebuilt automatically after new ingestions |
| Grounding | Numbered evidence and `[n]` citations; invalid citation numbers dropped with a warning; evidence threshold short-circuits before calling the LLM; partial-answer handling for multi-source questions |
| SQL safety | sqlglot AST validation (single SELECT/CTE/set-op, no DML/DDL/`SELECT INTO`/locks, allow-listed tables, no `pg_*`/`dblink`/`set_config`, LIMIT cap), `SET TRANSACTION READ ONLY`, `statement_timeout`, and a SELECT-only role with `default_transaction_read_only` |
| Ingestion | PDF / DOCX / TXT / MD / CSV, content-addressed IDs (re-uploads are idempotent), per-file errors on multi-file upload, upload size limit |
| Errors | Typed domain errors → consistent JSON (`415`, `422`, `413`, `400`, `503`); unhandled errors return a generic 500 without leaking internals |
| MCP | `search_documents`, `list_documents`, `list_tables`, `describe_table`, `run_readonly_sql`, `ask_research_agent` (stdio transport) |
| Observability | Langfuse LangChain callback, enabled only when keys are set; a broken tracing backend cannot take the API down |
| Evaluation | `scripts/evaluate.py` + `eval/benchmark.jsonl` (34 questions): routing, hit@k/MRR, evidence gate, SQL result match against reference queries, keyword recall, citation precision, refusal accuracy |

## Quick start (local)

Requirements: Python 3.11+ and PostgreSQL 14+.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env            # set OPENAI_API_KEY, DB passwords

# Schema, deterministic demo data (2,500 orders) and the read-only role
.venv/bin/python -m scripts.seed_db
# Index the demo documents (downloads ~180 MB of models on first run)
.venv/bin/python -m scripts.ingest_samples

.venv/bin/uvicorn app.main:app --reload     # http://localhost:8000/docs
```

## Docker

```bash
cp .env.example .env            # set OPENAI_API_KEY, POSTGRES_PASSWORD, READONLY_DB_PASSWORD
docker compose up --build       # postgres -> init (seed + ingest) -> api on :8000
```

The image bakes in the embedding and reranker models, so containers start without downloading them.

## API examples

```bash
# Upload documents (multipart, multiple files allowed)
curl -F files=@data/documents/return_policy.txt -F files=@data/documents/annual_report_2025.pdf \
     http://localhost:8000/documents

# Ask: routing is automatic; "mode" can force documents | sql | both
curl -s localhost:8000/query -H 'content-type: application/json' \
     -d '{"question": "How long do Gold members have to return items?"}'

curl -s localhost:8000/query -H 'content-type: application/json' \
     -d '{"question": "What were the top 3 stores by completed revenue in 2025?"}'

# Hand-written SQL goes through the same guard
curl -s localhost:8000/sql -H 'content-type: application/json' -d '{"sql": "DROP TABLE orders"}'
# -> 400 {"error":"unsafe_sql","message":"Only SELECT queries are allowed (got DROP)."}
```

Abridged response from a live run (local PostgreSQL, `gpt-4o-mini`):

```json
{
  "answer": "Gold members have an extended return window of 60 days to return items [1]. However, electronics must be returned within 30 days ... [2].",
  "route": "documents",
  "citations": [{"id": 1, "source": "return_policy.txt", "location": "return_policy.txt", "...": "..."}],
  "insufficient_evidence": false,
  "steps": ["route", "retrieve", "synthesize"]
}
```

### Example questions

| Route | Question |
|---|---|
| documents | What restocking fee applies to opened tents returned after 14 days? |
| documents | According to the annual report, what share of revenue came from e-commerce? |
| sql | Which store had the highest revenue from completed orders? |
| sql | How many store and product combinations are below their reorder point? |
| both | How many Gold tier customers do we have, and what return window does the policy give them? |
| refused | What is the parental leave policy for store employees? |

## MCP server

```bash
.venv/bin/python -m app.mcp_server        # stdio transport
```

Example client configuration (e.g. Claude Desktop):

```json
{
  "mcpServers": {
    "research-agent": {
      "command": "/abs/path/enterprise-ai-research-agent/.venv/bin/python",
      "args": ["-m", "app.mcp_server"],
      "cwd": "/abs/path/enterprise-ai-research-agent",
      "env": {"DATABASE_URL": "postgresql://agent_readonly:...@localhost:5432/retail"}
    }
  }
}
```

## Tests

```bash
make test                                            # 118 unit tests: fakes only, no network
TEST_DATABASE_URL=postgresql://postgres:pw@localhost:5432/retail_test make test-integration   # 12 tests
make test-models                                     # 7 tests with the real HF models
```

Unit tests cover the SQL guard (33 allow/deny cases), loaders and malformed files, RRF math, BM25, hybrid retrieval, every graph path (documents, SQL, retry, both, fallback, LLM outage, DB outage, unsafe SQL), the HTTP API (validation, size limits, error codes) and the MCP tools. PostgreSQL tests check the read-only role, statement timeout, truncation and a real agent SQL round trip.

## Evaluation

```bash
make eval-retrieval     # offline: no LLM calls
make eval-full          # full agent: needs OPENAI_API_KEY + DATABASE_URL
```

Results from the runs recorded in `eval/results/` (34 questions; `gpt-4o-mini`, MiniLM embedder, MS MARCO MiniLM cross-encoder):

| Metric | Value |
|---|---|
| Retrieval hit@1 / hit@5 / MRR (20 document questions) | 1.00 / 1.00 / 1.00 |
| Evidence gate rejects unanswerable (4), with reranker / without | 4/4 / 3/4 |
| Routing accuracy, LLM router / keyword fallback | 0.97 / 0.91 |
| SQL executed successfully (14 questions with reference SQL) | 1.00 |
| SQL result matches reference query | 0.93 (13/14) |
| Answer keyword recall | 1.00 |
| Citation precision | 1.00 |
| Refusals on unanswerable questions | 4/4 |
| False refusals on answerable questions | 0/30 |
| Median end-to-end latency (local, CPU) | ~1.7 s |

**How to read these numbers:**
- The corpus is five short documents, so retrieval saturates. These results show the pipeline works; they do not show retrieval quality at scale.
- The benchmark was also used while developing the prompts (see `VERIFICATION.md`), so it is not a held-out test set.
- The single SQL miss (b01) computed an online share of *revenue* where the reference expects a share of *orders*.
- LLM outputs vary between runs.

## Project layout

```
app/
  agent/        graph.py (LangGraph), router.py, sql_generation.py, citations.py
  api/          routes.py, schemas.py (Pydantic v2)
  ingestion/    loaders.py, chunking.py, pipeline.py
  retrieval/    embeddings.py, vector_store.py, bm25.py, fusion.py, reranker.py, hybrid.py
  sql/          guard.py (sqlglot), database.py (psycopg 3)
  llm/          client.py (langchain-openai), prompts.py
  observability/tracing.py (Langfuse)
  mcp_server.py, services.py, config.py, main.py
data/documents/ demo corpus (PDF, DOCX, TXT, MD, CSV)   data/retail/schema.sql
scripts/        seed_db.py, ingest_samples.py, evaluate.py, make_sample_docs.py
eval/           benchmark.jsonl, results/
tests/          unit/, integration/
```

## Limitations

- **Docker and CI not run.** The Docker images, Compose stack and GitHub Actions workflow were written but not executed in this environment (Docker was unavailable). The same steps were run locally against a native PostgreSQL 16.
- **Langfuse not exercised.** Tracing is wired in but was never run against a live Langfuse instance; no credentials were available.
- **BM25 index is in memory.** It is rebuilt from Chroma after each write, which suits thousands of chunks, not millions.
- **No authentication on the API.** Add auth before exposing it beyond localhost.
- **Text-to-SQL depends on the LLM.** Without an API key, SQL questions return a configuration message. Document questions fall back to showing the retrieved passages.
- **All data is fictional.** The company (Maple Ridge Outfitters) and its documents were made up for the demo.
