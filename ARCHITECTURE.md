# Architecture

## System context

```mermaid
flowchart LR
    U[HTTP client] -->|REST| API[FastAPI<br/>app/main.py, app/api/routes.py]
    M[MCP client<br/>Claude Desktop / IDE / agent] -->|stdio| MCP[MCP server<br/>app/mcp_server.py]
    API --> SVC[Services container<br/>app/services.py]
    MCP --> SVC
    SVC --> AG[LangGraph agent]
    SVC --> RET[Hybrid retriever]
    SVC --> DB[RetailDatabase]
    SVC --> ING[Ingestion pipeline]
    AG --> LLM[(OpenAI API)]
    RET --> CH[(ChromaDB<br/>embedded, on disk)]
    DB --> PG[(PostgreSQL<br/>agent_readonly role)]
    AG -.LangChain callbacks.-> LF[(Langfuse, optional)]
```

`app/services.py` is the composition root. It wires settings into concrete components exactly once. The API, the MCP server and the tests all build their dependencies there, and the tests swap in fakes: `FakeLLM`, `FakeDatabase`, `HashingEmbedder` and `OverlapReranker` in `tests/conftest.py`.

## Agent graph

```mermaid
stateDiagram-v2
    [*] --> route
    route --> retrieve: documents / both
    route --> generate_sql: sql
    retrieve --> generate_sql: both (first pass)
    retrieve --> synthesize: otherwise
    generate_sql --> execute_sql
    execute_sql --> generate_sql: retryable error and attempts < max
    execute_sql --> retrieve: route=sql and no usable rows
    execute_sql --> synthesize
    synthesize --> retrieve: route=sql and LLM refused (once)
    synthesize --> [*]
```

`AgentState` (`app/agent/graph.py`) is a `TypedDict`. `warnings` and `steps` use `Annotated[list, operator.add]` reducers, so every node appends rather than overwrites. Nodes never raise domain errors: they convert them into state (`sql_error`, `warnings`), and `synthesize` decides how to degrade.

## Retrieval pipeline

```mermaid
flowchart LR
    Q[question] --> E[embed query<br/>all-MiniLM-L6-v2]
    Q --> T[tokenize]
    E --> D[Chroma cosine top-20]
    T --> B[BM25, Lucene IDF, top-20]
    D --> F[Reciprocal Rank Fusion, k=60]
    B --> F
    F --> R[cross-encoder rerank<br/>ms-marco-MiniLM-L-6-v2]
    R --> K[top-5]
    K --> G{best rerank score >= -9.0?}
    G -->|no| X[insufficient evidence<br/>no LLM call]
    G -->|yes| S[synthesize with numbered evidence]
    S --> V{LLM says INSUFFICIENT_EVIDENCE?}
    V -->|yes| X
    V -->|no| C[validate citations + numeric grounding]
```

Refusal is two-stage:
1. **Stage 1: a cheap evidence gate.** It is calibrated for recall and rejects clearly irrelevant questions without calling the LLM.
2. **Stage 2: the LLM.** It is instructed to return `INSUFFICIENT_EVIDENCE` when the evidence does not answer the question. On the 34-question benchmark it refused all 4 unanswerable questions in 3 of 3 runs, even though the gate passed 3 of them.

## SQL safety layers

```mermaid
flowchart TB
    L[LLM-generated SQL] --> A
    subgraph A[Layer 1 - app/sql/guard.py]
        A1[sqlglot parse, postgres dialect] --> A2[exactly one SELECT / CTE / set-op]
        A2 --> A3[walk AST: no DML/DDL/COPY/SET,<br/>no SELECT INTO, no FOR UPDATE]
        A3 --> A4[function deny-list: pg_*, lo_*, dblink*, set_config, ...]
        A4 --> A5[no casts to regclass/oid/...]
        A5 --> A6[tables allow-listed, schema public only]
        A6 --> A7[LIMIT capped at row_limit+1, comments stripped]
    end
    A --> B
    subgraph B[Layer 2 - app/sql/database.py]
        B1[SET TRANSACTION READ ONLY] --> B2[SET LOCAL statement_timeout]
        B2 --> B3[execute without params; always rollback]
    end
    B --> C
    subgraph C[Layer 3 - scripts/seed_db.py]
        C1[role agent_readonly: LOGIN, SELECT only] --> C2[default_transaction_read_only = on]
        C2 --> C3[CREATE on schema public revoked from PUBLIC]
    end
```

Each layer assumes the one before it has failed. The integration tests check layer 3 on its own by connecting as the role directly and attempting a `DELETE` and a `CREATE TABLE` (`tests/integration/test_postgres.py`).

## Ingestion

```mermaid
flowchart LR
    A[bytes + filename] --> B{extension}
    B -->|pdf| P[pypdf, one section per page]
    B -->|docx| W[python-docx paragraphs + tables]
    B -->|txt / md| T[UTF-8, latin-1 fallback, NUL check]
    B -->|csv| C[pandas, 25 rows per section as 'col: value']
    B -->|other| E[UnsupportedFileTypeError -> 415]
    P & W & T & C --> S[RecursiveCharacterTextSplitter 800/120]
    S --> H[doc_id = sha256 of content, first 16 hex]
    H -->|already stored| DUP[status: duplicate]
    H -->|new| EM[embed] --> CH[(Chroma upsert)]
    CH --> V[store.version += 1 -> BM25 rebuild on next query]
```

## Deployment (Docker Compose)

```mermaid
flowchart LR
    PG[(postgres:16-alpine)] -->|healthy| INIT[init: seed_db + ingest_samples<br/>one-shot]
    INIT -->|completed_successfully| API[api: uvicorn :8000]
    INIT --- VOL[(chroma volume)]
    API --- VOL
```

The image bakes in both Hugging Face models at build time, so containers start offline. `init` and `api` share the Chroma volume.
