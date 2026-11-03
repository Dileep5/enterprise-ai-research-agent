# Test Results

Every entry below was executed. Nothing is estimated or carried over without being re-run. Timestamps are UTC.

## Environment

| Item | Value |
|---|---|
| Machine | Apple M3, 8 GB RAM, macOS (Darwin 27) |
| Python | 3.12.16 |
| PostgreSQL | 16.15 (portable binaries, localhost:5432, trust auth) |
| Docker | **not installed**, so no container was built or run (see "Not verified") |
| LLM | OpenAI `gpt-4o-mini`, temperature 0 |
| Key packages | `requirements.lock`: fastapi 0.162.0, langgraph 1.2.19, langchain-openai 1.7.3, chromadb 1.5.9, sentence-transformers 5.7.0, sqlglot 29.0.1, psycopg 3.3.6, mcp 1.30.0, langfuse 3.15.0, pytest 9.2.0 |

## 1. Final test pass (commit `9ec1f00`)

```
[2026-11-03T00:09:03Z] $ ruff check app scripts tests
All checks passed!
[2026-11-03T00:09:03Z] $ pytest tests/unit -q
140 passed in 2.27s
[2026-11-03T00:09:11Z] $ TEST_DATABASE_URL=postgresql://postgres@localhost:5432/retail_test pytest tests/integration -m 'not models' -q
13 passed, 7 deselected in 2.80s
[2026-11-03T00:09:15Z] $ RUN_MODEL_TESTS=1 pytest tests/integration -m models -q
7 passed, 13 deselected in 9.57s
[2026-11-03T00:09:26Z] $ TEST_DATABASE_URL=... RUN_MODEL_TESTS=1 pytest -q
160 passed in 11.80s
[2026-11-03T00:09:42Z] $ pytest -q            # no DB / models configured
141 passed, 19 skipped in 2.93s
[2026-11-03T00:09:50Z] $ pytest tests/unit --cov=app
TOTAL  1283 statements, 183 missed, 86% line coverage
[2026-11-03T00:10:04Z] $ uv pip check
All installed packages are compatible
```

| Suite | Passed | Failed | Skipped |
|---|---|---|---|
| Unit (fakes, no network) | 140 | 0 | 0 |
| Integration: PostgreSQL (12) + MCP stdio (1) | 13 | 0 | 0 |
| Integration: real Hugging Face models | 7 | 0 | 0 |
| **Total** | **160** | **0** | **0** |

**Baseline before this audit** (re-run at 2026-11-02 ~23:40Z, commit `d18d661`): 137 passed, 0 failed.

The audit added 23 tests:
- 10 SQL-guard attack cases;
- 7 numeric-grounding tests;
- 3 Langfuse wiring tests;
- 1 MCP stdio discovery test;
- 2 agent tests (the SQL→documents fallback and evidence-to-SQL context).

## 2. Evaluation (commit `8840310`)

| Run | File | Result |
|---|---|---|
| Retrieval (offline) | `eval/results/retrieval-20261102-185327.json` | hit@1 = hit@5 = MRR = 1.0 on 20 doc questions; chunk-level answer hit@1 1.0; gate passes 20/20 answerable, rejects 1/4 unanswerable (the rest go to the LLM) |
| Ablation (offline) | `eval/results/ablation-20261102-185341.json` | see table below |
| Full, 34 questions × 3 repeats | `eval/results/full-x3-20261102-185718.json` | see table below |
| Full, robustness set (40) | `eval/results/full-20261102-185819.json` | see table below |

### Retriever ablation (chunk-level, top-5)

| Set | Dense | BM25 | RRF | RRF + rerank |
|---|---|---|---|---|
| Benchmark doc questions (19), hit@1 | 0.947 | 0.947 | 1.000 | 1.000 |
| Paraphrases (20), hit@1 | 0.75 | 0.60 | 0.80 | **0.85** |
| Paraphrases (20), hit@5 | 0.95 | 0.80 | 1.00 | 1.00 |
| Paraphrases (20), MRR | 0.838 | 0.679 | 0.888 | **0.917** |

### Full agent: 34-question benchmark, 3 runs (mean; min–max shown only where runs differed)

| Metric | Value |
|---|---|
| Errors | 0 |
| Routing accuracy | 0.971 (33/34; `d15` routed to SQL, then answered correctly via the documents fallback) |
| SQL executed | 1.00 (14/14) |
| SQL result matches reference query | 0.929 (13/14; `b01` computed a revenue share instead of an order share) |
| Answer keyword recall | 1.00 |
| Citation precision | 1.00 |
| Citation support (cited chunks contain the expected facts) | 1.00 |
| Answers with all numbers grounded | 0.989 (0.967–1.0). One run's `b01` stated a derived difference ("0.03") |
| Unanswerable refused | 1.00 (4/4 in every run) |
| Answerable wrongly refused | 0.00 |
| Median latency | 1.90 s (1.82–1.94) |

### Full agent: robustness set (20 paraphrases + 20 in-domain unanswerable)

| Metric | Value |
|---|---|
| Unanswerable refused | 1.00 (20/20) |
| Answerable wrongly refused | 0.20 (4/20: p04, p05, p19 rejected by the gate; p11 routed to SQL and its document fallback was also below the gate) |
| Keyword recall on answered questions | 0.75 |
| Citation precision / support / grounded numbers | 1.00 / 1.00 / 1.00 |
| Known wrong answer | p12: routed to SQL, which computed `completed/completed = 1.0` as a "return fraction". This is a semantic text-to-SQL error that no runtime check catches |

**Caveat:** the paraphrase and unanswerable sets were used to calibrate the gate, so robustness numbers are in-sample. The main benchmark was not used for calibration, but it was used while developing the prompts.

### Evaluation history

| File | Change before run | SQL match | False refusals (benchmark) | Paraphrase gate pass |
|---|---|---|---|---|
| `full-20261102-175410` | initial | 0.86 | 0.10 | not measured |
| `full-20261102-175654` | partial-answer prompt, doc→SQL context, SQL→docs fallback | 0.93 | 0.00 | not measured |
| `full-20261102-180035` | prompt wording | 0.93 | 0.00 | 0.30 (measured later at −3.0) |
| `full-x3-20261102-185113` | gate −3.0 → −9.0, guard + grounding | 0.93 | 0.00 | 0.80 |
| `full-x3-20261102-185718` | SQL-refusal → documents fallback | 0.93 | 0.00 | 0.80 |

## 3. Live API checks (2026-11-02, local server, real OpenAI + PostgreSQL)

| Request | Observed |
|---|---|
| `POST /documents` with all 5 sample files | 5 × `ingested` (3, 2, 3, 3, 2 chunks) |
| "Which product category had the highest completed revenue in October 2025, and how much?" | route `sql`; "Tents … $29,462.09 [SQL]"; independently confirmed with a hand-written query (Tents 29462.09); grounding: no unsupported numbers |
| "How many Gold tier customers do we have, and what extra return time does the policy give them?" | route `both`; "44 Gold tier customers [SQL] … 60 days [1][4]"; cites return_policy.txt and loyalty_program.docx; 44 confirmed with SQL |
| "My boots fell apart after two months, what are my options?" | refused by the evidence gate (known recall limit, = p04) |
| Fresh clone: "Which supplier has the longest lead time, and how long is it?" | steps `route → generate_sql → execute_sql → generate_sql → execute_sql → retrieve → synthesize`; "TrailTech Electronics … 60 days [1]" citing `supplier_directory.csv, rows 1-10` |

Earlier live checks (first build) also covered:
- upload of a `.exe` → per-file error;
- re-upload → `duplicate`;
- `POST /sql` with `DROP` → 400 `unsafe_sql`;
- empty question → 422.

MCP was checked with a real `mcp.client.stdio` session: listing tools, a SELECT, a DELETE (rejected) and a search.

## 4. Fresh-clone reproduction (2026-11-03, `git clone` into an empty directory)

| Step (README commands) | Result |
|---|---|
| `make install` (plain pip) | exit 0 |
| `cp .env.example .env` + set DB URLs / read-only password | — |
| `make seed` | **failed** at first: database did not exist. Fixed in `6d9325a` (seed creates it); after the fix: created DB, 2,500 orders, role ready |
| `make ingest` | 5 documents ingested |
| `make lint` / `make test` | clean / 140 passed |
| `make test-integration` | 13 passed |
| `make test-models` | 7 passed |
| `uvicorn` + `/health` + `/query` | `status: ok`, correct cited answer |

## 5. Security and dependency checks

| Check | Result |
|---|---|
| SQL guard adversarial probe (25 statements) | Found the `regclass` cast gap → fixed + tested |
| `pip check` | compatible |
| `pip-audit -r requirements.lock` | pytest 8.4.2 advisory → upgraded to 9.2.0. chromadb 1.5.9: 4 advisories with **no fixed release** (1.5.9 is the latest); accepted, see TROUBLESHOOTING.md. The CI step ignores only those 4 IDs |
| Secret scan (`sk-…` patterns, `.env` files) | none committed |
| Docker base images (Docker Hub API) | `python:3.12-slim`, `postgres:16-alpine` exist |
| Linux dependency resolution (`uv pip compile --python-platform` x86_64 + aarch64) | resolves (147 pins, `torch==2.15.0+cpu`) |

## Not verified

- **Docker:** `docker build` and `docker compose up` were never run, because Docker is not installed on this machine. Base images and Linux dependency resolution were checked instead.
- **GitHub Actions:** the workflow has not run; the repository is not pushed. Every command in it was run locally.
- **Langfuse:** no live server was available. Handler construction and callback propagation are unit-tested.
