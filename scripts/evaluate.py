"""Evaluate routing, retrieval, SQL and answers on eval/benchmark.jsonl.

Modes
  --mode retrieval   Offline: no LLM calls. Heuristic routing accuracy, retrieval
                     hit@1 / hit@k / MRR on source documents, and evidence-gate accuracy.
  --mode full        Runs the whole LangGraph agent (needs OPENAI_API_KEY and DATABASE_URL).
                     Adds LLM routing accuracy, SQL execution success, SQL result match
                     against reference queries, answer keyword recall, citation precision
                     and refusal accuracy on unanswerable questions.

Every number is computed from this run and written to eval/results/. Nothing is hard-coded.

Usage:
  python -m scripts.evaluate --mode retrieval
  python -m scripts.evaluate --mode full --limit 10
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent.router import heuristic_route  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.errors import AppError  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.services import build_services  # noqa: E402

BENCHMARK = ROOT / "eval" / "benchmark.jsonl"
PARAPHRASE = ROOT / "eval" / "retrieval_paraphrase.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
DOCS_DIR = ROOT / "data" / "documents"


def load_benchmark(limit: int | None = None, path: Path = BENCHMARK) -> list[dict]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return items[:limit] if limit else items


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


# ---------------------------------------------------------------- matching
def _cells(rows: list[list]) -> list:
    return [c for row in rows for c in row]


def _cell_match(expected, actual_cells: list) -> bool:
    if isinstance(expected, (int, float)):
        for a in actual_cells:
            if isinstance(a, (int, float)) and abs(a - expected) <= max(0.01, abs(expected) * 0.005):
                return True
        return False
    return any(str(expected).strip().lower() == str(a).strip().lower() for a in actual_cells)


def sql_result_match(reference_rows: list[list], generated_rows: list[list]) -> bool:
    """Every value in the reference result must appear in the generated result.
    Numbers match within 0.5% (or 0.01). Column names/order are ignored."""
    actual = _cells(generated_rows)
    return all(_cell_match(v, actual) for v in _cells(reference_rows))


def _number_variants(value: float) -> set[str]:
    variants = {f"{value:.2f}", f"{value:,.2f}", f"{value:.1f}", f"{value:,.1f}"}
    if float(value).is_integer():
        variants |= {str(int(value)), f"{int(value):,}"}
    return variants


def answer_mentions(values: list, answer: str) -> bool:
    text = answer.replace("$", "")
    for v in values:
        if isinstance(v, (int, float)):
            if not any(variant in text for variant in _number_variants(v)):
                return False
        elif str(v).lower() not in text.lower():
            return False
    return True


def _contains(keyword: str, text: str) -> bool:
    return bool(re.search(r"(?<![\d.])" + re.escape(keyword.lower()) + r"(?![\d])", text.lower()))


def keyword_recall(keywords: list[str], answer: str) -> float | None:
    if not keywords:
        return None
    hits = [_contains(k, answer) for k in keywords]
    return sum(hits) / len(hits)


def contains_all(keywords: list[str], text: str) -> bool:
    return bool(keywords) and all(_contains(k, text) for k in keywords)


# ---------------------------------------------------------------- evaluation
def evaluate_retrieval(services, items: list[dict], k: int) -> dict:
    per_item = []
    for item in items:
        route, _ = heuristic_route(item["question"])
        result = services.retriever.retrieve(item["question"], top_k=k)
        sources = [c.metadata.get("source") for c in result.chunks]
        expected = set(item.get("expected_sources", []))
        rank = next((i for i, s in enumerate(sources, 1) if s in expected), None)
        keywords = item.get("expected_keywords", [])
        answer_rank = next((i for i, c in enumerate(result.chunks, 1) if contains_all(keywords, c.text)), None)
        per_item.append({
            "id": item["id"],
            "expected_route": item.get("expected_route", "documents"),
            "heuristic_route": route,
            "answerable": item["answerable"],
            "expected_sources": sorted(expected),
            "retrieved_sources": sources,
            "first_relevant_rank": rank,
            "has_keywords": bool(keywords),
            "first_answer_chunk_rank": answer_rank,
            "evidence_sufficient": result.sufficient,
            "evidence_reason": result.reason,
        })

    doc_items = [p for p in per_item if p["expected_sources"]]
    chunk_items = [p for p in doc_items if p["has_keywords"]]
    unanswerable = [p for p in per_item if not p["answerable"]]
    return {
        "per_item": per_item,
        "metrics": {
            "heuristic_routing_accuracy": _mean([p["heuristic_route"] == p["expected_route"] for p in per_item]),
            "retrieval_questions": len(doc_items),
            "hit_at_1": _mean([p["first_relevant_rank"] == 1 for p in doc_items]),
            f"hit_at_{k}": _mean([p["first_relevant_rank"] is not None for p in doc_items]),
            "mrr": _mean([1 / p["first_relevant_rank"] if p["first_relevant_rank"] else 0.0 for p in doc_items]),
            # Stricter, chunk-level: a retrieved chunk must contain every expected answer keyword.
            "answer_chunk_questions": len(chunk_items),
            "answer_chunk_hit_at_1": _mean([p["first_answer_chunk_rank"] == 1 for p in chunk_items]),
            f"answer_chunk_hit_at_{k}": _mean([p["first_answer_chunk_rank"] is not None for p in chunk_items]),
            "evidence_gate_pass_rate_answerable": _mean([p["evidence_sufficient"] for p in doc_items]),
            "evidence_gate_reject_rate_unanswerable": _mean([not p["evidence_sufficient"] for p in unanswerable]),
        },
    }


def evaluate_full(services, items: list[dict]) -> dict:
    per_item = []
    for item in items:
        record = {"id": item["id"], "question": item["question"],
                  "expected_route": item.get("expected_route", "documents")}
        try:
            state = services.agent.run(item["question"])
        except Exception as exc:  # evaluation must survive individual failures
            record["error"] = f"{type(exc).__name__}: {exc}"
            per_item.append(record)
            continue
        answer = state.get("answer", "")
        cited_sources = [c["source"] for c in state.get("citations", [])]
        record.update({
            "route": state.get("route"),
            "answer": answer,
            "cited_sources": cited_sources,
            "insufficient_evidence": state.get("insufficient_evidence"),
            "sql_query": state.get("sql_query"),
            "sql_error": state.get("sql_error"),
            "sql_attempts": state.get("sql_attempts"),
            "warnings": state.get("warnings", []),
            "latency_ms": state.get("latency_ms"),
            "keyword_recall": keyword_recall(item.get("expected_keywords", []), answer),
        })
        if item.get("expected_sources") and cited_sources:
            relevant = sum(s in item["expected_sources"] for s in cited_sources)
            record["citation_precision"] = relevant / len(cited_sources)
        keywords = item.get("expected_keywords", [])
        if keywords and state.get("citations"):
            # Does the text the answer cites actually contain the expected facts?
            chunks = {c["chunk_id"]: c["text"] for c in (state.get("retrieval") or {}).get("chunks", [])}
            cited_text = "\n".join(chunks.get(c["chunk_id"], "") for c in state["citations"])
            record["citation_supports_answer"] = contains_all(keywords, cited_text)
        if not state.get("insufficient_evidence") and state.get("grounding", {}).get("checked"):
            record["numbers_unsupported"] = state["grounding"]["numbers_unsupported"]
        if item.get("reference_sql"):
            try:
                reference = services.database.execute(item["reference_sql"])
                record["reference_rows"] = reference.rows
                generated = (state.get("sql_result") or {}).get("rows")
                record["sql_executed"] = generated is not None
                record["sql_result_match"] = bool(generated is not None and sql_result_match(reference.rows, generated))
                record["answer_mentions_reference_values"] = answer_mentions(_cells(reference.rows)[:3], answer)
            except AppError as exc:
                record["reference_error"] = exc.message
        if not item["answerable"]:
            record["correct_refusal"] = bool(state.get("insufficient_evidence"))
        per_item.append(record)

    ok = [p for p in per_item if "error" not in p]
    sql_items = [p for p in ok if "sql_executed" in p]
    return {
        "per_item": per_item,
        "metrics": {
            "questions": len(per_item),
            "errors": len(per_item) - len(ok),
            "routing_accuracy": _mean([p["route"] == p["expected_route"] for p in ok]),
            "sql_execution_success": _mean([p["sql_executed"] for p in sql_items]),
            "sql_result_match": _mean([p["sql_result_match"] for p in sql_items]),
            "answer_mentions_reference_values": _mean([p["answer_mentions_reference_values"] for p in sql_items]),
            "keyword_recall": _mean([p["keyword_recall"] for p in ok if p.get("keyword_recall") is not None]),
            "citation_precision": _mean([p["citation_precision"] for p in ok if "citation_precision" in p]),
            "citation_supports_answer": _mean(
                [p["citation_supports_answer"] for p in ok if "citation_supports_answer" in p]),
            "answers_fully_grounded_numbers": _mean(
                [not p["numbers_unsupported"] for p in ok if "numbers_unsupported" in p]),
            "refusal_accuracy_unanswerable": _mean([p["correct_refusal"] for p in ok if "correct_refusal" in p]),
            "false_refusal_rate_answerable": _mean(
                [bool(p["insufficient_evidence"]) for p in ok if "correct_refusal" not in p]
            ),
            "median_latency_ms": sorted(p["latency_ms"] for p in ok)[len(ok) // 2] if ok else None,
        },
    }


def evaluate_ablation(services, k: int) -> dict:
    """Compare retriever variants on chunk-level answer retrieval.

    Uses the document questions of the main benchmark plus the paraphrase set, whose
    questions deliberately avoid the documents' wording.
    """
    from app.retrieval.bm25 import BM25Index
    from app.retrieval.hybrid import HybridRetriever

    store, embedder = services.store, services.retriever.embedder
    bm25 = BM25Index(store.all_chunks())
    variants = {
        "dense": lambda q: [h.chunk.text for h in store.query(embedder.embed_query(q), k)],
        "bm25": lambda q: [c.text for c, _ in bm25.search(q, k)],
        "hybrid_rrf": lambda q: [c.text for c in HybridRetriever(services.settings, embedder, store, None)
                                 .retrieve(q, top_k=k).chunks],
        "hybrid_rrf_rerank": lambda q: [c.text for c in services.retriever.retrieve(q, top_k=k).chunks],
    }
    sets = {
        "benchmark_document_questions": [i for i in load_benchmark() if i.get("expected_keywords")],
        "paraphrase": load_benchmark(path=PARAPHRASE),
    }
    metrics, per_item = {}, []
    for set_name, items in sets.items():
        for name, search in variants.items():
            ranks = []
            for item in items:
                texts = search(item["question"])
                rank = next((i for i, t in enumerate(texts, 1) if contains_all(item["expected_keywords"], t)), None)
                ranks.append(rank)
                per_item.append({"set": set_name, "variant": name, "id": item["id"], "answer_chunk_rank": rank})
            n = len(items)
            metrics[f"{set_name}/{name}"] = {
                "n": n,
                "hit_at_1": round(sum(r == 1 for r in ranks) / n, 4),
                f"hit_at_{k}": round(sum(r is not None for r in ranks) / n, 4),
                "mrr": round(sum(1 / r for r in ranks if r) / n, 4),
            }
        gate = [services.retriever.retrieve(i["question"], top_k=k).sufficient for i in items]
        metrics[f"{set_name}/evidence_gate_pass_rate"] = round(sum(gate) / len(gate), 4)
    return {"metrics": metrics, "per_item": per_item}


def aggregate_runs(runs: list[dict]) -> dict:
    """Mean of each metric over runs, plus min/max so LLM variance is visible."""
    if len(runs) == 1:
        return runs[0]
    metrics, spread = {}, {}
    for key in runs[0]["metrics"]:
        values = [r["metrics"][key] for r in runs if r["metrics"][key] is not None]
        metrics[key] = round(sum(values) / len(values), 4) if values else None
        spread[key] = [min(values), max(values)] if values else None
    return {"metrics": metrics, "metric_min_max": spread,
            "per_item": [dict(p, run=i) for i, r in enumerate(runs) for p in r["per_item"]]}


def reproducibility_metadata() -> dict:
    import hashlib
    import importlib.metadata as md
    import platform
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                    text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = None, None
    packages = ["langgraph", "langchain-openai", "chromadb", "sentence-transformers", "rank-bm25", "sqlglot"]
    return {
        "git_commit": sha,
        "git_dirty": dirty,
        "benchmark_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                             for p in sorted((ROOT / "eval").glob("*.jsonl"))},
        "corpus_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in sorted(DOCS_DIR.iterdir()))).hexdigest()[:16],
        "python": platform.python_version(),
        "packages": {p: md.version(p) for p in packages},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["retrieval", "ablation", "full"], default="retrieval")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-reranker", action="store_true", help="ablation: disable cross-encoder")
    parser.add_argument("--benchmark", type=Path, nargs="+", default=[BENCHMARK],
                        help="JSONL file(s) for retrieval/full modes, e.g. eval/retrieval_paraphrase.jsonl "
                             "eval/calibration_unanswerable.jsonl")
    parser.add_argument("--repeats", type=int, default=1, help="full mode: repeat runs to measure LLM variance")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING")
    if args.mode == "full" and (not settings.llm_configured or not settings.database_url):
        parser.error("--mode full needs OPENAI_API_KEY and DATABASE_URL.")

    # Index the demo corpus into a throwaway collection so results are reproducible.
    with tempfile.TemporaryDirectory() as tmp:
        eval_settings = settings.model_copy(update={"chroma_path": Path(tmp), "chroma_collection": "eval"})
        services = build_services(eval_settings, use_reranker=not args.no_reranker)
        for path in sorted(DOCS_DIR.iterdir()):
            services.ingestion.ingest_path(path)

        items = [i for path in args.benchmark for i in load_benchmark(path=path)][: args.limit]
        if args.mode == "retrieval":
            report = evaluate_retrieval(services, items, args.k)
        elif args.mode == "ablation":
            report = evaluate_ablation(services, args.k)
        else:
            runs = [evaluate_full(services, items) for _ in range(max(1, args.repeats))]
            report = aggregate_runs(runs)

    report["config"] = {
        "mode": args.mode,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "benchmark_files": [str(p) for p in args.benchmark],
        "benchmark_items": len(items),
        "embedding_model": settings.embedding_model,
        "reranker_model": None if args.no_reranker else settings.reranker_model,
        "llm_model": settings.openai_model if args.mode == "full" else None,
        "llm_temperature": settings.llm_temperature if args.mode == "full" else None,
        "repeats": args.repeats if args.mode == "full" else 1,
        "top_k": args.k,
        "min_rerank_score": settings.min_rerank_score,
        **reproducibility_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ("-norerank" if args.no_reranker else "") + (f"-x{args.repeats}" if args.repeats > 1 else "")
    out = RESULTS_DIR / f"{args.mode}{suffix}-{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    print(f"\nResults written to {out.relative_to(ROOT)}\n")
    print("| metric | value |\n|---|---|")
    for key, value in report["metrics"].items():
        print(f"| {key} | {value} |")


if __name__ == "__main__":
    main()
