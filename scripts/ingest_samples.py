"""Ingest the demo documents (or any folder) into the configured Chroma store.

Usage: python -m scripts.ingest_samples [--dir data/documents]
Re-running is safe: documents are content-addressed and duplicates are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.errors import AppError  # noqa: E402
from app.ingestion.loaders import SUPPORTED_EXTENSIONS  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.retrieval.embeddings import SentenceTransformerEmbedder  # noqa: E402
from app.retrieval.vector_store import ChromaVectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=ROOT / "data" / "documents")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = IngestionPipeline(
        settings,
        SentenceTransformerEmbedder(settings.embedding_model),
        ChromaVectorStore(settings.chroma_path, settings.chroma_collection),
    )
    failures = 0
    for path in sorted(args.dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            result = pipeline.ingest_path(path)
            print(f"{result.status:9s} {path.name:32s} {result.chunks:3d} chunks  doc_id={result.doc_id}")
        except AppError as exc:
            failures += 1
            print(f"error     {path.name:32s} {exc.message}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
