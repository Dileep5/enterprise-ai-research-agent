"""Ingestion pipeline: bytes -> sections -> chunks -> embeddings -> vector store."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.ingestion.chunking import chunk_sections
from app.ingestion.loaders import file_type_for, load_bytes
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    doc_id: str
    filename: str
    chunks: int
    status: str  # "ingested" | "duplicate"


class IngestionPipeline:
    def __init__(self, settings: Settings, embedder: Embedder, store: ChromaVectorStore):
        self.settings = settings
        self.embedder = embedder
        self.store = store

    @staticmethod
    def document_id(data: bytes) -> str:
        # Content-addressed IDs make re-uploads idempotent.
        return hashlib.sha256(data).hexdigest()[:16]

    def ingest_bytes(self, data: bytes, filename: str) -> IngestResult:
        filename = Path(filename).name  # never trust client-supplied paths
        file_type = file_type_for(filename)
        doc_id = self.document_id(data)

        if self.store.has_document(doc_id):
            logger.info("Skipping duplicate document %s (%s)", filename, doc_id)
            return IngestResult(doc_id, filename, self.store.count_chunks(doc_id), "duplicate")

        sections = load_bytes(data, filename)
        chunks = chunk_sections(
            sections,
            doc_id=doc_id,
            source=filename,
            file_type=file_type,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, embeddings)
        logger.info("Ingested %s as %s (%d chunks)", filename, doc_id, len(chunks))
        return IngestResult(doc_id, filename, len(chunks), "ingested")

    def ingest_path(self, path: Path) -> IngestResult:
        return self.ingest_bytes(path.read_bytes(), path.name)
