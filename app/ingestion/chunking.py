"""Split loaded sections into overlapping chunks with stable IDs."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.loaders import Section


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_sections(
    sections: list[Section],
    *,
    doc_id: str,
    source: str,
    file_type: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for section in sections:
        for piece in splitter.split_text(section.text):
            piece = piece.strip()
            if not piece:
                continue
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{index}",
                    text=piece,
                    metadata={
                        "doc_id": doc_id,
                        "source": source,
                        "file_type": file_type,
                        "chunk_index": index,
                        **section.metadata,
                    },
                )
            )
    return chunks
