"""File loaders for PDF, DOCX, TXT/MD and CSV.

Each loader returns a list of ``Section`` objects: a block of text plus the
location metadata (page number, row range) needed for precise citations.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.errors import DocumentParseError, EmptyDocumentError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
CSV_ROWS_PER_SECTION = 25


@dataclass
class Section:
    text: str
    metadata: dict = field(default_factory=dict)


def file_type_for(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext or '(none)'}'. Supported types: {allowed}."
        )
    return ext.lstrip(".")


def load_bytes(data: bytes, filename: str) -> list[Section]:
    """Parse raw file bytes into text sections based on the file extension."""
    file_type = file_type_for(filename)
    if not data or not data.strip():
        raise EmptyDocumentError(f"'{filename}' is empty.")

    loader = {
        "pdf": _load_pdf,
        "docx": _load_docx,
        "txt": _load_text,
        "md": _load_text,
        "csv": _load_csv,
    }[file_type]

    try:
        sections = loader(data)
    except (EmptyDocumentError, DocumentParseError):
        raise
    except Exception as exc:  # parser libraries raise many different types
        logger.warning("Failed to parse %s: %s", filename, exc)
        raise DocumentParseError(f"Could not parse '{filename}' as {file_type.upper()}: {exc}") from exc

    sections = [s for s in sections if s.text.strip()]
    if not sections:
        raise EmptyDocumentError(f"No extractable text found in '{filename}'.")
    return sections


def load_path(path: Path) -> list[Section]:
    return load_bytes(path.read_bytes(), path.name)


def _load_pdf(data: bytes) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise DocumentParseError("Encrypted PDFs are not supported.")
    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            sections.append(Section(text=text, metadata={"page": page_number}))
    return sections


def _load_docx(data: bytes) -> list[Section]:
    import docx

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return [Section(text="\n".join(parts))]


def _load_text(data: bytes) -> list[Section]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    if "\x00" in text:
        raise DocumentParseError("File appears to be binary, not text.")
    return [Section(text=text)]


def _load_csv(data: bytes) -> list[Section]:
    try:
        frame = pd.read_csv(io.BytesIO(data))
    except pd.errors.EmptyDataError as exc:
        raise EmptyDocumentError("CSV file has no data.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DocumentParseError(f"Malformed CSV: {exc}") from exc
    if frame.empty:
        raise EmptyDocumentError("CSV file has a header but no rows.")

    frame = frame.fillna("")
    sections = []
    # Render each row as "column: value" pairs so it is meaningful to both
    # BM25 and the embedding model, and keep row ranges for citations.
    for start in range(0, len(frame), CSV_ROWS_PER_SECTION):
        chunk = frame.iloc[start : start + CSV_ROWS_PER_SECTION]
        lines = [
            "; ".join(f"{col}: {row[col]}" for col in frame.columns)
            for _, row in chunk.iterrows()
        ]
        sections.append(
            Section(
                text="\n".join(lines),
                metadata={"row_start": start + 1, "row_end": start + len(chunk)},
            )
        )
    return sections
