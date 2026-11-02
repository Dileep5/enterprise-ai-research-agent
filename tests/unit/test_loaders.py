import io

import pytest

from app.errors import DocumentParseError, EmptyDocumentError, UnsupportedFileTypeError
from app.ingestion.chunking import chunk_sections
from app.ingestion.loaders import load_bytes, load_path
from tests.conftest import SAMPLE_DOCS


def _pdf_bytes(pages: list[str]) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


def _docx_bytes(paragraphs: list[str], table: list[list[str]] | None = None) -> bytes:
    import docx

    document = docx.Document()
    for p in paragraphs:
        document.add_paragraph(p)
    if table:
        t = document.add_table(rows=0, cols=len(table[0]))
        for row in table:
            cells = t.add_row().cells
            for cell, value in zip(cells, row, strict=True):
                cell.text = value
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def test_pdf_keeps_page_numbers():
    sections = load_bytes(_pdf_bytes(["Alpha page text", "Beta page text"]), "report.pdf")
    assert [s.metadata["page"] for s in sections] == [1, 2]
    assert "Beta" in sections[1].text


def test_docx_includes_paragraphs_and_tables():
    data = _docx_bytes(["Intro paragraph"], table=[["Tier", "Rate"], ["Gold", "2"]])
    text = load_bytes(data, "guide.docx")[0].text
    assert "Intro paragraph" in text
    assert "Gold | 2" in text


def test_txt_falls_back_to_latin1():
    sections = load_bytes("Caf\xe9 policy".encode("latin-1"), "notes.txt")
    assert sections[0].text == "Café policy"


def test_csv_rows_rendered_with_column_names_and_row_ranges():
    data = b"name,lead_time_days\nBoreal,21\nSummit,18\n"
    section = load_bytes(data, "suppliers.csv")[0]
    assert "name: Boreal; lead_time_days: 21" in section.text
    assert section.metadata == {"row_start": 1, "row_end": 2}


def test_large_csv_split_into_multiple_sections():
    rows = "\n".join(f"item{i},{i}" for i in range(60))
    sections = load_bytes(f"name,qty\n{rows}\n".encode(), "big.csv")
    assert len(sections) == 3
    assert sections[-1].metadata == {"row_start": 51, "row_end": 60}


@pytest.mark.parametrize("filename", ["malware.exe", "archive.zip", "noextension"])
def test_unsupported_extension(filename):
    with pytest.raises(UnsupportedFileTypeError):
        load_bytes(b"data", filename)


@pytest.mark.parametrize("data, filename", [(b"", "a.txt"), (b"   \n ", "a.txt"), (b"", "a.pdf")])
def test_empty_files_rejected(data, filename):
    with pytest.raises(EmptyDocumentError):
        load_bytes(data, filename)


def test_csv_header_only_rejected():
    with pytest.raises(EmptyDocumentError):
        load_bytes(b"a,b,c\n", "empty.csv")


def test_corrupt_pdf_rejected():
    with pytest.raises(DocumentParseError):
        load_bytes(b"%PDF-1.4 this is not really a pdf", "broken.pdf")


def test_corrupt_docx_rejected():
    with pytest.raises(DocumentParseError):
        load_bytes(b"PK\x03\x04 not a docx", "broken.docx")


def test_binary_txt_rejected():
    with pytest.raises(DocumentParseError):
        load_bytes(b"abc\x00\x01\x02", "binary.txt")


def test_chunking_produces_stable_ids_and_metadata():
    sections = load_path(SAMPLE_DOCS / "annual_report_2025.pdf")
    chunks = chunk_sections(sections, doc_id="abc", source="annual_report_2025.pdf", file_type="pdf",
                            chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 3
    assert [c.chunk_id for c in chunks] == [f"abc:{i}" for i in range(len(chunks))]
    assert all(c.metadata["page"] in (1, 2, 3) for c in chunks)
    assert all(len(c.text) <= 300 for c in chunks)


@pytest.mark.parametrize("path", sorted(SAMPLE_DOCS.iterdir()), ids=lambda p: p.name)
def test_all_sample_documents_load(path):
    assert load_path(path)
