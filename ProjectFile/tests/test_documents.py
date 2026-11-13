from io import BytesIO

from pypdf import PdfWriter
import pytest

from rag.documents import DocumentError, chunk_document, read_pdf, safe_filename
from conftest import FakeEmbedder, pdf_bytes


def test_real_pdf_extraction_preserves_physical_page_numbers():
    document = read_pdf("Handbook.pdf", pdf_bytes("First page text.", "", "Third page text."))
    assert document.page_count == 3
    assert [p.number for p in document.pages] == [1, 3]
    assert "Third page" in document.pages[-1].text


@pytest.mark.parametrize("name", ["../escape.pdf", "/tmp/file.pdf", "folder\\file.pdf", ".pdf", "file.txt"])
def test_rejects_unsafe_or_invalid_names(name):
    with pytest.raises(DocumentError):
        safe_filename(name)


def test_rejects_disguised_and_scanned_documents():
    with pytest.raises(DocumentError, match="does not appear"):
        read_pdf("fake.pdf", b"not a PDF")
    with pytest.raises(DocumentError, match="no selectable text"):
        read_pdf("scan.pdf", pdf_bytes(""))


def test_rejects_password_protected_documents():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret")
    stream = BytesIO()
    writer.write(stream)
    with pytest.raises(DocumentError, match="password-protected"):
        read_pdf("locked.pdf", stream.getvalue())


def test_chunking_keeps_tail_text_and_never_crosses_pages(settings):
    text = " ".join(f"word{i}" for i in range(1100))
    document = read_pdf("Long.pdf", pdf_bytes(text, "A separate page."))
    tokenizer = FakeEmbedder(settings).tokenizer
    chunks = chunk_document(document, tokenizer)
    assert "word0 " in chunks[0].text
    assert any("word1099" in c.text for c in chunks)
    assert chunks[-1].page == 2
    assert all(len(tokenizer.encode(c.text).ids) <= 320 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
