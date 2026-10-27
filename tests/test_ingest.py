import pytest

from rag_doc_qa.ingest import _slugify, extract_text, ingest_document
from rag_doc_qa.vectorstore import VectorStore


def test_extract_text_plain_text_passthrough():
    text = extract_text("notes.txt", b"hello world")
    assert text == "hello world"


def test_extract_text_pdf_branch(monkeypatch):
    """Exercise the pypdf extraction branch without a real PDF file or network access, by
    stubbing PdfReader with a fake that mimics its page/extract_text interface."""

    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, stream):
            self.pages = [FakePage("Page one content."), FakePage("Page two content.")]

    monkeypatch.setattr("rag_doc_qa.ingest.PdfReader", FakeReader)

    text = extract_text(
        "doc.pdf", b"irrelevant-bytes-for-this-test", content_type="application/pdf"
    )
    assert text == "Page one content.\nPage two content."


def test_extract_text_detects_pdf_by_extension_even_without_content_type(monkeypatch):
    class FakePage:
        def extract_text(self):
            return "From extension detection."

    class FakeReader:
        def __init__(self, stream):
            self.pages = [FakePage()]

    monkeypatch.setattr("rag_doc_qa.ingest.PdfReader", FakeReader)

    text = extract_text("report.PDF", b"bytes", content_type=None)
    assert text == "From extension detection."


def test_extract_text_malformed_pdf_raises_value_error(monkeypatch):
    class ExplodingReader:
        def __init__(self, stream):
            raise RuntimeError("not a real PDF")

    monkeypatch.setattr("rag_doc_qa.ingest.PdfReader", ExplodingReader)

    with pytest.raises(ValueError, match="Could not parse PDF"):
        extract_text("broken.pdf", b"not-a-pdf", content_type="application/pdf")


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("My Report.txt", "my-report"),
        ("weird!!name###.pdf", "weird-name"),
        ("no_extension", "no-extension"),
        ("...", "document"),
    ],
)
def test_slugify(filename, expected):
    assert _slugify(filename) == expected


def test_ingest_document_raises_on_empty_extracted_text(tmp_path, shared_embedder):
    store = VectorStore(str(tmp_path / "chroma"), "empty-test")
    with pytest.raises(ValueError, match="No extractable text content"):
        ingest_document(store, shared_embedder, "blank.txt", b"   \n\t  ")


def test_ingest_document_stores_chunks_with_metadata(
    tmp_path, shared_embedder, sample_document_bytes
):
    store = VectorStore(str(tmp_path / "chroma"), "ingest-test")
    result = ingest_document(
        store, shared_embedder, "sample_document.txt", sample_document_bytes
    )

    assert result["filename"] == "sample_document.txt"
    assert result["num_chunks"] > 1
    assert store.count() == result["num_chunks"]
