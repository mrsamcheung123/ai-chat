import numpy as np
import pytest

from conftest import pdf_bytes
from rag.documents import DocumentError
from rag.index import load_index, rank


def test_persistence_retrieval_and_page_metadata(corpus, settings):
    corpus.update(uploads=[("Handbook.pdf", pdf_bytes(
        "The project deadline is December 15.",
        "The library contains books and magazines.",
    ))])
    index = load_index(settings.index_path)
    assert index.metadata["documents"][0]["pages"] == 2
    hits = corpus.retrieve("project deadline December", index)
    assert hits[0].chunk.document == "Handbook.pdf"
    assert hits[0].chunk.page == 1
    assert 0 <= hits[0].score <= 1.000001
    assert corpus.snapshot()[1] is None


def test_bad_upload_does_not_change_existing_collection(corpus, settings):
    corpus.update(uploads=[("Good.pdf", pdf_bytes("The project deadline is December 15."))])
    old_index = settings.index_path.read_bytes()
    old_pdf = (settings.pdf_dir / "Good.pdf").read_bytes()
    with pytest.raises(DocumentError):
        corpus.update(uploads=[("Good.pdf", b"broken file"), ("Another.pdf", pdf_bytes("Hello."))])
    assert settings.index_path.read_bytes() == old_index
    assert (settings.pdf_dir / "Good.pdf").read_bytes() == old_pdf
    assert not (settings.pdf_dir / "Another.pdf").exists()


def test_failed_embedding_preserves_last_working_index(corpus, settings, monkeypatch):
    corpus.update(uploads=[("Good.pdf", pdf_bytes("A project deadline."))])
    previous = settings.index_path.read_bytes()
    def fail(texts):
        raise RuntimeError("Model failure")
    monkeypatch.setattr(corpus.embedder(), "passages", fail)
    with pytest.raises(RuntimeError):
        corpus.update(uploads=[("New.pdf", pdf_bytes("A new document."))])
    assert settings.index_path.read_bytes() == previous
    assert not (settings.pdf_dir / "New.pdf").exists()


def test_replace_remove_and_detect_external_changes(corpus, settings):
    corpus.update(uploads=[("A.pdf", pdf_bytes("Apples grow on trees.")), ("B.pdf", pdf_bytes("Blue ocean."))])
    corpus.update(uploads=[("a.pdf", pdf_bytes("Bananas are yellow."))], remove=["B.pdf"])
    index, issue = corpus.snapshot()
    assert issue is None
    assert [d["name"] for d in index.metadata["documents"]] == ["A.pdf"]
    assert "Bananas" in index.chunks[0].text
    (settings.pdf_dir / "A.pdf").write_bytes(pdf_bytes("A changed PDF."))
    assert "changed" in corpus.snapshot()[1]


def test_cannot_remove_entire_collection(corpus):
    corpus.update(uploads=[("A.pdf", pdf_bytes("An apple."))])
    with pytest.raises(DocumentError, match="at least one"):
        corpus.update(remove=["A.pdf"])


def test_threshold_abstains_on_orthogonal_query(corpus):
    index = corpus.update(uploads=[("A.pdf", pdf_bytes("Apples grow on trees."))])
    vector = np.zeros(384, dtype=np.float32)
    unused = next(i for i in range(384) if index.vectors[:, i].sum() == 0)
    vector[unused] = 1
    assert rank(index, vector, top_k=4, min_similarity=0.45) == []


def test_rejects_corrupted_index(settings):
    settings.ensure_dirs()
    settings.index_path.write_bytes(b"bad index")
    with pytest.raises(ValueError, match="Rebuild"):
        load_index(settings.index_path)


def test_backend_pdfs_are_indexed_without_browser_upload(corpus, settings, monkeypatch):
    monkeypatch.setattr("rag.index.embedding_ready", lambda _: True)
    (settings.pdf_dir / "Host.pdf").write_bytes(pdf_bytes("The project deadline is December 15."))
    (settings.pdf_dir / ".DS_Store").write_bytes(b"Finder metadata")
    index, issue = corpus.refresh_if_needed()
    assert issue is None
    assert index.metadata["documents"][0]["name"] == "Host.pdf"
    assert corpus.retrieve("project deadline December", index)[0].chunk.page == 1
    def forbidden_reindex(texts):
        raise AssertionError("Unchanged PDFs must not be embedded again")
    monkeypatch.setattr(corpus.embedder(), "passages", forbidden_reindex)
    assert corpus.refresh_if_needed()[1] is None


def test_backend_replacements_and_removals_refresh_automatically(corpus, settings, monkeypatch):
    monkeypatch.setattr("rag.index.embedding_ready", lambda _: True)
    (settings.pdf_dir / "A.pdf").write_bytes(pdf_bytes("Original instructions."))
    (settings.pdf_dir / "B.PDF").write_bytes(pdf_bytes("Library hours."))
    corpus.refresh_if_needed()
    (settings.pdf_dir / "A.pdf").write_bytes(pdf_bytes("Updated instructions."))
    (settings.pdf_dir / "B.PDF").unlink()
    index, issue = corpus.refresh_if_needed()
    assert issue is None
    assert len(index.metadata["documents"]) == 1
    assert "Updated" in index.chunks[0].text
