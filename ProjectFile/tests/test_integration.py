import json
import os
from pathlib import Path
import socket
from time import perf_counter

import pytest

from conftest import pdf_bytes
from rag.answer import Answerer
from rag.config import Settings
from rag.index import Corpus
from rag.models import embedding_ready


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="Set RUN_MODEL_TESTS=1 after downloading models")
def test_real_models_answer_offline_and_abstain(tmp_path, monkeypatch):
    base = Settings.from_env()
    settings = Settings(data_dir=tmp_path / "data", model_dir=base.model_dir, threads=2)
    assert embedding_ready(settings) and settings.llm_path.is_file()
    def deny_network(*args, **kwargs):
        raise AssertionError("Normal chatbot use must not open a network connection")
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    corpus, answerer = Corpus(settings), Answerer(settings)
    started = perf_counter()
    (settings.pdf_dir / "Course.pdf").write_bytes(pdf_bytes(
        "The final project contributes 40 percent of the course grade. Students must submit the project on December 15.",
        "The library opens at 9 AM Monday to Friday. Students may borrow up to five books.",
    ))
    index, issue = corpus.refresh_if_needed()
    assert issue is None
    indexing_seconds = perf_counter() - started
    hits = corpus.retrieve("What percentage of the course grade is the final project worth?", index)
    assert hits and hits[0].chunk.page == 1
    answer = answerer.answer("What percentage of the course grade is the final project worth?", hits)
    assert not answer.abstained, answer.text
    assert "40" in answer.text and "[S1]" in answer.text
    assert answer.sources[0]["page"] == 1
    unknown = "Who won the 2022 football World Cup?"
    unknown_hits = corpus.retrieve(unknown, index)
    abstention = answerer.answer(unknown, unknown_hits)
    assert abstention.abstained, abstention.text
    import psutil
    report = {
        "model": settings.llm_path.name,
        "cpu_only": True,
        "threads": settings.threads,
        "indexing_seconds": round(indexing_seconds, 3),
        "answer_seconds": round(answer.seconds, 3),
        "process_rss_mb": round(psutil.Process().memory_info().rss / 1024 ** 2, 1),
        "answer": answer.text,
        "out_of_scope_answer": abstention.text,
        "network_connections_allowed": False,
        "note": "Synthetic test PDFs on the development machine; Azure performance must be measured separately.",
    }
    print(json.dumps(report, indent=2))
    destination = Path(__file__).resolve().parent.parent / "work" / "integration-report.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
