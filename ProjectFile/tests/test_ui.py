from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_empty_app_and_management_screen(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py")).run()
    assert not app.exception
    assert app.title[0].value == "Ask the document assistant"
    assert app.chat_input[0].disabled
    app.checkbox[0].check().run()
    assert not app.exception
    assert app.title[0].value == "Manage documents"
    assert any(b.label == "Download local models" for b in app.button)


def test_new_visitor_can_chat_with_prepared_library(corpus, settings, monkeypatch):
    from conftest import pdf_bytes
    corpus.update(uploads=[("Host.pdf", pdf_bytes("The project deadline is December 15."))])
    settings.llm_path.write_bytes(b"placeholder: this test checks UI readiness only")
    monkeypatch.setenv("RAG_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("RAG_MODEL_DIR", str(settings.model_dir))
    monkeypatch.setenv("ADMIN_PASSWORD", "host-only")
    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    for _ in range(2):
        app = AppTest.from_file(app_path).run()
        assert not app.exception
        assert not app.chat_input[0].disabled
        assert not app.get("file_uploader")
        assert any("no PDF uploads needed" in m.value for m in app.markdown)


def test_management_requires_admin_password(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("ADMIN_PASSWORD", "a-test-password")
    app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py")).run()
    app.checkbox[0].check().run()
    assert not app.exception
    assert not any(b.label == "Download local models" for b in app.button)
    app.text_input[0].set_value("wrong")
    app.button(key="FormSubmitter:admin_login-Sign in").click().run()
    assert any("incorrect" in e.value for e in app.error)
    app.text_input[0].set_value("a-test-password")
    app.button(key="FormSubmitter:admin_login-Sign in").click().run()
    assert not app.exception
    assert any(b.label == "Download local models" for b in app.button)
