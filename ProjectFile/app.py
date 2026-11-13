"""Browser UI. Run with streamlit run app.py (or use a supplied launcher)."""
import hmac
from html import escape
import logging
import os
from pathlib import Path
from time import perf_counter

import streamlit as st

from rag.answer import Answerer, is_greeting
from rag.config import MAX_QUESTION_CHARS, ROOT, Settings
from rag.index import Corpus
from rag.models import embedding_ready, prepare_models

log = logging.getLogger(__name__)
st.set_page_config(page_title="PDF Desk", page_icon="📘", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .block-container {max-width: 1080px; padding-top: 2.5rem;}
    [data-testid="stSidebar"] {display: none;}
    .eyebrow {color: #52667b; font-size: .78rem; font-weight: 650;
              letter-spacing: .13em; text-transform: uppercase; margin-bottom: .6rem;}
    .answer-text {font-family: inherit; white-space: pre-wrap; overflow-wrap: anywhere;
                  line-height: 1.7; font-size: 1rem;}
    [data-testid="stChatMessage"] {border: 1px solid #e3e9ef; border-radius: 14px;}
    [data-testid="stMetric"] {border: 1px solid #e3e9ef; border-radius: 12px; padding: 14px;}
    .empty-state {padding: 2rem; border: 1px dashed #aabbd0; border-radius: 16px; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def resources(settings):
    return Corpus(settings), Answerer(settings)


def show_text(container, text):
    # Render model text without allowing model-generated HTML, images or remote links.
    container.markdown(f'<div class="answer-text">{escape(text)}</div>', unsafe_allow_html=True)


def show_sources(sources, abstained=False):
    if not sources:
        return
    label = "Retrieved passages" if abstained else "Sources"
    with st.expander(f"{label} · {len(sources)}"):
        if abstained:
            st.caption("These passages were retrieved, but an answer could not be supported.")
        for source in sources:
            st.markdown(f"**[{source['label']}] {escape(source['document'])} · page {source['page']}**")
            st.text(source["text"])


def administrator():
    expected = os.getenv("ADMIN_PASSWORD", "")
    if not expected:
        st.info("Document management is open for this demo. Set an administrator password before sharing the app.")
        return True
    if st.session_state.get("admin_authenticated"):
        if st.button("Sign out", key="admin_logout"):
            st.session_state.pop("admin_authenticated", None)
            st.rerun()
        return True
    st.caption("Sign in to manage the document collection and local models.")
    with st.form("admin_login"):
        password = st.text_input("Administrator password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if hmac.compare_digest(password.encode(), expected.encode()):
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("That password is incorrect.")
    return False


def run_update(corpus, uploads=None, remove=None):
    try:
        with st.status("Updating the document collection…", expanded=True) as status:
            index = corpus.update(uploads=uploads, remove=remove, progress=status.write)
            status.update(label="Your documents are ready.", state="complete", expanded=False)
        st.session_state.flash = f"Indexed {len(index.metadata['documents'])} PDFs and {len(index.chunks)} passages."
        st.session_state.upload_version = st.session_state.get("upload_version", 0) + 1
        st.rerun()
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
    except Exception:
        log.exception("Document update failed")
        st.error("The collection could not be updated. Your previous index was kept. Check server logs and try again.")


def manage(settings, corpus, index, issue):
    st.markdown('<div class="eyebrow">Document library</div>', unsafe_allow_html=True)
    st.title("Manage documents")
    if not administrator():
        return
    if flash := st.session_state.pop("flash", None):
        st.success(flash)
    st.subheader("Backend PDF library")
    st.write("Place your prepared PDFs in this server folder. They are indexed automatically at startup and when the app refreshes.")
    st.code(str(settings.pdf_dir), language=None)
    with st.container(border=True):
        st.subheader("1. Prepare local models")
        ready = embedding_ready(settings) and settings.llm_path.is_file()
        if ready:
            st.success("Both models are available on this machine.")
        else:
            st.write("Download the search and answer models once (about 1.4 GB total). After setup, questions are processed locally.")
            if st.button("Download local models", type="primary"):
                try:
                    with corpus.lock, st.status("Preparing local models…", expanded=True) as status:
                        prepare_models(settings, progress=status.write)
                        status.update(label="Local models are ready.", state="complete", expanded=False)
                    st.rerun()
                except Exception:
                    log.exception("Model preparation failed")
                    st.error("The models could not be downloaded. Check internet access and free disk space, then retry. Completed downloads are reused.")
    with st.expander("Optional: upload PDFs through the browser"):
        st.write("An alternative way for the host to add documents to the same shared library.")
        st.caption("English text PDFs · up to 10 files · 10 MB per file · 50 MB total · 20 pages per file")
        with st.form("upload_documents"):
            files = st.file_uploader(
                "Choose PDF files", type=["pdf"], accept_multiple_files=True,
                key=f"uploads_{st.session_state.get('upload_version', 0)}",
                help="A file with the same name replaces the existing document after indexing succeeds.",
            )
            submitted = st.form_submit_button("Save and index PDFs", type="primary", disabled=not ready)
        if submitted:
            if not files:
                st.warning("Choose at least one PDF first.")
            else:
                run_update(corpus, uploads=[(f.name, f.getvalue()) for f in files])
    st.subheader("Current collection")
    if index:
        documents = index.metadata["documents"]
        st.dataframe([{
            "Document": d["name"], "Pages": d["pages"], "Searchable pages": d["text_pages"],
            "Passages": d["chunks"], "Size (KB)": round(d["size"] / 1024, 1),
        } for d in documents], hide_index=True, width="stretch")
        if any(d["text_pages"] < d["pages"] for d in documents):
            st.info("Some pages contain no selectable text and were skipped. Answers can only use the searchable pages.")
        with st.form("remove_documents"):
            remove = st.multiselect("Remove documents", [d["name"] for d in documents])
            deleting = st.form_submit_button("Remove selected and rebuild", disabled=not ready)
        if deleting:
            if remove:
                run_update(corpus, remove=remove)
            else:
                st.warning("Select a document to remove first.")
    else:
        st.info("No PDFs have been indexed yet.")
    if issue:
        st.warning(issue)
    if st.button("Rebuild search index", disabled=not ready):
        run_update(corpus)


def chat(settings, corpus, answerer, index, issue):
    if issue:
        st.error("The document assistant is not ready yet. Please try again shortly.")
    ready = index is not None and issue is None and settings.llm_path.is_file()
    if index and not settings.llm_path.is_file():
        st.error("The document assistant is still starting. Please try again shortly.")
    messages = st.session_state.setdefault("messages", [{
        "role": "assistant",
        "content": "How can I help you today?",
        "welcome": True,
    }])
    for message in messages:
        with st.chat_message(message["role"]):
            show_text(st, message["content"])
            if message["role"] == "assistant" and not message.get("welcome"):
                show_sources(message.get("sources", []), message.get("abstained", False))
                if "seconds" in message:
                    st.caption(f"Answered in {message['seconds']:.1f} seconds")
    question = st.chat_input("Ask a question about the document library…", disabled=not ready, max_chars=MAX_QUESTION_CHARS)
    if not question or not question.strip():
        return
    question = question.strip()
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        show_text(st, question)
    with st.chat_message("assistant"):
        output = st.empty()
        started = perf_counter()
        try:
            with st.spinner("Finding relevant passages…"):
                # Re-read at submission time: another session may have updated PDFs.
                if is_greeting(question):
                    hits = []
                else:
                    current, current_issue = corpus.snapshot()
                    if current_issue or current is None:
                        raise RuntimeError("The document collection changed. Refresh the page and try again.")
                    hits = corpus.retrieve(question, current)
            with st.spinner("Preparing your answer on this machine…"):
                result = answerer.answer(question, hits, on_text=lambda text: show_text(output, text + " ▌"))
            show_text(output, result.text)
            show_sources(result.sources, result.abstained)
            elapsed = perf_counter() - started
            st.caption(f"Answered in {elapsed:.1f} seconds")
            messages.append({
                "role": "assistant", "content": result.text, "sources": result.sources,
                "seconds": elapsed, "abstained": result.abstained,
            })
        except (ValueError, RuntimeError) as exc:
            output.empty()
            st.error(str(exc))
            messages.pop()  # A failed request should not leave an unanswered history entry.
        except Exception:
            output.empty()
            log.exception("Question answering failed")
            st.error("The answer could not be generated. Please try again; the administrator can check server logs.")
            messages.pop()
    # Bound per-session memory without losing the active turn while rendering.
    st.session_state.messages = messages[-40:]


def main():
    page = "Chat"
    trigger_space, trigger = st.columns([24, 1])
    with trigger:
        if st.button("⋮", key="management_trigger", help="Open management"):
            st.session_state.management_open = not st.session_state.get("management_open", False)
            st.rerun()

    if st.session_state.get("management_open"):
        st.markdown('<div class="eyebrow">Management</div>', unsafe_allow_html=True)
        st.title("PDF Desk management")
        management_action = st.radio(
            "Management options",
            ["Manage documents"],
            horizontal=True,
            label_visibility="collapsed",
        )
        controls = st.columns(2)
        if controls[0].button("Back to chat", width="stretch"):
            st.session_state.management_open = False
            st.rerun()
        if controls[1].button("Clear chat", width="stretch"):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "How can I help you today?",
                "welcome": True,
            }]
            st.session_state.management_open = False
            st.rerun()
        page = management_action
    try:
        settings = Settings.from_env()
        corpus, answerer = resources(settings)
        with st.spinner("Preparing the document library…"):
            index, issue = corpus.refresh_if_needed()
    except (ValueError, RuntimeError, OSError) as exc:
        # A damaged index should still allow access to the rebuild screen.
        try:
            settings = Settings.from_env()
            corpus, answerer = resources(settings)
        except ValueError:
            st.error(f"Application configuration needs attention: {exc}")
            st.stop()
        index, issue = None, str(exc)
    if page == "Manage documents":
        manage(settings, corpus, index, issue)
    else:
        chat(settings, corpus, answerer, index, issue)


if __name__ == "__main__":
    main()
