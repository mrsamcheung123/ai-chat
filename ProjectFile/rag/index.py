from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import threading

import numpy as np

from .config import (
    CHUNK_OVERLAP, CHUNK_TOKENS, EMBEDDING_MODEL, MAX_FILES,
    MAX_TOTAL_BYTES, MAX_TOTAL_PAGES,
)
from .documents import Chunk, DocumentError, chunk_document, read_pdf, safe_filename
from .models import Embedder, embedding_ready

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Index:
    metadata: dict
    chunks: list[Chunk]
    vectors: np.ndarray


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("The embedding model returned invalid values.")
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("The embedding model returned an empty vector.")
    return vectors / norms


def load_index(path: Path):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            vectors = np.asarray(saved["vectors"], dtype=np.float32)
        if metadata["version"] != SCHEMA_VERSION or metadata["embedding_model"] != EMBEDDING_MODEL:
            raise ValueError("The search index uses an older format. Rebuild it in Manage documents.")
        chunks = [Chunk(**c) for c in metadata["chunks"]]
        if vectors.ndim != 2 or vectors.shape != (len(chunks), 384):
            raise ValueError("The search index has inconsistent dimensions.")
        return Index(metadata, chunks, normalize(vectors))
    except Exception as exc:
        raise ValueError("The search index could not be loaded. Rebuild it in Manage documents.") from exc


def rank(index, query_vector, top_k, min_similarity):
    query = normalize(query_vector)
    if query.shape != (index.vectors.shape[1],):
        raise ValueError("The query and document embedding models do not match.")
    scores = index.vectors @ query
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [Hit(index.chunks[i], float(scores[i])) for i in order if scores[i] >= min_similarity]


class Corpus:
    """One shared instance per app process; readers see complete index snapshots."""

    def __init__(self, settings, embedder_factory=Embedder):
        self.settings = settings
        settings.ensure_dirs()
        self.lock = threading.RLock()
        self._embedder = None
        self._embedder_factory = embedder_factory

    def embedder(self):
        with self.lock:
            if self._embedder is None:
                self._embedder = self._embedder_factory(self.settings)
            return self._embedder

    def _files(self):
        return {p.name: p.read_bytes() for p in sorted(self.settings.pdf_dir.glob("*"))
                if p.is_file() and p.suffix.lower() == ".pdf"}

    def refresh_if_needed(self, progress=lambda message: None):
        """Prepare the host's disk-backed library; reuse unchanged indexes."""
        with self.lock:
            try:
                index, issue = self.snapshot()
            except ValueError as exc:
                index, issue = None, str(exc)
            if issue is None:
                return index, None
            if not self._files() or not embedding_ready(self.settings):
                return index, issue
            return self.update(progress=progress), None

    def snapshot(self):
        with self.lock:
            index = load_index(self.settings.index_path)
            if index is None:
                return None, "The host has not prepared the document library yet."
            expected = {d["name"]: d["digest"] for d in index.metadata["documents"]}
            actual = {name: sha256(data).hexdigest() for name, data in self._files().items()}
            if expected != actual:
                return index, "The document collection has changed. Rebuild the index in Manage documents."
            manifest = self.settings.embedding_dir / "manifest.json"
            if not manifest.is_file() or manifest.read_text() != index.metadata["embedding_signature"]:
                return index, "The search model has changed or is missing. Prepare models and rebuild the index."
            return index, None

    def update(self, uploads=None, remove=None, progress=lambda message: None):
        """Validate and embed the proposed collection before changing the current one."""
        with self.lock:
            original = self._files()
            proposed = dict(original)
            for name in remove or []:
                proposed.pop(safe_filename(name), None)
            incoming = set()
            for name, content in uploads or []:
                name = safe_filename(name)
                if name.casefold() in incoming:
                    raise DocumentError("Two uploads have the same filename. Rename one and try again.")
                incoming.add(name.casefold())
                collision = next((n for n in proposed if n.casefold() == name.casefold()), None)
                if collision:
                    # Retain spelling on case-insensitive filesystems; otherwise
                    # removing A.pdf after writing a.pdf would delete the new file.
                    name = collision
                proposed[name] = content
            if not proposed:
                raise DocumentError("Keep at least one PDF in the collection.")
            if len(proposed) > MAX_FILES:
                raise DocumentError("The collection can contain up to 10 PDFs.")
            if sum(map(len, proposed.values())) > MAX_TOTAL_BYTES:
                raise DocumentError("The total PDF collection must be 50 MB or smaller.")
            documents = []
            for name, content in sorted(proposed.items()):
                progress(f"Reading {name}…")
                documents.append(read_pdf(name, content))
            if sum(d.page_count for d in documents) > MAX_TOTAL_PAGES:
                raise DocumentError("The collection can contain up to 100 pages in total.")
            progress("Loading the local search model…")
            embedder = self.embedder()
            chunks = [c for d in documents for c in chunk_document(d, embedder.tokenizer)]
            if not chunks:
                raise DocumentError("No searchable text was found in the PDFs.")
            progress(f"Indexing {len(chunks)} passages on the CPU…")
            vectors = normalize(embedder.passages([c.text for c in chunks]))
            if vectors.shape != (len(chunks), 384):
                raise ValueError("The embedding model returned unexpected dimensions.")
            metadata = {
                "version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "embedding_model": EMBEDDING_MODEL,
                "embedding_signature": embedder.signature,
                "chunk_tokens": CHUNK_TOKENS,
                "chunk_overlap": CHUNK_OVERLAP,
                "documents": [{
                    "name": d.name, "digest": d.digest, "size": d.size,
                    "pages": d.page_count, "text_pages": len(d.pages),
                    "chunks": sum(c.document == d.name for c in chunks),
                } for d in documents],
                "chunks": [c.to_dict() for c in chunks],
            }
            fd, temporary = tempfile.mkstemp(dir=self.settings.data_dir, suffix=".npz")
            try:
                with os.fdopen(fd, "wb") as target:
                    np.savez_compressed(target, vectors=vectors, metadata=np.array(json.dumps(metadata)))
                # Commit under the same lock as readers. If a write fails, restore PDFs;
                # the old index remains intact until the final atomic replacement.
                try:
                    for name, content in proposed.items():
                        (self.settings.pdf_dir / name).write_bytes(content)
                    for name in original.keys() - proposed.keys():
                        (self.settings.pdf_dir / name).unlink()
                    os.replace(temporary, self.settings.index_path)
                except Exception:
                    for name in proposed.keys() - original.keys():
                        (self.settings.pdf_dir / name).unlink(missing_ok=True)
                    for name, content in original.items():
                        (self.settings.pdf_dir / name).write_bytes(content)
                    raise
            finally:
                Path(temporary).unlink(missing_ok=True)
            return Index(metadata, chunks, vectors)

    def retrieve(self, question, index):
        embedder = self.embedder()
        if embedder.signature != index.metadata["embedding_signature"]:
            raise ValueError("The index needs rebuilding for the current search model.")
        return rank(index, embedder.query(question), self.settings.top_k, self.settings.min_similarity)
