from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GGUF_REPO = "bartowski/Qwen_Qwen3-1.7B-GGUF"
GGUF_FILE = "Qwen_Qwen3-1.7B-Q4_K_M.gguf"
MAX_FILES = 10
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 50 * 1024 * 1024
MAX_PAGES_PER_FILE = 20
MAX_TOTAL_PAGES = 100
CHUNK_TOKENS = 320
CHUNK_OVERLAP = 40
MAX_QUESTION_CHARS = 1500
ABSTENTION = "I couldn't find enough information in the documents to answer that question."


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_dir: Path
    threads: int = 2
    context_tokens: int = 4096
    answer_tokens: int = 256
    top_k: int = 4
    min_similarity: float = 0.45

    @classmethod
    def from_env(cls):
        settings = cls(
            data_dir=Path(os.getenv("RAG_DATA_DIR", str(ROOT / "data"))).resolve(),
            model_dir=Path(os.getenv("RAG_MODEL_DIR", str(ROOT / "models"))).resolve(),
            threads=int(os.getenv("RAG_THREADS", "2")),
            context_tokens=int(os.getenv("RAG_CONTEXT_TOKENS", "4096")),
            answer_tokens=int(os.getenv("RAG_ANSWER_TOKENS", "256")),
            top_k=int(os.getenv("RAG_TOP_K", "4")),
            min_similarity=float(os.getenv("RAG_MIN_SIMILARITY", "0.45")),
        )
        if settings.threads < 1 or not 1 <= settings.top_k <= 8:
            raise ValueError("RAG_THREADS must be positive; RAG_TOP_K must be between 1 and 8.")
        if not 1024 <= settings.context_tokens <= 8192:
            raise ValueError("RAG_CONTEXT_TOKENS must be between 1024 and 8192.")
        if not 32 <= settings.answer_tokens <= min(1024, settings.context_tokens // 2):
            raise ValueError("RAG_ANSWER_TOKENS must leave room for the question and evidence.")
        if not 0 <= settings.min_similarity <= 1:
            raise ValueError("RAG_MIN_SIMILARITY must be between 0 and 1.")
        return settings

    @property
    def pdf_dir(self):
        return self.data_dir / "pdfs"

    @property
    def index_path(self):
        return self.data_dir / "index.npz"

    @property
    def embedding_dir(self):
        return self.model_dir / "embeddings"

    @property
    def llm_path(self):
        return self.model_dir / GGUF_FILE

    def ensure_dirs(self):
        for path in (self.data_dir, self.pdf_dir, self.model_dir):
            path.mkdir(parents=True, exist_ok=True)
