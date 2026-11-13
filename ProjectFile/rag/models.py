"""Model downloads are explicit; normal embedding and generation are local-only."""
import json
import os
import ctypes
from pathlib import Path
import threading

import numpy as np

from .config import EMBEDDING_MODEL, GGUF_FILE, GGUF_REPO

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def disable_telemetry():
    import onnxruntime
    onnxruntime.disable_telemetry_events()


def embedding_ready(settings):
    manifest = settings.embedding_dir / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        info = json.loads(manifest.read_text())
        return info["model"] == EMBEDDING_MODEL and all(
            (settings.embedding_dir / name).is_file() for name in info["files"]
        )
    except (ValueError, KeyError, OSError):
        return False


def prepare_models(settings, progress=lambda message: None):
    os.environ.setdefault("HF_HOME", str(settings.model_dir / ".hub"))
    os.environ.setdefault("HF_XET_CACHE", str(settings.model_dir / ".xet"))
    disable_telemetry()
    from fastembed import TextEmbedding
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    settings.ensure_dirs()
    if not embedding_ready(settings):
        progress("Downloading the English search model…")
        model = next(m for m in TextEmbedding.list_supported_models() if m["model"] == EMBEDDING_MODEL)
        repo = model["sources"]["hf"]
        revision = HfApi().model_info(repo).sha
        snapshot_download(
            repo_id=repo,
            revision=revision,
            allow_patterns=[model["model_file"], "*.json", "*.txt"],
            local_dir=str(settings.embedding_dir),
            cache_dir=str(settings.model_dir / ".hub"),
        )
        files = sorted(
            str(p.relative_to(settings.embedding_dir))
            for p in settings.embedding_dir.rglob("*")
            if p.is_file() and ".cache" not in p.parts and p.name != "manifest.json"
        )
        (settings.embedding_dir / "manifest.json").write_text(json.dumps({
            "model": EMBEDDING_MODEL, "repository": repo, "revision": revision, "files": files,
        }, indent=2))
    if not settings.llm_path.is_file():
        progress("Downloading the answer model (about 1.28 GB). You can keep this page open…")
        revision = HfApi().model_info(GGUF_REPO).sha
        hf_hub_download(
            repo_id=GGUF_REPO, filename=GGUF_FILE, revision=revision,
            local_dir=str(settings.model_dir),
            cache_dir=str(settings.model_dir / ".hub"),
        )
        (settings.model_dir / "llm-manifest.json").write_text(json.dumps({
            "repository": GGUF_REPO, "revision": revision, "filename": GGUF_FILE,
        }, indent=2))
    progress("Models are downloaded and ready for local use.")


class Embedder:
    def __init__(self, settings):
        disable_telemetry()
        from fastembed import TextEmbedding
        from tokenizers import Tokenizer

        if not embedding_ready(settings):
            raise RuntimeError("Download the local models from Manage documents first.")
        self.model = TextEmbedding(
            model_name=EMBEDDING_MODEL,
            specific_model_path=str(settings.embedding_dir),
            local_files_only=True,
            threads=settings.threads,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(str(settings.embedding_dir / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.signature = (settings.embedding_dir / "manifest.json").read_text()
        self.lock = threading.Lock()

    def passages(self, texts):
        with self.lock:
            return np.asarray(list(self.model.passage_embed(texts, batch_size=8)), dtype=np.float32)

    def query(self, text):
        with self.lock:
            return np.asarray(list(self.model.query_embed(text)), dtype=np.float32)[0]


def load_llm(settings):
    if not settings.llm_path.is_file():
        raise RuntimeError("Download the local answer model from Manage documents first.")
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("The answer-model runtime is missing. Install requirements.txt and restart the app.") from exc
    class CpuLlama(Llama):
        # The high-level API does not expose llama_model_params.devices. An
        # explicit empty device list prevents GPU backend initialization even
        # when the installed macOS wheel was built with Metal enabled.
        @property
        def model_params(self):
            return self._cpu_model_params

        @model_params.setter
        def model_params(self, params):
            self._device_list = (ctypes.c_void_p * 1)()
            params.devices = ctypes.cast(self._device_list, ctypes.c_void_p)
            self._cpu_model_params = params

    return CpuLlama(
        model_path=str(Path(settings.llm_path)),
        n_ctx=settings.context_tokens,
        n_threads=settings.threads,
        n_threads_batch=settings.threads,
        n_batch=128,
        n_gpu_layers=0,
        offload_kqv=False,
        op_offload=False,
        use_mmap=True,
        verbose=False,
        seed=42,
    )
