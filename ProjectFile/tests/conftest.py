from hashlib import sha256
from io import BytesIO
import re

import numpy as np
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from rag.config import Settings


def pdf_bytes(*pages):
    """Small real PDFs for extraction tests; no external files or model downloads."""
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_ref = writer._add_object(font)
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
        })
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 740 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    target = BytesIO()
    writer.write(target)
    return target.getvalue()


class FakeEmbedder:
    def __init__(self, settings):
        self.tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        settings.embedding_dir.mkdir(parents=True, exist_ok=True)
        manifest = settings.embedding_dir / "manifest.json"
        manifest.write_text("test-embedding-version-1")
        self.signature = manifest.read_text()

    def vector(self, text):
        vector = np.zeros(384, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vector[int.from_bytes(sha256(word.encode()).digest()[:4], "big") % 384] += 1
        return vector

    def passages(self, texts):
        return np.asarray([self.vector(t) for t in texts])

    def query(self, text):
        return self.vector(text)


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path / "data", tmp_path / "models")


@pytest.fixture
def corpus(settings):
    from rag.index import Corpus
    return Corpus(settings, embedder_factory=FakeEmbedder)
