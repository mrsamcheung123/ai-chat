from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_TOKENS, MAX_FILE_BYTES, MAX_PAGES_PER_FILE


class DocumentError(ValueError):
    pass


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class Document:
    name: str
    digest: str
    size: int
    page_count: int
    pages: list[Page]


@dataclass(frozen=True)
class Chunk:
    id: str
    document: str
    page: int
    text: str

    def to_dict(self):
        return asdict(self)


def safe_filename(name: str) -> str:
    # Reject paths rather than silently accepting an upload outside the PDF directory.
    if "/" in name or "\\" in name or Path(name).name != name:
        raise DocumentError("Please upload a PDF filename without a folder path.")
    clean = re.sub(r"[^A-Za-z0-9 ._()-]", "_", name).strip(" .")
    if not clean.lower().endswith(".pdf") or len(clean[:-4].strip(" ._")) == 0:
        raise DocumentError("Each document must have a name ending in .pdf.")
    if len(clean) > 150:
        raise DocumentError("Please shorten PDF filenames to 150 characters or fewer.")
    return clean


def read_pdf(name: str, content: bytes) -> Document:
    name = safe_filename(name)
    if len(content) > MAX_FILE_BYTES:
        raise DocumentError(f"{name}: the maximum file size is 10 MB.")
    if not content.startswith(b"%PDF-"):
        raise DocumentError(f"{name}: this does not appear to be a PDF file.")
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise DocumentError(f"{name}: password-protected PDFs are not supported.")
        count = len(reader.pages)
        if count == 0 or count > MAX_PAGES_PER_FILE:
            raise DocumentError(f"{name}: upload a PDF with 1–{MAX_PAGES_PER_FILE} pages.")
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\x00", "")
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            # Bound pathological extraction output before tokenization.
            if len(text) > 100_000:
                raise DocumentError(f"{name}, page {number}: the extracted text is too large.")
            if text:
                pages.append(Page(number, text))
        if not pages:
            raise DocumentError(f"{name}: no selectable text was found. Scanned PDFs need OCR first.")
        return Document(name, sha256(content).hexdigest(), len(content), count, pages)
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"{name}: the PDF could not be read. Try exporting it again.") from exc


def chunk_document(document: Document, tokenizer, size=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    if not 0 <= overlap < size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")
    chunks = []
    for page in document.pages:
        encoding = tokenizer.encode(page.text, add_special_tokens=False)
        offsets = encoding.offsets
        start = 0
        while start < len(offsets):
            end = min(start + size, len(offsets))
            text = page.text[offsets[start][0]:offsets[end - 1][1]].strip()
            if text:
                identity = f"{document.name}:{document.digest}:{page.number}:{start}"
                chunks.append(Chunk(sha256(identity.encode()).hexdigest()[:20], document.name, page.number, text))
            if end == len(offsets):
                break
            start = end - overlap
    return chunks
