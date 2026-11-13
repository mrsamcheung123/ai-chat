"""Optional administrator command for preparing models before an offline deployment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import Settings  # noqa: E402
from rag.models import prepare_models  # noqa: E402

if __name__ == "__main__":
    prepare_models(Settings.from_env(), progress=lambda message: print(message, flush=True))
