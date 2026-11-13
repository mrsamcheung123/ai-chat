"""Prepare backend PDFs before starting the public WebUI."""
import os
import sys

from .config import ROOT, Settings
from .index import Corpus


def main():
    settings = Settings.from_env()
    try:
        _, issue = Corpus(settings).refresh_if_needed(progress=lambda text: print(text, flush=True))
        if issue:
            print(f"Host setup: {issue}", flush=True)
    except Exception as exc:
        # Keep host tools accessible so the administrator can resolve setup errors.
        print(f"Host library needs attention: {exc}", file=sys.stderr, flush=True)
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), *sys.argv[1:]])


if __name__ == "__main__":
    main()
