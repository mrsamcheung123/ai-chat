"""Local administrator launcher; ordinary users only need the web address."""
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def main():
    if sys.version_info < (3, 11):
        raise SystemExit("Please install Python 3.11 or newer before starting PDF Desk.")
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        print("Preparing PDF Desk for the first time…", flush=True)
        venv.create(environment, with_pip=True)
    requirements = ROOT / "requirements.txt"
    digest = sha256(requirements.read_bytes()).hexdigest()
    stamp = environment / ".requirements-sha256"
    if not stamp.is_file() or stamp.read_text().strip() != digest:
        print("Installing application dependencies. The first setup can take several minutes…", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], cwd=ROOT, check=True)
        stamp.write_text(digest)
    print("Opening at http://localhost:8501. Keep this window open while using the app.", flush=True)
    subprocess.run([
        str(python), "-m", "rag.server",
        "--server.address=127.0.0.1", "--server.port=8501", "--server.headless=false",
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"Setup could not finish: {exc}", file=sys.stderr)
        print("See README.md for compiler prerequisites or the Docker installation option.", file=sys.stderr)
        raise SystemExit(1)
