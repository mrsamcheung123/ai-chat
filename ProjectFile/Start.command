#!/bin/zsh
set -e
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3.11 or newer is needed for local setup. Install Python, then open this launcher again."
    read -r "?Press Enter to close."
    exit 1
fi
python3 launcher.py
