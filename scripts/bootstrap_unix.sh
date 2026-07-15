#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
python scripts/run_pipeline.py --check
pytest

echo "Foundation ready. Run 'python scripts/download_data.py' for the licensed real dataset."
