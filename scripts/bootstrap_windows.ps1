$ErrorActionPreference = "Stop"

python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python scripts/run_pipeline.py --check
pytest

Write-Host "Foundation ready. Run 'python scripts/download_data.py' for the licensed real dataset."
