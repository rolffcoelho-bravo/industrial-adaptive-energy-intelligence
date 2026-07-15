.PHONY: install check test data pipeline brief

install:
	python -m pip install -e ".[dev]"

check:
	python scripts/run_pipeline.py --check
	ruff check src scripts tests

test:
	pytest

data:
	python scripts/download_data.py

pipeline:
	python scripts/run_pipeline.py --all

brief:
	python scripts/build_brief.py
