from __future__ import annotations

import argparse
from pathlib import Path

from iaei.contracts import ContractError, validate_repository_contracts

ROOT = Path(__file__).resolve().parents[1]


def check_foundation() -> None:
    validate_repository_contracts()
    required_dirs = [
        ROOT / "data" / "manifests",
        ROOT / "outputs" / "charts",
        ROOT / "outputs" / "tables",
        ROOT / "outputs" / "brief",
        ROOT / "notebooks" / "databricks",
        ROOT / "sql",
    ]
    missing = [str(path) for path in required_dirs if not path.exists()]
    if missing:
        raise ContractError(f"Missing required directories: {missing}")
    print("Foundation contracts: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate repository foundation.")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline when implemented.")
    args = parser.parse_args()

    if args.check:
        check_foundation()
        return
    if args.all:
        check_foundation()
        raise ContractError(
            "Full analytical stages are intentionally gated. Complete Decision Gates 1-7 "
            "before enabling --all; no placeholder results will be generated."
        )
    parser.error("Choose --check or --all")


if __name__ == "__main__":
    main()
