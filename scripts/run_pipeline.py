from __future__ import annotations

import argparse
from pathlib import Path

from iaei.contracts import ContractError, validate_repository_contracts
from iaei.data import write_silver_artifacts


ROOT = Path(__file__).resolve().parents[1]


def check_foundation() -> None:
    validate_repository_contracts()

    required_dirs = [
        ROOT / "data" / "manifests",
        ROOT / "data" / "processed",
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


def build_silver() -> None:
    check_foundation()
    manifest = write_silver_artifacts(ROOT)

    print(
        "Silver analytical layer: PASS | "
        f"rows={manifest['output']['row_count']} | "
        f"columns={manifest['output']['column_count']} | "
        f"dq_any={manifest['quality']['quality_flag_counts']['dq_any']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate repository foundation.",
    )
    parser.add_argument(
        "--silver",
        action="store_true",
        help="Build and validate the governed Silver analytical layer.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run implemented analytical stages.",
    )

    args = parser.parse_args()

    if args.check:
        check_foundation()
        return

    if args.silver:
        build_silver()
        return

    if args.all:
        build_silver()
        raise ContractError(
            "Decision Gates 4-7 remain locked. "
            "No model, drift, optimization, or reporting evidence "
            "will be manufactured."
        )

    parser.error("Choose --check, --silver, or --all")


if __name__ == "__main__":
    main()
