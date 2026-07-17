from __future__ import annotations

import argparse
from pathlib import Path

from iaei.modeling.locked_test_harness import (
    execute_locked_test_once,
)


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the single authorized Gate 4E locked-test run"
        )
    )
    parser.add_argument(
        "--authorization",
        required=True,
        help="Exact authorization phrase derived from the contract",
    )
    parser.add_argument(
        "--expected-commit",
        required=True,
        help="Exact CI-green commit authorized for execution",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    run = execute_locked_test_once(
        ROOT,
        authorization=args.authorization,
        expected_commit=args.expected_commit,
    )
    aggregate = run.evaluation.results["metrics"]["aggregate"]
    peak = run.evaluation.results["metrics"]["peak_state"]

    print(
        "Gate 4E locked-test evaluation: COMPLETE | "
        f"rows={run.evaluation.results['prediction_row_count']} | "
        f"candidate_mae={aggregate['candidate_mae']:.6f} | "
        f"persistence_mae={aggregate['persistence_mae']:.6f} | "
        f"peak_candidate_mae={peak['candidate_mae']:.6f} | "
        f"peak_persistence_mae={peak['persistence_mae']:.6f}"
    )
    print(f"Predictions: {run.predictions_path}")
    print(f"Results: {run.results_path}")


if __name__ == "__main__":
    main()
