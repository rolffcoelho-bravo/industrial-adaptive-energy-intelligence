from __future__ import annotations

import argparse
from pathlib import Path

from iaei.v2.foundation_evidence import build_foundation_evidence


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate complete Gate 6D2 foundation-model evidence."
    )
    parser.add_argument("--staging-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_foundation_evidence(ROOT, args.staging_root.resolve())
    print(
        "Gate 6D2 aggregate evidence: PASS | candidates={} | predictions={} | "
        "locked_test={} | next_gate={}".format(
            int(manifest["candidate_count"]),
            int(manifest["prediction_row_count"]),
            bool(manifest["locked_test_accessed"]),
            manifest["next_gate"],
        )
    )


if __name__ == "__main__":
    main()
