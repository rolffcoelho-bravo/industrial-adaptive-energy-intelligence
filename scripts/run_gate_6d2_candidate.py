from __future__ import annotations

import argparse
from pathlib import Path

from iaei.v2.foundation_forecasting import execute_foundation_candidate


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute one pinned Gate 6D2 foundation-model candidate."
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_id = str(args.candidate_id)
    staging_root = args.staging_root.resolve()
    result = execute_foundation_candidate(
        ROOT,
        candidate_id=candidate_id,
        output_directory=staging_root / "candidates" / candidate_id,
        cache_directory=staging_root / "model_cache",
    )
    print(
        "Gate 6D2 candidate: PASS | candidate={} | mean_mae={:.6f} | "
        "peak_mae={:.6f} | deterministic={}".format(
            candidate_id,
            float(result["mean_mae"]),
            float(result["mean_peak_mae"]),
            bool(result["deterministic_replay_passed"]),
        )
    )


if __name__ == "__main__":
    main()
