from __future__ import annotations

from pathlib import Path

from iaei.v2.neural_forecasting import execute_neural_forecasting_gate


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    artifacts = execute_neural_forecasting_gate(ROOT)
    leaderboard = artifacts.candidate_leaderboard
    print(
        "Gate 6C execution complete | candidates={} | seeds={} | "
        "recommendation={}".format(
            leaderboard["algorithm_id"].nunique(),
            artifacts.execution_manifest["seed_count"],
            artifacts.promotion_recommendation["recommendation"],
        )
    )
    print(leaderboard.to_csv(index=False))


if __name__ == "__main__":
    main()
