from __future__ import annotations

from pathlib import Path

from iaei.v2.neural_forecasting import execute_neural_forecasting_gate


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    artifacts = execute_neural_forecasting_gate(ROOT)
    leaderboard = artifacts.candidate_leaderboard.sort_values("mean_mae", kind="stable")
    best = leaderboard.iloc[0]
    print(
        "Gate 6C2 validation evidence: PASS | candidate={} | mean_mae={:.6f} | "
        "eligible={} | status={}".format(
            best["algorithm_id"],
            float(best["mean_mae"]),
            bool(best["promotion_eligible"]),
            artifacts.execution_manifest["status"],
        )
    )


if __name__ == "__main__":
    main()
