from __future__ import annotations

from pathlib import Path

from iaei.v2.advanced_tabular import execute_advanced_tabular_gate


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    artifacts = execute_advanced_tabular_gate(ROOT)
    leaderboard = artifacts.candidate_leaderboard.sort_values(
        "mean_mae",
        kind="stable",
    )
    best = leaderboard.iloc[0]
    print(
        "Gate 6B validation evidence: PASS | candidate={} | mean_mae={:.6f} | "
        "eligible={} | recommendation={}".format(
            best["algorithm_id"],
            float(best["mean_mae"]),
            bool(best["promotion_eligible"]),
            artifacts.promotion_recommendation["recommendation"],
        )
    )


if __name__ == "__main__":
    main()
