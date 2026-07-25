from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from iaei.v2.advanced_tabular import (
    _validate_trial_record,
    validate_advanced_tabular_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    contract = validate_advanced_tabular_contract(ROOT)
    output_directory = ROOT / str(contract["outputs"]["directory"])
    manifest_path = output_directory / str(contract["outputs"]["execution_manifest"])
    if not manifest_path.exists():
        raise SystemExit("Gate 6B execution manifest is missing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["gate"] != "6B":
        raise SystemExit("Unexpected Gate 6B manifest identity")
    if manifest["status"] != "validation_complete_pending_human_decision":
        raise SystemExit("Gate 6B is not at the expected decision boundary")
    if manifest["v1_immutable"] is not True:
        raise SystemExit("Gate 6B manifest does not preserve V1")
    if manifest["locked_test_accessed"] is not False:
        raise SystemExit("Gate 6B accessed the locked test")
    if manifest["locked_predictions_parsed"] is not False:
        raise SystemExit("Gate 6B parsed locked predictions")
    if manifest["confirmatory_evaluation_performed"] is not False:
        raise SystemExit("Gate 6B performed a confirmatory evaluation")
    if int(manifest["unique_configuration_count"]) != 12:
        raise SystemExit("Gate 6B configuration count changed")
    if int(manifest["outer_fold_count"]) != 4:
        raise SystemExit("Gate 6B outer-fold count changed")
    if int(manifest["inner_fold_count"]) != 3:
        raise SystemExit("Gate 6B inner-fold count changed")
    if int(manifest["maximum_prediction_origin"]) >= 28028:
        raise SystemExit("Gate 6B predictions reached the locked-test purge")
    if int(manifest["maximum_target_dependency"]) >= 28032:
        raise SystemExit("Gate 6B target dependency reached the locked test")

    for artifact in manifest["artifacts"].values():
        path = ROOT / str(artifact["path"])
        if not path.exists():
            raise SystemExit(f"Gate 6B artifact is missing: {path}")
        if _sha256(path) != artifact["sha256"]:
            raise SystemExit(f"Gate 6B artifact hash changed: {path}")

    trial_path = output_directory / str(contract["outputs"]["trial_evidence"])
    trials = json.loads(trial_path.read_text(encoding="utf-8"))["trials"]
    if len(trials) != 12:
        raise SystemExit("Gate 6B trial-evidence count changed")
    for record in trials:
        _validate_trial_record(ROOT, record)

    inner = pd.read_csv(
        output_directory / str(contract["outputs"]["inner_search_results"])
    )
    outer = pd.read_csv(
        output_directory / str(contract["outputs"]["outer_fold_results"])
    )
    leaderboard = pd.read_csv(
        output_directory / str(contract["outputs"]["candidate_leaderboard"])
    )
    predictions = pd.read_parquet(
        output_directory / str(contract["outputs"]["out_of_fold_predictions"])
    )

    if len(inner) != 3 * 4 * 4 * 3:
        raise SystemExit("Gate 6B inner-search evidence is incomplete")
    if len(outer) != 3 * 4:
        raise SystemExit("Gate 6B outer-fold evidence is incomplete")
    if len(leaderboard) != 3:
        raise SystemExit("Gate 6B leaderboard is incomplete")
    if predictions["algorithm_id"].nunique() != 3:
        raise SystemExit("Gate 6B prediction evidence is incomplete")
    if predictions.groupby("algorithm_id")["row_position"].nunique().nunique() != 1:
        raise SystemExit("Gate 6B candidates do not cover identical origins")
    if int(predictions["row_position"].max()) >= 28028:
        raise SystemExit("Gate 6B predictions cross the governed boundary")

    numeric = leaderboard.select_dtypes(include="number")
    if not numeric.apply(lambda series: series.map(pd.notna).all()).all():
        raise SystemExit("Gate 6B leaderboard contains missing numeric evidence")

    print(
        "Gate 6B evidence validation: PASS | candidates={} | origins_per_candidate={} | "
        "recommendation={}".format(
            leaderboard["algorithm_id"].nunique(),
            manifest["validation_origin_count_per_candidate"],
            manifest["recommendation"]["recommendation"],
        )
    )


if __name__ == "__main__":
    main()
