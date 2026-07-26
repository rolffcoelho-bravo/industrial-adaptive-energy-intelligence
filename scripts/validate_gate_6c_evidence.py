from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from iaei.v2.neural_forecasting import (
    _validate_trial_record,
    validate_neural_forecasting_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    contract = validate_neural_forecasting_contract(ROOT)
    output_directory = ROOT / str(contract["outputs"]["directory"])
    manifest_path = output_directory / str(contract["outputs"]["execution_manifest"])
    if not manifest_path.exists():
        raise SystemExit("Gate 6C execution manifest is missing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["gate"] != "6C":
        raise SystemExit("Unexpected Gate 6C manifest identity")
    if manifest["status"] != "validation_complete_pending_human_decision":
        raise SystemExit("Gate 6C is not at the expected decision boundary")
    if manifest["v1_immutable"] is not True:
        raise SystemExit("Gate 6C manifest does not preserve V1")
    if manifest["locked_test_accessed"] is not False:
        raise SystemExit("Gate 6C accessed the locked test")
    if manifest["locked_predictions_parsed"] is not False:
        raise SystemExit("Gate 6C parsed locked predictions")
    if manifest["confirmatory_evaluation_performed"] is not False:
        raise SystemExit("Gate 6C performed a confirmatory evaluation")
    if int(manifest["unique_configuration_count"]) != 6:
        raise SystemExit("Gate 6C configuration count changed")
    if int(manifest["outer_fold_count"]) != 4:
        raise SystemExit("Gate 6C outer-fold count changed")
    if int(manifest["inner_fold_count"]) != 3:
        raise SystemExit("Gate 6C inner-fold count changed")
    if int(manifest["seed_count"]) != 5:
        raise SystemExit("Gate 6C seed count changed")
    if int(manifest["maximum_prediction_origin"]) >= 28028:
        raise SystemExit("Gate 6C predictions reached the locked-test purge")
    if int(manifest["maximum_target_dependency"]) >= 28032:
        raise SystemExit("Gate 6C target dependency reached the locked test")

    for artifact in manifest["artifacts"].values():
        path = ROOT / str(artifact["path"])
        if not path.exists():
            raise SystemExit(f"Gate 6C artifact is missing: {path}")
        if _sha256(path) != artifact["sha256"]:
            raise SystemExit(f"Gate 6C artifact hash changed: {path}")

    dependency_lock = ROOT / str(manifest["dependency_lock"]["path"])
    if not dependency_lock.exists():
        raise SystemExit("Gate 6C environment lock is missing")
    if _sha256(dependency_lock) != manifest["dependency_lock"]["sha256"]:
        raise SystemExit("Gate 6C environment lock hash changed")

    trial_path = output_directory / str(contract["outputs"]["trial_evidence"])
    trials = json.loads(trial_path.read_text(encoding="utf-8"))["trials"]
    if len(trials) != 3:
        raise SystemExit("Gate 6C trial-evidence count changed")
    expected_seeds = [int(value) for value in contract["training"]["stochastic_seeds"]]
    for record in trials:
        _validate_trial_record(ROOT, record)
        if [int(value) for value in record["seeds"]] != expected_seeds:
            raise SystemExit("Gate 6C trial seed evidence changed")
        if len(record["objective_records"]) != 4 * 5:
            raise SystemExit("Gate 6C objective evidence is incomplete")

    inner = pd.read_csv(
        output_directory / str(contract["outputs"]["inner_search_results"])
    )
    outer = pd.read_csv(
        output_directory / str(contract["outputs"]["outer_seed_results"])
    )
    seed_summary = pd.read_csv(
        output_directory / str(contract["outputs"]["seed_summary"])
    )
    leaderboard = pd.read_csv(
        output_directory / str(contract["outputs"]["candidate_leaderboard"])
    )
    predictions = pd.read_parquet(
        output_directory / str(contract["outputs"]["out_of_fold_predictions"])
    )

    if len(inner) != 3 * 4 * 3 * 2:
        raise SystemExit("Gate 6C inner-search evidence is incomplete")
    if len(outer) != 3 * 4 * 5:
        raise SystemExit("Gate 6C outer-seed evidence is incomplete")
    if len(seed_summary) != 3 * 5:
        raise SystemExit("Gate 6C seed summary is incomplete")
    if len(leaderboard) != 3:
        raise SystemExit("Gate 6C leaderboard is incomplete")
    if predictions["algorithm_id"].nunique() != 3:
        raise SystemExit("Gate 6C prediction evidence is incomplete")
    if predictions["seed"].nunique() != 5:
        raise SystemExit("Gate 6C prediction seed evidence is incomplete")

    origins = predictions.groupby(["algorithm_id", "seed"])["row_position"].nunique()
    if origins.nunique() != 1 or int(origins.iloc[0]) != 7004:
        raise SystemExit("Gate 6C candidates do not cover identical origins")
    if int(predictions["row_position"].max()) >= 28028:
        raise SystemExit("Gate 6C predictions cross the governed boundary")
    if predictions.duplicated(["algorithm_id", "seed", "row_position"]).any():
        raise SystemExit("Gate 6C prediction evidence contains duplicates")

    for algorithm_id, rows in outer.groupby("algorithm_id"):
        if rows["seed"].nunique() != 5:
            raise SystemExit(f"Gate 6C seed evidence is incomplete for {algorithm_id}")
        if rows["fold_id"].nunique() != 4:
            raise SystemExit(f"Gate 6C fold evidence is incomplete for {algorithm_id}")

    numeric_frames = [inner, outer, seed_summary, leaderboard]
    for frame in numeric_frames:
        numeric = frame.select_dtypes(include="number")
        if not numeric.apply(lambda series: series.map(pd.notna).all()).all():
            raise SystemExit("Gate 6C evidence contains missing numeric values")

    recommendation = manifest["recommendation"]
    if recommendation["human_decision_required"] is not True:
        raise SystemExit("Gate 6C bypassed human promotion authority")

    print(
        "Gate 6C evidence validation: PASS | candidates={} | seeds={} | "
        "origins_per_candidate_seed={} | recommendation={}".format(
            leaderboard["algorithm_id"].nunique(),
            manifest["seed_count"],
            manifest["validation_origin_count_per_candidate_seed"],
            recommendation["recommendation"],
        )
    )


if __name__ == "__main__":
    main()
