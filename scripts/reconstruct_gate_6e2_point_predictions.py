from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from iaei.contracts import load_yaml
from iaei.modeling.splits import build_expanding_window_folds
from iaei.paths import ROOT
from iaei.v2.point_prediction_reconstruction import rebuild_frozen_point_predictions
from iaei.v2.uncertainty_execution import _load_json


SILVER_PATH = ROOT / "data" / "processed" / "steel_energy_silver.parquet"
POINT_PATH = (
    ROOT
    / "outputs"
    / "modeling"
    / "hist_gradient_boosting_out_of_fold_predictions.parquet"
)
VERIFICATION_PATH = (
    ROOT
    / "outputs"
    / "modeling"
    / "gate_6e2_point_reconstruction_verification.json"
)


def main() -> None:
    if POINT_PATH.exists():
        print("Frozen HGB validation predictions already present; reconstruction skipped.")
        return
    model_contract = load_yaml(ROOT / "configs" / "model_contract.yml")
    target_contract = load_yaml(ROOT / "configs" / "target_contract.yml")
    point_manifest = _load_json(
        ROOT
        / "outputs"
        / "modeling"
        / "hist_gradient_boosting_candidate_manifest.json"
    )
    committed_results = pd.read_csv(
        ROOT
        / "outputs"
        / "modeling"
        / "hist_gradient_boosting_validation_results.csv"
    )
    silver = pd.read_parquet(SILVER_PATH)
    folds = build_expanding_window_folds(
        silver["effective_timestamp"],
        model_contract,
    )
    predictions, verification = rebuild_frozen_point_predictions(
        silver,
        folds,
        model_contract,
        target_contract,
        point_manifest,
        committed_results,
    )
    POINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(POINT_PATH, index=False)
    VERIFICATION_PATH.write_text(
        json.dumps(verification, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "Gate 6E2 point reconstruction: PASS | origins={} | folds={} | "
        "search=false | committed_metrics_verified=true".format(
            len(predictions),
            len(folds),
        )
    )


if __name__ == "__main__":
    main()
