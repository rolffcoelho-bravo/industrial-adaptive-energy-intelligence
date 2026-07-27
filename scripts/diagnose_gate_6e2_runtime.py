from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from iaei.contracts import load_yaml
from iaei.modeling.splits import build_expanding_window_folds
from iaei.paths import ROOT
from iaei.v2.point_prediction_reconstruction import rebuild_frozen_point_predictions


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected an object in {path}")
    return value


def main() -> None:
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
    silver = pd.read_parquet(
        ROOT / "data" / "processed" / "steel_energy_silver.parquet"
    )
    folds = build_expanding_window_folds(
        silver["effective_timestamp"],
        model_contract,
    )
    _, verification = rebuild_frozen_point_predictions(
        silver,
        folds,
        model_contract,
        target_contract,
        point_manifest,
        committed_results,
        strict_metric_verification=False,
    )
    print(json.dumps(verification, indent=2, sort_keys=True, allow_nan=False))
    if not verification["committed_fold_metrics_verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
