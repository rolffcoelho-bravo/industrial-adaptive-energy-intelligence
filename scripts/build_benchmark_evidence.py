from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from iaei.contracts import load_yaml
from iaei.modeling.benchmarks import evaluate_benchmarks
from iaei.modeling.splits import build_expanding_window_folds


ROOT = Path(__file__).resolve().parents[1]
SILVER_PATH = (
    ROOT
    / "data"
    / "processed"
    / "steel_energy_silver.parquet"
)
MODEL_CONTRACT_PATH = ROOT / "configs" / "model_contract.yml"
TARGET_CONTRACT_PATH = ROOT / "configs" / "target_contract.yml"
OUTPUT_DIRECTORY = ROOT / "outputs" / "modeling"


def main() -> None:
    silver = pd.read_parquet(SILVER_PATH)
    model_contract = load_yaml(MODEL_CONTRACT_PATH)
    target_contract = load_yaml(TARGET_CONTRACT_PATH)

    folds = build_expanding_window_folds(
        silver["effective_timestamp"],
        model_contract,
    )
    evaluation = evaluate_benchmarks(
        silver,
        folds,
        model_contract,
        target_contract=target_contract,
    )

    regression = evaluation.regression_results.sort_values(
        ["fold_id", "benchmark"],
        kind="stable",
    ).reset_index(drop=True)
    classification = evaluation.classification_results.sort_values(
        ["fold_id", "benchmark"],
        kind="stable",
    ).reset_index(drop=True)
    predictions = evaluation.predictions.sort_values(
        ["row_position", "task", "benchmark"],
        kind="stable",
    ).reset_index(drop=True)

    mean_mae = regression.groupby("benchmark")["mae"].mean()
    strongest_naive = str(mean_mae.idxmin())
    maximum_origin = int(predictions["row_position"].max())
    horizon_steps = int(
        model_contract["objectives"][
            "classification_horizon_minutes"
        ]
        // 15
    )
    maximum_target_dependency = maximum_origin + horizon_steps

    if maximum_origin >= folds[0].test_purge_start:
        raise RuntimeError(
            "Prediction origins enter the locked-test purge"
        )

    if maximum_target_dependency >= folds[0].test_start:
        raise RuntimeError(
            "Validation targets consume locked-test observations"
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    regression_path = (
        OUTPUT_DIRECTORY / "regression_validation_results.csv"
    )
    classification_path = (
        OUTPUT_DIRECTORY / "peak_validation_results.csv"
    )
    predictions_path = (
        OUTPUT_DIRECTORY / "out_of_fold_predictions.parquet"
    )
    manifest_path = OUTPUT_DIRECTORY / "benchmark_manifest.json"

    regression.to_csv(regression_path, index=False)
    classification.to_csv(classification_path, index=False)
    predictions.to_parquet(predictions_path, index=False)

    mean_peak_mae = regression.groupby("benchmark")[
        "peak_mae"
    ].mean()
    worst_fold_mae = regression.groupby("benchmark")[
        "mae"
    ].max()

    manifest = {
        "contract_version": str(
            model_contract["contract_version"]
        ),
        "governance_gate": "4B",
        "status": "validated",
        "silver_row_count": int(len(silver)),
        "validation_fold_count": int(len(folds)),
        "validation_origin_count": int(
            predictions["row_position"].nunique()
        ),
        "prediction_row_count": int(len(predictions)),
        "maximum_prediction_origin": maximum_origin,
        "maximum_target_dependency": maximum_target_dependency,
        "test_purge_start": int(folds[0].test_purge_start),
        "test_purge_stop": int(folds[0].test_purge_stop),
        "locked_test_start": int(folds[0].test_start),
        "locked_test_stop": int(folds[0].test_stop),
        "strongest_naive_regression_reference": strongest_naive,
        "classification_reference": "training_prevalence",
        "regression_mean_mae": {
            str(name): float(value)
            for name, value in mean_mae.items()
        },
        "regression_mean_peak_mae": {
            str(name): float(value)
            for name, value in mean_peak_mae.items()
        },
        "regression_worst_fold_mae": {
            str(name): float(value)
            for name, value in worst_fold_mae.items()
        },
        "classification_mean_metrics": {
            "pr_auc": float(classification["pr_auc"].mean()),
            "roc_auc": float(classification["roc_auc"].mean()),
            "brier_score": float(
                classification["brier_score"].mean()
            ),
            "log_loss": float(
                classification["log_loss"].mean()
            ),
            "expected_calibration_error": float(
                classification[
                    "expected_calibration_error"
                ].mean()
            ),
            "worst_fold_recall": float(
                classification[
                    "recall_at_controlled_alert_rate"
                ].min()
            ),
        },
        "artifacts": {
            "regression_validation_results": str(
                regression_path.relative_to(ROOT).as_posix()
            ),
            "peak_validation_results": str(
                classification_path.relative_to(ROOT).as_posix()
            ),
            "out_of_fold_predictions": str(
                predictions_path.relative_to(ROOT).as_posix()
            ),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "Benchmark evidence: PASS | origins={} | predictions={} | "
        "reference={}".format(
            manifest["validation_origin_count"],
            manifest["prediction_row_count"],
            strongest_naive,
        )
    )


if __name__ == "__main__":
    main()
