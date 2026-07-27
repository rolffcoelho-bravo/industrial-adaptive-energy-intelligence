from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

from iaei.paths import ROOT, SCHEMAS
from iaei.uncertainty_contracts import (
    validate_gate_6e1_closure_manifest,
    validate_uncertainty_contract,
)
from iaei.v2.uncertainty_calibration import (
    configurations_from_contract,
    summarize_configuration,
)


class Gate6E2EvidenceError(RuntimeError):
    """Raised when committed Gate 6E2 evidence is incomplete or inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Gate6E2EvidenceError(f"Expected an object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_schema(payload: dict[str, Any], schema_name: str, label: str) -> None:
    schema = _load_json(SCHEMAS / schema_name)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "\n".join(
            f"- {'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise Gate6E2EvidenceError(f"{label} failed schema validation:\n{details}")


def _assert_close(observed: float, expected: float, label: str) -> None:
    if not np.isclose(observed, expected, rtol=1e-10, atol=1e-12):
        raise Gate6E2EvidenceError(
            f"{label} mismatch: observed={observed}, expected={expected}"
        )


def _directory_size(path: Path) -> int:
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def validate_gate_6e2(*, allow_absent: bool = False) -> None:
    contract = validate_uncertainty_contract()
    closure = validate_gate_6e1_closure_manifest()
    if closure["status"] != "closed" or closure["next_gate"] != "6E2":
        raise Gate6E2EvidenceError("Gate 6E1 does not authorize the Gate 6E2 boundary")

    output_directory = ROOT / str(contract["outputs"]["directory"])
    manifest_path = output_directory / str(contract["outputs"]["execution_manifest"])
    if not manifest_path.exists():
        if allow_absent:
            print("Gate 6E2 evidence: ABSENT AND PERMITTED")
            return
        raise Gate6E2EvidenceError("Gate 6E2 execution manifest is absent")

    paths = {
        "configuration_results": output_directory
        / str(contract["outputs"]["configuration_results"]),
        "coverage_results": output_directory
        / str(contract["outputs"]["coverage_results"]),
        "outer_fold_results": output_directory
        / str(contract["outputs"]["outer_fold_results"]),
        "interval_predictions": output_directory
        / str(contract["outputs"]["interval_predictions"]),
        "calibration_lineage": output_directory
        / str(contract["outputs"]["calibration_lineage"]),
        "calibration_residuals": output_directory / "calibration_residuals.parquet",
        "resource_evidence": output_directory
        / str(contract["outputs"]["resource_evidence"]),
        "failure_records": output_directory
        / str(contract["outputs"]["failure_records"]),
        "promotion_recommendation": output_directory
        / str(contract["outputs"]["promotion_recommendation"]),
        "environment_lock": output_directory / "environment_lock.txt",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise Gate6E2EvidenceError(f"Gate 6E2 artifacts are missing: {missing}")

    manifest = _load_json(manifest_path)
    lineage = _load_json(paths["calibration_lineage"])
    recommendation = _load_json(paths["promotion_recommendation"])
    failures = _load_json(paths["failure_records"])
    _validate_schema(
        manifest,
        "gate_6e_execution_manifest.schema.json",
        "Gate 6E2 execution manifest",
    )
    _validate_schema(
        lineage,
        "uncertainty_calibration_lineage.schema.json",
        "Gate 6E2 calibration lineage",
    )
    _validate_schema(
        recommendation,
        "uncertainty_promotion_recommendation.schema.json",
        "Gate 6E2 promotion recommendation",
    )
    _validate_schema(
        failures,
        "uncertainty_failure_records.schema.json",
        "Gate 6E2 failure records",
    )
    if failures["records"]:
        raise Gate6E2EvidenceError("Gate 6E2 contains recorded execution failures")

    configurations = pd.read_csv(paths["configuration_results"])
    coverage = pd.read_csv(paths["coverage_results"])
    folds = pd.read_csv(paths["outer_fold_results"])
    predictions = pd.read_parquet(paths["interval_predictions"])
    residuals = pd.read_parquet(paths["calibration_residuals"])
    resources = pd.read_csv(paths["resource_evidence"])
    point = pd.read_parquet(
        ROOT
        / "outputs"
        / "modeling"
        / "hist_gradient_boosting_out_of_fold_predictions.parquet"
    ).sort_values(["fold_id", "row_position"], kind="stable")

    expected_configurations = configurations_from_contract(contract)
    expected_ids = [item.configuration_id for item in expected_configurations]
    if set(configurations["configuration_id"]) != set(expected_ids):
        raise Gate6E2EvidenceError("Configuration-result identities are incomplete")
    if len(configurations) != 9 or len(coverage) != 108 or len(folds) != 36:
        raise Gate6E2EvidenceError("Gate 6E2 summary row counts are inconsistent")
    if len(predictions) != 63036:
        raise Gate6E2EvidenceError("Gate 6E2 interval predictions are incomplete")
    if len(point) != 7004 or point["row_position"].nunique() != 7004:
        raise Gate6E2EvidenceError("Frozen point evidence is inconsistent")
    if predictions["row_position"].max() != 28027:
        raise Gate6E2EvidenceError("Gate 6E2 prediction boundary is inconsistent")
    if predictions["row_position"].min() < 21024:
        raise Gate6E2EvidenceError("Gate 6E2 predictions precede validation")
    if predictions["configuration_id"].value_counts().ne(7004).any():
        raise Gate6E2EvidenceError("A configuration does not cover 7,004 origins")
    if residuals["selected_for_initial_calibration"].sum() < 4 * 672:
        raise Gate6E2EvidenceError("Initial calibration residual evidence is incomplete")

    iqr_by_fold = {
        int(item["fold_id"]): float(item["outer_training_target_iqr"])
        for item in lineage["folds"]
    }
    levels = [float(value) for value in contract["calibration_protocol"]["target_coverage_levels"]]
    primary = float(contract["calibration_protocol"]["primary_coverage_level"])
    tolerance = float(
        contract["point_model_boundary"]["point_prediction_parity_absolute_tolerance"]
    )

    for configuration_id in expected_ids:
        observed = predictions.loc[
            predictions["configuration_id"].eq(configuration_id)
        ].sort_values(["fold_id", "row_position"], kind="stable")
        if not np.array_equal(
            observed["row_position"].to_numpy(dtype=int),
            point["row_position"].to_numpy(dtype=int),
        ):
            raise Gate6E2EvidenceError(
                f"{configuration_id} point-origin parity failed"
            )
        if not np.allclose(
            observed["point_prediction"].to_numpy(dtype=float),
            point["prediction"].to_numpy(dtype=float),
            rtol=0.0,
            atol=tolerance,
        ):
            raise Gate6E2EvidenceError(
                f"{configuration_id} point-prediction parity failed"
            )
        if not np.allclose(
            observed["actual"].to_numpy(dtype=float),
            point["actual"].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        ):
            raise Gate6E2EvidenceError(f"{configuration_id} target parity failed")
        for suffix in ("80", "90", "95"):
            if observed[f"lower_{suffix}"].lt(0.0).any():
                raise Gate6E2EvidenceError(f"{configuration_id} violates support")
            if observed[f"upper_{suffix}"].lt(observed[f"lower_{suffix}"]).any():
                raise Gate6E2EvidenceError(f"{configuration_id} has inverted bounds")
        if (
            observed["lower_95"].gt(observed["lower_90"]).any()
            or observed["lower_90"].gt(observed["lower_80"]).any()
            or observed["upper_95"].lt(observed["upper_90"]).any()
            or observed["upper_90"].lt(observed["upper_80"]).any()
        ):
            raise Gate6E2EvidenceError(f"{configuration_id} violates interval nesting")

        recalculated_coverage, recalculated_folds, aggregate = summarize_configuration(
            observed,
            levels,
            iqr_by_fold,
            primary_coverage=primary,
            rolling_window=672,
        )
        committed_config = configurations.loc[
            configurations["configuration_id"].eq(configuration_id)
        ].iloc[0]
        _assert_close(
            float(committed_config["weighted_interval_score"]),
            float(aggregate["weighted_interval_score"]),
            f"{configuration_id} weighted interval score",
        )
        _assert_close(
            float(committed_config["peak_state_weighted_interval_score"]),
            float(aggregate["peak_state_weighted_interval_score"]),
            f"{configuration_id} peak weighted interval score",
        )
        for level in levels:
            suffix = int(round(level * 100))
            _assert_close(
                float(committed_config[f"empirical_coverage_{suffix}"]),
                float(observed[f"covered_{suffix}"].mean()),
                f"{configuration_id} aggregate {suffix} coverage",
            )
        committed_coverage = coverage.loc[
            coverage["configuration_id"].eq(configuration_id)
        ].sort_values(["fold_id", "coverage_level"], kind="stable")
        recalculated_coverage = recalculated_coverage.sort_values(
            ["fold_id", "coverage_level"], kind="stable"
        )
        for column in (
            "empirical_coverage",
            "absolute_coverage_error",
            "mean_interval_width_kwh",
            "normalized_mean_interval_width",
            "interval_score",
            "peak_state_coverage",
            "peak_state_mean_interval_width_kwh",
            "peak_state_interval_score",
        ):
            if not np.allclose(
                committed_coverage[column].to_numpy(dtype=float),
                recalculated_coverage[column].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise Gate6E2EvidenceError(
                    f"{configuration_id} coverage metric {column} is inconsistent"
                )
        committed_folds = folds.loc[
            folds["configuration_id"].eq(configuration_id)
        ].sort_values("fold_id", kind="stable")
        recalculated_folds = recalculated_folds.sort_values("fold_id", kind="stable")
        for column in (
            "weighted_interval_score",
            "peak_state_weighted_interval_score",
            "primary_empirical_coverage",
            "primary_absolute_coverage_error",
            "primary_mean_interval_width_kwh",
            "maximum_rolling_672_absolute_coverage_error",
        ):
            if not np.allclose(
                committed_folds[column].to_numpy(dtype=float),
                recalculated_folds[column].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise Gate6E2EvidenceError(
                    f"{configuration_id} fold metric {column} is inconsistent"
                )

    if len(resources) != 10:
        raise Gate6E2EvidenceError("Resource evidence does not cover all execution stages")
    if not resources["cpu_execution_passed"].astype(bool).all():
        raise Gate6E2EvidenceError("CPU portability evidence failed")
    if not resources["deterministic_replay_passed"].astype(bool).all():
        raise Gate6E2EvidenceError("Deterministic replay evidence failed")
    if configurations["point_prediction_parity_violation_count"].sum() != 0:
        raise Gate6E2EvidenceError("Point-prediction parity violations were recorded")

    for name, expected_hash in manifest["output_hashes"].items():
        if name not in paths:
            raise Gate6E2EvidenceError(f"Unknown manifest output hash: {name}")
        if _sha256(paths[name]) != expected_hash:
            raise Gate6E2EvidenceError(f"Output hash mismatch for {name}")
    actual_size = _directory_size(output_directory)
    maximum_size = int(contract["execution_budget"]["maximum_artifact_size_mb"]) * 1024 * 1024
    if actual_size > maximum_size or manifest["artifact_size_bytes"] > maximum_size:
        raise Gate6E2EvidenceError("Gate 6E2 artifact-size budget was exceeded")
    if abs(actual_size - int(manifest["artifact_size_bytes"])) > 16:
        raise Gate6E2EvidenceError("Recorded artifact size is inconsistent")
    if recommendation["recommended_next_gate"] != "6E3":
        raise Gate6E2EvidenceError("Gate 6E2 recommendation has an invalid next gate")
    if recommendation["next_gate_authorized"] is not False:
        raise Gate6E2EvidenceError("Gate 6E3 was authorized automatically")

    print(
        "Gate 6E2 evidence: PASS | configurations=9 | origins=7004 | "
        "interval_rows=63036 | next_gate=6E3 | authorized=false"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-absent", action="store_true")
    arguments = parser.parse_args()
    validate_gate_6e2(allow_absent=arguments.allow_absent)


if __name__ == "__main__":
    main()
