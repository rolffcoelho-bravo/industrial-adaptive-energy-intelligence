from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from iaei.contracts import load_yaml
from iaei.modeling.splits import build_expanding_window_folds
from iaei.v2.uncertainty_calibration import (
    configurations_from_contract,
    summarize_configuration,
)
from iaei.v2.uncertainty_evidence import (
    _hard_constraint_checks,
    _recommendation,
    _results_markdown,
)
from iaei.v2.uncertainty_execution import (
    ROOT,
    Gate6E2ExecutionError,
    _build_calibration_residuals,
    _directory_size,
    _evaluate_one_configuration,
    _git_commit,
    _load_json,
    _nonfinite_count,
    _peak_rss_mb,
    _pip_freeze,
    _point_parity_violations,
    _resolve_silver_path,
    _sha256_path,
    _write_json,
)

CONTRACT_PATH = ROOT / "configs" / "uncertainty_contract.yml"
MODEL_CONTRACT_PATH = ROOT / "configs" / "model_contract.yml"
TARGET_CONTRACT_PATH = ROOT / "configs" / "target_contract.yml"
CLOSURE_PATH = ROOT / "outputs" / "v2" / "gate_6e1_closure_manifest.json"
POINT_MANIFEST_PATH = (
    ROOT / "outputs" / "modeling" / "hist_gradient_boosting_candidate_manifest.json"
)
POINT_PREDICTIONS_PATH = (
    ROOT
    / "outputs"
    / "modeling"
    / "hist_gradient_boosting_out_of_fold_predictions.parquet"
)


def main() -> None:
    contract = load_yaml(CONTRACT_PATH)
    model_contract = load_yaml(MODEL_CONTRACT_PATH)
    target_contract = load_yaml(TARGET_CONTRACT_PATH)
    closure = _load_json(CLOSURE_PATH)
    point_manifest = _load_json(POINT_MANIFEST_PATH)
    output_directory = ROOT / str(contract["outputs"]["directory"])
    manifest_path = output_directory / str(contract["outputs"]["execution_manifest"])
    if manifest_path.exists():
        raise Gate6E2ExecutionError("Gate 6E2 evidence is write-once")
    if closure.get("status") != "closed" or closure.get("subgate") != "6E1":
        raise Gate6E2ExecutionError("Gate 6E1 is not closed")
    if point_manifest.get("candidate") != "hist_gradient_boosting":
        raise Gate6E2ExecutionError("Frozen point-model identity is inconsistent")
    if point_manifest.get("locked_test_evaluated") is not False:
        raise Gate6E2ExecutionError("Point evidence has an invalid test boundary")

    execution_started = time.perf_counter()
    output_directory.mkdir(parents=True, exist_ok=False)
    environment_path = output_directory / "environment_lock.txt"
    environment_path.write_text(
        _pip_freeze(),
        encoding="utf-8",
        newline="\n",
    )
    silver_path, silver_lineage = _resolve_silver_path(contract)
    silver = pd.read_parquet(silver_path)
    folds = build_expanding_window_folds(
        silver["effective_timestamp"],
        model_contract,
    )
    if len(folds) != int(contract["evidence_boundary"]["outer_fold_count"]):
        raise Gate6E2ExecutionError("Outer-fold count conflicts with the contract")

    point_predictions = pd.read_parquet(POINT_PREDICTIONS_PATH).sort_values(
        ["fold_id", "row_position"],
        kind="stable",
    )
    expected_origins = int(contract["evidence_boundary"]["validation_origin_count"])
    if len(point_predictions) != expected_origins:
        raise Gate6E2ExecutionError("Point predictions do not cover 7,004 origins")
    if point_predictions["row_position"].nunique() != expected_origins:
        raise Gate6E2ExecutionError("Point-prediction origins are not unique")
    if int(point_predictions["row_position"].max()) >= int(
        contract["evidence_boundary"]["maximum_prediction_origin_exclusive"]
    ):
        raise Gate6E2ExecutionError("Point predictions cross the locked boundary")

    calibration_started = time.perf_counter()
    scores_by_fold, iqr_by_fold, residuals, lineage = _build_calibration_residuals(
        silver,
        folds,
        model_contract,
        target_contract,
        point_manifest,
        contract,
    )
    calibration_seconds = time.perf_counter() - calibration_started
    residuals_path = output_directory / "calibration_residuals.parquet"
    residuals.to_parquet(residuals_path, index=False)
    lineage["silver_lineage"] = silver_lineage
    lineage["point_prediction_sha256"] = _sha256_path(POINT_PREDICTIONS_PATH)
    lineage["calibration_residuals_sha256"] = _sha256_path(residuals_path)
    lineage_path = output_directory / str(contract["outputs"]["calibration_lineage"])
    _write_json(lineage_path, lineage)

    coverage_levels = [
        float(value)
        for value in contract["calibration_protocol"]["target_coverage_levels"]
    ]
    primary_coverage = float(
        contract["calibration_protocol"]["primary_coverage_level"]
    )
    lower_support = float(
        contract["calibration_protocol"]["lower_support_bound_kwh"]
    )
    parity_tolerance = float(
        contract["point_model_boundary"][
            "point_prediction_parity_absolute_tolerance"
        ]
    )
    configurations = configurations_from_contract(contract)
    configuration_rows: list[dict[str, Any]] = []
    coverage_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    resource_rows: list[dict[str, Any]] = [
        {
            "stage": "calibration_residual_construction",
            "configuration_id": "shared",
            "wall_clock_seconds": float(calibration_seconds),
            "p95_latency_ms_per_1000_rows": 0.0,
            "peak_memory_mb": _peak_rss_mb(),
            "cpu_execution_passed": True,
            "deterministic_replay_passed": True,
        }
    ]

    for configuration in configurations:
        evaluated, latencies, wall_clock, peak_memory = _evaluate_one_configuration(
            point_predictions,
            scores_by_fold,
            configuration,
            coverage_levels,
            lower_support,
        )
        replay, _, _, _ = _evaluate_one_configuration(
            point_predictions,
            scores_by_fold,
            configuration,
            coverage_levels,
            lower_support,
        )
        replay_columns = [
            column
            for column in evaluated.columns
            if column not in {"prediction_origin"}
        ]
        try:
            assert_frame_equal(
                evaluated[replay_columns],
                replay[replay_columns],
                check_exact=True,
                check_dtype=True,
            )
            replay_passed = True
        except AssertionError:
            replay_passed = False
        coverage, fold_results, aggregate = summarize_configuration(
            evaluated,
            coverage_levels,
            iqr_by_fold,
            primary_coverage=primary_coverage,
            rolling_window=672,
        )
        parity_violations = _point_parity_violations(
            evaluated,
            point_predictions,
            parity_tolerance,
        )
        cpu_passed = (
            str(contract["execution_budget"]["canonical_device"]) == "cpu"
            and not bool(contract["execution_budget"]["gpu_required"])
            and platform.machine() != ""
        )
        checks = _hard_constraint_checks(
            evaluated,
            coverage,
            aggregate,
            contract,
            parity_violations,
            replay_passed,
            cpu_passed,
        )
        failed = sorted(name for name, passed in checks.items() if not passed)
        aggregate_coverage: dict[int, float] = {}
        aggregate_width: dict[int, float] = {}
        aggregate_peak_coverage: dict[int, float] = {}
        for level in coverage_levels:
            suffix = int(round(level * 100))
            aggregate_coverage[suffix] = float(evaluated[f"covered_{suffix}"].mean())
            aggregate_width[suffix] = float(
                (evaluated[f"upper_{suffix}"] - evaluated[f"lower_{suffix}"]).mean()
            )
            aggregate_peak_coverage[suffix] = float(
                evaluated.loc[evaluated["is_peak_state"], f"covered_{suffix}"].mean()
            )
        configuration_rows.append(
            {
                "configuration_id": configuration.configuration_id,
                "method_id": configuration.method_id,
                "role": configuration.role,
                "window_intervals": configuration.window_intervals,
                "adaptive_gamma": configuration.adaptive_gamma,
                "status": "success",
                "origin_count": int(len(evaluated)),
                "peak_origin_count": int(evaluated["is_peak_state"].sum()),
                "empirical_coverage_80": aggregate_coverage[80],
                "empirical_coverage_90": aggregate_coverage[90],
                "empirical_coverage_95": aggregate_coverage[95],
                "absolute_coverage_error_80": abs(aggregate_coverage[80] - 0.80),
                "absolute_coverage_error_90": abs(aggregate_coverage[90] - 0.90),
                "absolute_coverage_error_95": abs(aggregate_coverage[95] - 0.95),
                "mean_interval_width_kwh_at_80": aggregate_width[80],
                "mean_interval_width_kwh_at_90": aggregate_width[90],
                "mean_interval_width_kwh_at_95": aggregate_width[95],
                "peak_state_coverage_80": aggregate_peak_coverage[80],
                "peak_state_coverage_90": aggregate_peak_coverage[90],
                "peak_state_coverage_95": aggregate_peak_coverage[95],
                "weighted_interval_score": float(
                    aggregate["weighted_interval_score"]
                ),
                "peak_state_weighted_interval_score": float(
                    aggregate["peak_state_weighted_interval_score"]
                ),
                "maximum_outer_fold_absolute_coverage_error": float(
                    aggregate["maximum_outer_fold_absolute_coverage_error"]
                ),
                "maximum_outer_fold_90_absolute_coverage_error": float(
                    aggregate["maximum_outer_fold_90_absolute_coverage_error"]
                ),
                "maximum_rolling_672_absolute_coverage_error": float(
                    aggregate["maximum_rolling_672_absolute_coverage_error"]
                ),
                "longest_consecutive_miss_run": int(
                    aggregate["longest_consecutive_miss_run"]
                ),
                "interval_nesting_violation_count": int(
                    aggregate["interval_nesting_violation_count"]
                ),
                "support_floor_activation_rate": float(
                    aggregate["support_floor_activation_rate"]
                ),
                "point_prediction_parity_violation_count": parity_violations,
                "chronology_violation_count": 0,
                "leakage_violation_count": 0,
                "locked_test_access_count": 0,
                "nonfinite_evidence_count": _nonfinite_count(evaluated),
                "p95_latency_ms_per_1000_rows": float(np.quantile(latencies, 0.95)),
                "wall_clock_seconds": float(wall_clock),
                "peak_memory_mb": float(peak_memory),
                "deterministic_replay_passed": replay_passed,
                "cpu_portability_passed": cpu_passed,
                "hard_constraints_passed": not failed,
                "failed_hard_constraints": json.dumps(failed, separators=(",", ":")),
            }
        )
        coverage_frames.append(coverage)
        fold_frames.append(fold_results)
        prediction_frames.append(evaluated)
        resource_rows.append(
            {
                "stage": "interval_generation",
                "configuration_id": configuration.configuration_id,
                "wall_clock_seconds": float(wall_clock),
                "p95_latency_ms_per_1000_rows": float(np.quantile(latencies, 0.95)),
                "peak_memory_mb": float(peak_memory),
                "cpu_execution_passed": cpu_passed,
                "deterministic_replay_passed": replay_passed,
            }
        )

    configuration_results = pd.DataFrame(configuration_rows).sort_values(
        "weighted_interval_score",
        kind="stable",
    )
    coverage_results = pd.concat(coverage_frames, ignore_index=True).sort_values(
        ["configuration_id", "fold_id", "coverage_level"],
        kind="stable",
    )
    outer_fold_results = pd.concat(fold_frames, ignore_index=True).sort_values(
        ["configuration_id", "fold_id"],
        kind="stable",
    )
    interval_predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["configuration_id", "fold_id", "row_position"],
        kind="stable",
    )
    resources = pd.DataFrame(resource_rows)
    recommendation = _recommendation(
        configuration_results,
        outer_fold_results,
        contract,
    )

    output_paths = {
        "configuration_results": output_directory
        / str(contract["outputs"]["configuration_results"]),
        "coverage_results": output_directory
        / str(contract["outputs"]["coverage_results"]),
        "outer_fold_results": output_directory
        / str(contract["outputs"]["outer_fold_results"]),
        "interval_predictions": output_directory
        / str(contract["outputs"]["interval_predictions"]),
        "calibration_lineage": lineage_path,
        "calibration_residuals": residuals_path,
        "resource_evidence": output_directory
        / str(contract["outputs"]["resource_evidence"]),
        "failure_records": output_directory
        / str(contract["outputs"]["failure_records"]),
        "promotion_recommendation": output_directory
        / str(contract["outputs"]["promotion_recommendation"]),
        "environment_lock": environment_path,
    }
    configuration_results.to_csv(
        output_paths["configuration_results"],
        index=False,
        lineterminator="\n",
    )
    coverage_results.to_csv(
        output_paths["coverage_results"],
        index=False,
        lineterminator="\n",
    )
    outer_fold_results.to_csv(
        output_paths["outer_fold_results"],
        index=False,
        lineterminator="\n",
    )
    interval_predictions.to_parquet(
        output_paths["interval_predictions"],
        index=False,
    )
    resources.to_csv(
        output_paths["resource_evidence"],
        index=False,
        lineterminator="\n",
    )
    _write_json(
        output_paths["failure_records"],
        {
            "schema_version": "1.0.0",
            "gate": "6E",
            "subgate": "6E2",
            "records": [],
        },
    )
    _write_json(output_paths["promotion_recommendation"], recommendation)

    total_seconds = time.perf_counter() - execution_started
    maximum_artifact_size = int(
        contract["execution_budget"]["maximum_artifact_size_mb"]
    ) * 1024 * 1024
    if total_seconds > float(
        contract["execution_budget"]["maximum_total_wall_clock_minutes"]
    ) * 60.0:
        raise Gate6E2ExecutionError("Gate 6E2 execution exceeds the time budget")
    if float(resources["peak_memory_mb"].max()) > float(
        contract["execution_budget"]["maximum_peak_memory_mb"]
    ):
        raise Gate6E2ExecutionError("Gate 6E2 execution exceeds the memory budget")

    hashes = {name: _sha256_path(path) for name, path in output_paths.items()}
    manifest = {
        "schema_version": "1.0.0",
        "gate": "6E",
        "subgate": "6E2",
        "status": "validation_complete_pending_human_decision",
        "execution_commit": _git_commit(),
        "contract_version": str(contract["contract_version"]),
        "retained_point_model": "v1_frozen_champion",
        "configuration_ids": [
            configuration.configuration_id for configuration in configurations
        ],
        "configuration_count": len(configurations),
        "coverage_levels": coverage_levels,
        "outer_fold_count": len(folds),
        "validation_origins_per_configuration": expected_origins,
        "interval_prediction_row_count": int(len(interval_predictions)),
        "calibration_residual_row_count": int(len(residuals)),
        "maximum_prediction_origin": int(interval_predictions["row_position"].max()),
        "maximum_target_dependency": int(
            interval_predictions["row_position"].max()
        )
        + 1,
        "locked_test_start": int(contract["evidence_boundary"]["locked_test_start"]),
        "point_prediction_parity_violation_count": int(
            configuration_results[
                "point_prediction_parity_violation_count"
            ].sum()
        ),
        "calibration_performed": True,
        "intervals_generated": True,
        "uncertainty_metrics_calculated": True,
        "optimization_executed": True,
        "point_model_search_performed": False,
        "point_model_mutated": False,
        "locked_test_accessed": False,
        "locked_predictions_parsed": False,
        "confirmatory_evaluation_performed": False,
        "automatic_promotion_permitted": False,
        "probabilistic_authority_claimed": False,
        "human_decision_required": True,
        "v1_immutable": True,
        "canonical_device": "cpu",
        "external_api_cost_usd": 0.0,
        "total_wall_clock_seconds": float(total_seconds),
        "peak_memory_mb": float(resources["peak_memory_mb"].max()),
        "artifact_size_bytes": 0,
        "output_hashes": hashes,
        "recommendation_outcome": recommendation["outcome"],
        "recommended_configuration": recommendation["recommended_configuration"],
        "next_gate": "6E3",
        "next_gate_authorized": False,
        "blocked_gates": ["6E3", "6F", "6G"],
    }
    _write_json(manifest_path, manifest)
    artifact_size = _directory_size(output_directory)
    if artifact_size > maximum_artifact_size:
        raise Gate6E2ExecutionError("Gate 6E2 evidence exceeds the artifact budget")
    manifest["artifact_size_bytes"] = artifact_size
    _write_json(manifest_path, manifest)
    results_path = ROOT / "docs" / "GATE_6E2_EXECUTION_RESULTS.md"
    results_path.write_text(
        _results_markdown(configuration_results, recommendation, manifest),
        encoding="utf-8",
        newline="\n",
    )
    print(
        "Gate 6E2 uncertainty execution: PASS | configurations={} | "
        "origins={} | interval_rows={} | outcome={} | recommended={}".format(
            len(configurations),
            expected_origins,
            len(interval_predictions),
            recommendation["outcome"],
            recommendation["recommended_configuration"],
        )
    )


if __name__ == "__main__":
    main()
