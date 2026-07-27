from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from iaei.foundation_contracts import (
    validate_foundation_execution_manifest,
    validate_foundation_model_contract,
    validate_foundation_promotion_recommendation,
    validate_foundation_provenance_manifest,
    validate_gate_6d1_closure_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "v2" / "gate_6d"
IMPLEMENTATION_MODULES = (
    ROOT / "src" / "iaei" / "v2" / "foundation_adapters.py",
    ROOT / "src" / "iaei" / "v2" / "foundation_forecasting.py",
    ROOT / "src" / "iaei" / "v2" / "foundation_evidence.py",
)
WORKFLOW = ROOT / ".github" / "workflows" / "gate-6d2-foundation-validation.yml"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected an object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_boundary() -> None:
    prohibited_imports = {"boto3", "google.cloud", "openai", "requests"}
    observed_imports: set[str] = set()
    for path in IMPLEMENTATION_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                observed_imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                observed_imports.add(node.module)
    hits = sorted(
        observed
        for observed in observed_imports
        if any(observed == item or observed.startswith(f"{item}.") for item in prohibited_imports)
    )
    if hits:
        raise SystemExit(f"Gate 6D2 imports prohibited hosted-service clients: {hits}")

    workflow = WORKFLOW.read_text(encoding="utf-8")
    required = (
        "amazon-science/chronos-forecasting.git@7dc4435706a4454feb79df44ca9f33631f3027bf",
        "google-research/timesfm.git@3dae50b20d7a724981e8ea36cda75578f80dd2dc",
        "SalesforceAIResearch/uni2ts.git@cfd46d4510ed8896f263116f32928eede05b0a75",
        "scripts/run_gate_6d2_candidate.py",
        "scripts/build_gate_6d2_evidence.py",
        "scripts/validate_gate_6d2_evidence.py",
    )
    missing = [fragment for fragment in required if fragment not in workflow]
    if missing:
        raise SystemExit(f"Gate 6D2 workflow lacks pinned execution fragments: {missing}")
    prohibited = (
        "trust_remote_code=True",
        "--revision main",
        "fine_tune",
        "lora",
        "locked_test_predictions",
    )
    hits = [fragment for fragment in prohibited if fragment in workflow]
    if hits:
        raise SystemExit(f"Gate 6D2 workflow weakens frozen controls: {hits}")


def _validate_predecessors() -> None:
    validate_foundation_model_contract()
    closure = validate_gate_6d1_closure_manifest()
    if closure["status"] != "closed":
        raise SystemExit("Gate 6D1 is not closed")
    if closure["next_gate"] != "6D2":
        raise SystemExit("Gate 6D1 does not identify Gate 6D2")
    completed = subprocess.run(
        [sys.executable, "scripts/validate_gate_6c3_closure.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "Gate 6C closure validation failed:\n"
            f"{completed.stdout}{completed.stderr}"
        )


def _validate_hashes(manifest: dict[str, Any]) -> None:
    contract = validate_foundation_model_contract()
    output_names = {
        "candidate_results": contract["outputs"]["candidate_results"],
        "outer_fold_results": contract["outputs"]["outer_fold_results"],
        "out_of_fold_predictions": contract["outputs"]["out_of_fold_predictions"],
        "resource_evidence": contract["outputs"]["resource_evidence"],
        "model_provenance_manifest": contract["outputs"]["model_provenance_manifest"],
        "failure_records": contract["outputs"]["failure_records"],
        "promotion_recommendation": contract["outputs"]["promotion_recommendation"],
    }
    if set(manifest["output_hashes"]) != set(output_names):
        raise SystemExit("Gate 6D2 output-hash key set changed")
    for key, name in output_names.items():
        path = OUTPUT / str(name)
        if not path.exists() or _sha256(path) != manifest["output_hashes"][key]:
            raise SystemExit(f"Gate 6D2 output hash mismatch: {key}")

    environment_directory = OUTPUT / "environment_locks"
    observed = {
        path.name: _sha256(path) for path in sorted(environment_directory.glob("*.txt"))
    }
    if observed != manifest["environment_hashes"]:
        raise SystemExit("Gate 6D2 environment hashes do not reconcile")


def _validate_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(OUTPUT / "candidate_results.csv")
    folds = pd.read_csv(OUTPUT / "outer_fold_results.csv")
    predictions = pd.read_parquet(OUTPUT / "out_of_fold_predictions.parquet")
    resources = pd.read_csv(OUTPUT / "resource_evidence.csv")

    expected_ids = {
        "chronos_2_zero_shot",
        "timesfm_2_5_zero_shot",
        "moirai_2_research_zero_shot",
    }
    if set(candidates["candidate_id"]) != expected_ids:
        raise SystemExit("Gate 6D2 candidate identities changed")
    if set(resources["candidate_id"]) != expected_ids:
        raise SystemExit("Gate 6D2 resource identities changed")
    if len(folds) != 12 or len(predictions) != 21012:
        raise SystemExit("Gate 6D2 fold or prediction evidence is incomplete")
    counts = predictions.groupby("candidate_id").size().to_dict()
    if counts != {candidate_id: 7004 for candidate_id in expected_ids}:
        raise SystemExit("Gate 6D2 per-candidate prediction counts changed")

    identity_columns = [
        "fold_id",
        "row_position",
        "actual",
        "is_peak_state",
        "peak_threshold_kwh",
        "maximum_target_dependency",
    ]
    uniqueness = predictions.groupby(["fold_id", "row_position"])[identity_columns].nunique()
    if int(uniqueness.to_numpy().max()) != 1:
        raise SystemExit("Gate 6D2 candidates do not share identical target evidence")

    observed_order = (
        predictions["prediction_q10"].le(predictions["prediction_q50"])
        & predictions["prediction_q50"].le(predictions["prediction_q90"])
    )
    if not observed_order.equals(predictions["quantile_order_passed"].astype(bool)):
        raise SystemExit("Gate 6D2 quantile-order flags do not reconcile")

    for row in folds.itertuples(index=False):
        group = predictions.loc[
            predictions["candidate_id"].eq(row.candidate_id)
            & predictions["fold_id"].eq(row.fold_id)
        ]
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["prediction_q50"].to_numpy(dtype=float)
        peak = group["is_peak_state"].to_numpy(dtype=bool)
        mae = float(np.mean(np.abs(actual - forecast)))
        peak_mae = float(np.mean(np.abs(actual[peak] - forecast[peak])))
        if not math.isclose(mae, float(row.mae), rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit("Gate 6D2 fold MAE does not reconcile")
        if not math.isclose(
            peak_mae, float(row.peak_mae), rel_tol=0.0, abs_tol=1e-12
        ):
            raise SystemExit("Gate 6D2 fold peak MAE does not reconcile")

    for row in candidates.itertuples(index=False):
        group = folds.loc[folds["candidate_id"].eq(row.candidate_id)]
        prediction_group = predictions.loc[
            predictions["candidate_id"].eq(row.candidate_id)
        ]
        if not math.isclose(
            float(group["mae"].mean()),
            float(row.mean_mae),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemExit("Gate 6D2 aggregate MAE does not reconcile")
        if not math.isclose(
            float(group["peak_mae"].mean()),
            float(row.mean_peak_mae),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemExit("Gate 6D2 aggregate peak MAE does not reconcile")
        crossing_rate = float(
            (~prediction_group["quantile_order_passed"].astype(bool)).mean()
        )
        if not math.isclose(
            crossing_rate,
            float(row.quantile_crossing_rate),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemExit("Gate 6D2 quantile-crossing rate does not reconcile")
        if bool(row.quantile_order_passed) != bool(
            prediction_group["quantile_order_passed"].all()
        ):
            raise SystemExit("Gate 6D2 aggregate quantile-order flag does not reconcile")
    return candidates, folds, resources


def _validate_provenance_and_decision(candidates: pd.DataFrame) -> None:
    provenance = validate_foundation_provenance_manifest()
    recommendation = validate_foundation_promotion_recommendation()
    expected_hashes = {
        "chronos_2_zero_shot": "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42",
        "timesfm_2_5_zero_shot": "2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8",
        "moirai_2_research_zero_shot": "fb5652a3db8ea572606221b7cb1e77bb8962b168e4d4cc752cf31ceb04074669",
    }
    observed = {item["candidate_id"]: item for item in provenance["candidates"]}
    if set(observed) != set(expected_hashes):
        raise SystemExit("Gate 6D2 provenance candidate set changed")
    for candidate_id, expected_hash in expected_hashes.items():
        if observed[candidate_id]["weight_sha256"] != expected_hash:
            raise SystemExit("Gate 6D2 weight identity changed")

    moirai = candidates.loc[
        candidates["candidate_id"].eq("moirai_2_research_zero_shot")
    ].iloc[0]
    if bool(moirai["commercial_use_eligible"]) or bool(
        moirai["promotion_eligible_by_license"]
    ):
        raise SystemExit("Gate 6D2 weakened Moirai's non-commercial boundary")
    decision = {
        item["candidate_id"]: item for item in recommendation["candidates"]
    }
    if decision["moirai_2_research_zero_shot"]["promotion_eligible"]:
        raise SystemExit("Gate 6D2 made research-only Moirai promotion eligible")
    if "commercial_use_eligibility" not in decision[
        "moirai_2_research_zero_shot"
    ]["failed_requirements"]:
        raise SystemExit("Gate 6D2 recommendation omits Moirai's license boundary")

    failures = _load_json(OUTPUT / "failure_records.json")
    if failures["records"]:
        raise SystemExit("Gate 6D2 contains unexpected failure records")


def main() -> None:
    _validate_predecessors()
    _validate_source_boundary()
    manifest_path = OUTPUT / "gate_6d_execution_manifest.json"
    if not manifest_path.exists():
        if OUTPUT.exists() and any(OUTPUT.iterdir()):
            raise SystemExit("Gate 6D2 has partial execution artifacts")
        print(
            "Gate 6D2 implementation validation: PASS | candidates=3 | "
            "pinned_sources=true | execution_evidence=false"
        )
        return

    manifest = validate_foundation_execution_manifest()
    _validate_hashes(manifest)
    candidates, _, resources = _validate_metrics()
    _validate_provenance_and_decision(candidates)
    if float(resources["peak_memory_mb"].max()) > 6144.0:
        raise SystemExit("Gate 6D2 peak memory exceeded its frozen limit")
    if float(resources["wall_clock_seconds"].sum()) > 21600.0:
        raise SystemExit("Gate 6D2 total runtime exceeded its frozen limit")
    print(
        "Gate 6D2 evidence validation: PASS | candidates=3 | predictions=21012 | "
        "locked_test=false | fine_tuning=false | next_gate=6D3"
    )


if __name__ == "__main__":
    main()
