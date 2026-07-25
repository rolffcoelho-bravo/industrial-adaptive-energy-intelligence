from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "v2" / "gate_6b"


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(payload: dict[str, object], schema_path: Path, label: str) -> None:
    schema = _load_json(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "\n".join(
            f"- {'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise SystemExit(f"{label} failed validation:\n{details}")


def main() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs" / "advanced_tabular_contract.yml").read_text(
            encoding="utf-8"
        )
    )
    if contract["status"] != "closed":
        raise SystemExit("Gate 6B contract is not closed")

    decision_path = OUTPUT / "promotion_decision.json"
    closure_path = OUTPUT / "gate_6b_closure_manifest.json"
    execution_path = OUTPUT / "gate_6b_execution_manifest.json"
    leaderboard_path = OUTPUT / "candidate_leaderboard.csv"

    decision = _load_json(decision_path)
    closure = _load_json(closure_path)
    execution = _load_json(execution_path)

    _validate(
        decision,
        ROOT / "schemas" / "promotion_decision.schema.json",
        "Gate 6B human promotion decision",
    )
    _validate(
        closure,
        ROOT / "schemas" / "gate_6b_closure_manifest.schema.json",
        "Gate 6B closure manifest",
    )

    if decision["decision"] != "reject":
        raise SystemExit("Gate 6B decision must reject the challenger")
    if decision["next_action"] != "retain_incumbent":
        raise SystemExit("Gate 6B must retain the frozen V1 incumbent")
    if decision["decided_by"]["authority"] != "human":
        raise SystemExit("Gate 6B decision authority is not human")

    if closure["execution"]["execution_manifest_sha256"] != _sha256(
        execution_path
    ):
        raise SystemExit("Gate 6B execution-manifest hash changed")
    if closure["execution"]["leaderboard_sha256"] != _sha256(leaderboard_path):
        raise SystemExit("Gate 6B leaderboard hash changed")
    if closure["human_decision"]["decision_artifact_sha256"] != _sha256(
        decision_path
    ):
        raise SystemExit("Gate 6B promotion-decision hash changed")

    if execution["locked_test_accessed"] is not False:
        raise SystemExit("Gate 6B accessed the locked test")
    if execution["locked_predictions_parsed"] is not False:
        raise SystemExit("Gate 6B parsed locked predictions")
    if execution["confirmatory_evaluation_performed"] is not False:
        raise SystemExit("Gate 6B performed a confirmatory evaluation")
    if closure["v1_boundary"]["immutable"] is not True:
        raise SystemExit("Gate 6B did not preserve immutable V1")

    outcomes = closure["candidate_outcomes"]
    if {item["algorithm_id"] for item in outcomes} != {
        "lightgbm_l1",
        "xgboost_hist",
        "catboost_mae",
    }:
        raise SystemExit("Gate 6B candidate closure set changed")
    if any(item["promotion_eligible"] for item in outcomes):
        raise SystemExit("A rejected Gate 6B challenger remains promotion eligible")

    print(
        "Gate 6B closure validation: PASS | decision=reject_all_challengers | "
        "retained=v1_frozen_champion | next_gate=6C"
    )


if __name__ == "__main__":
    main()
