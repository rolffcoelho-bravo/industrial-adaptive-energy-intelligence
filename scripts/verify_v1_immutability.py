from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_CONTRACT = ROOT / "configs" / "v2_architecture_contract.yml"
V1_RELEASE_MANIFEST = ROOT / "outputs" / "v1_release_manifest.json"


class ImmutabilityError(RuntimeError):
    """Raised when the frozen V1 boundary is not preserved."""


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ImmutabilityError(f"Expected a mapping in {path}")

    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise ImmutabilityError(
            f"Git command failed: git {' '.join(arguments)}\n{details}"
        )

    return completed.stdout.strip()


def _verify_hash(path_value: str, expected: str) -> None:
    path = ROOT / path_value

    if not path.exists():
        raise ImmutabilityError(f"Frozen V1 artifact is missing: {path_value}")

    actual = _sha256(path)

    if actual != expected:
        raise ImmutabilityError(
            f"Frozen V1 artifact changed: {path_value}\n"
            f"expected={expected}\nactual={actual}"
        )


def verify_v1_immutability() -> None:
    architecture = _load_mapping(ARCHITECTURE_CONTRACT)
    baseline = architecture["v1_baseline"]
    release = _load_mapping(V1_RELEASE_MANIFEST)

    expected_tag = baseline["tag"]
    expected_commit = baseline["commit"]
    observed_commit = _git("rev-list", "-n", "1", expected_tag)

    if observed_commit != expected_commit:
        raise ImmutabilityError(
            f"Unexpected {expected_tag} target: {observed_commit}"
        )

    frozen_paths = baseline["frozen_paths"]
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "--no-ext-diff",
            expected_tag,
            "--",
            *frozen_paths,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        details = completed.stdout.strip() or completed.stderr.strip()
        raise ImmutabilityError(
            "Frozen V1 paths differ from the immutable release tag:\n"
            f"{details}"
        )

    evidence = release["governed_evidence"]

    for key in ("report_payload", "latex_source"):
        item = evidence[key]
        _verify_hash(item["path"], item["sha256"])

    for group in ("figures", "tables"):
        for path_value, expected_hash in evidence[group].items():
            _verify_hash(path_value, expected_hash)

    locked_test = evidence["locked_test"]
    _verify_hash(
        "outputs/modeling/locked_test_predictions.csv",
        locked_test["prediction_sha256"],
    )
    _verify_hash(
        "outputs/modeling/locked_test_results.json",
        locked_test["results_sha256"],
    )

    if release["release_controls"]["v1_mutable"] is not False:
        raise ImmutabilityError("The V1 release manifest does not freeze V1")

    if baseline["locked_test_access_permitted"] is not False:
        raise ImmutabilityError("The V2 contract permits locked-test access")

    print(
        "V1 immutability verification: PASS\n"
        f"tag={expected_tag}\n"
        f"commit={expected_commit}\n"
        f"frozen_paths={len(frozen_paths)}"
    )


if __name__ == "__main__":
    verify_v1_immutability()
