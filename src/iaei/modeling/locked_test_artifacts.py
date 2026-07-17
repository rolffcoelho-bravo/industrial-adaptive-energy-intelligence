from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


from iaei.modeling.locked_test import LockedTestEvaluation


class LockedTestArtifactError(RuntimeError):
    """Raised when locked-test artifact governance is violated."""


_RESERVATION_MARKER = "reserved_pending_evaluation\n"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LockedTestArtifactError(message)


def _resolve_output_path(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / relative_path).resolve()

    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise LockedTestArtifactError(
            "Locked-test output path escapes the repository root"
        ) from error

    return candidate


def _output_paths(
    root: Path,
    locked_contract: dict[str, Any],
) -> tuple[Path, Path]:
    outputs = locked_contract["outputs"]

    _require(
        outputs["write_once"] is True,
        "Locked-test outputs must be write-once",
    )

    predictions_path = _resolve_output_path(
        root,
        str(outputs["predictions_path"]),
    )
    results_path = _resolve_output_path(
        root,
        str(outputs["results_path"]),
    )

    _require(
        predictions_path != results_path,
        "Prediction and result paths must differ",
    )

    return predictions_path, results_path


def _reserve_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as error:
        raise LockedTestArtifactError(
            f"Locked-test output already exists: {path}"
        ) from error

    with os.fdopen(
        descriptor,
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(_RESERVATION_MARKER)
        handle.flush()
        os.fsync(handle.fileno())


def reserve_locked_test_outputs(
    root: Path,
    locked_contract: dict[str, Any],
) -> tuple[Path, Path]:
    """Reserve both output paths before any test observation is read."""
    predictions_path, results_path = _output_paths(
        root,
        locked_contract,
    )
    created: list[Path] = []

    try:
        _reserve_path(predictions_path)
        created.append(predictions_path)
        _reserve_path(results_path)
        created.append(results_path)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise

    return predictions_path, results_path


def _validate_reservation(path: Path) -> None:
    _require(
        path.is_file(),
        f"Locked-test reservation is missing: {path}",
    )
    _require(
        path.read_text(encoding="utf-8") == _RESERVATION_MARKER,
        f"Locked-test reservation is not pending: {path}",
    )


def _validate_evaluation(
    evaluation: LockedTestEvaluation,
    locked_contract: dict[str, Any],
) -> None:
    required_columns = [
        str(value)
        for value in locked_contract["outputs"][
            "predictions_required_columns"
        ]
    ]

    _require(
        list(evaluation.predictions.columns) == required_columns,
        "Locked-test prediction schema is invalid",
    )
    _require(
        evaluation.results.get("governance_gate") == "4E",
        "Locked-test result has the wrong governance gate",
    )
    _require(
        evaluation.results.get("locked_test_evaluated") is True,
        "Locked-test result is not marked as evaluated",
    )
    _require(
        int(evaluation.results.get("prediction_row_count", -1))
        == len(evaluation.predictions),
        "Locked-test result row count is inconsistent",
    )


def _write_text_file(path: Path, content: str) -> None:
    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def finalize_locked_test_outputs(
    root: Path,
    locked_contract: dict[str, Any],
    evaluation: LockedTestEvaluation,
) -> tuple[Path, Path]:
    """Replace existing reservations with final immutable evidence."""
    predictions_path, results_path = _output_paths(
        root,
        locked_contract,
    )

    _validate_reservation(predictions_path)
    _validate_reservation(results_path)
    _validate_evaluation(evaluation, locked_contract)

    predictions_temporary = predictions_path.with_name(
        f"{predictions_path.name}.finalizing.tmp"
    )
    results_temporary = results_path.with_name(
        f"{results_path.name}.finalizing.tmp"
    )

    _require(
        not predictions_temporary.exists(),
        "Prediction finalization file already exists",
    )
    _require(
        not results_temporary.exists(),
        "Result finalization file already exists",
    )

    predictions_content = evaluation.predictions.to_csv(
        index=False,
        lineterminator="\n",
    )
    results_content = (
        json.dumps(
            evaluation.results,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    try:
        _write_text_file(
            predictions_temporary,
            predictions_content,
        )
        _write_text_file(
            results_temporary,
            results_content,
        )

        os.replace(predictions_temporary, predictions_path)
        os.replace(results_temporary, results_path)
    finally:
        predictions_temporary.unlink(missing_ok=True)
        results_temporary.unlink(missing_ok=True)

    return predictions_path, results_path
