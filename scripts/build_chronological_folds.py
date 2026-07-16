from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from iaei.contracts import load_yaml
from iaei.modeling import build_expanding_window_folds


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "model_contract.yml"
SILVER = ROOT / "data" / "processed" / "steel_energy_silver.parquet"
OUTPUT = ROOT / "outputs" / "modeling" / "chronological_folds.json"


def main() -> None:
    contract = load_yaml(CONTRACT)
    timestamps = pd.read_parquet(
        SILVER,
        columns=["effective_timestamp"],
    )["effective_timestamp"]

    folds = build_expanding_window_folds(timestamps, contract)

    payload = {
        "contract_version": contract["contract_version"],
        "split_type": contract["validation"]["split_type"],
        "row_count": len(timestamps),
        "validation_fold_count": len(folds),
        "purge_steps": contract["validation"]["purge_steps"],
        "locked_test_start": folds[0].test_start,
        "locked_test_stop": folds[0].test_stop,
        "folds": [fold.as_dict() for fold in folds],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "Chronological fold manifest: PASS | "
        f"folds={len(folds)} | "
        f"locked_test=[{folds[0].test_start},{folds[0].test_stop})"
    )


if __name__ == "__main__":
    main()
