from __future__ import annotations

from pathlib import Path

from iaei.data import write_silver_artifacts


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = write_silver_artifacts(ROOT)

    output = manifest["output"]
    quality = manifest["quality"]

    print(
        "Decision Gate 3 Silver build: PASS | "
        f"rows={output['row_count']} | "
        f"columns={output['column_count']} | "
        f"dq_any={quality['quality_flag_counts']['dq_any']} | "
        f"parquet_sha256={output['parquet_sha256']}"
    )


if __name__ == "__main__":
    main()
