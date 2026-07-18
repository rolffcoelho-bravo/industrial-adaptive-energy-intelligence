from __future__ import annotations

from pathlib import Path

from iaei.reporting.evidence import build_reporting_evidence


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    manifest = build_reporting_evidence(ROOT)
    print(
        (
            "Reporting evidence synthesis: PASS | "
            "gate={} | tables={}"
        ).format(
            manifest["governance_gate"],
            len(manifest["generated_artifacts"]),
        )
    )
