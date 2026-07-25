from __future__ import annotations

import hashlib

from iaei.reporting import build_report_payload


if __name__ == "__main__":
    output = build_report_payload()
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        "Report payload: PASS | "
        f"path={output} | "
        f"sha256={digest}"
    )
