from __future__ import annotations

from pathlib import Path

from iaei.content_audit import audit_repository

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    issues = audit_repository(ROOT)
    if issues:
        for issue in issues:
            print(
                f"{issue.path}:{issue.line_number}: {issue.rule}: {issue.line}"
            )
        raise SystemExit(
            f"Public-content audit failed with {len(issues)} issue(s)."
        )

    print("Public-content audit: PASS")


if __name__ == "__main__":
    main()
