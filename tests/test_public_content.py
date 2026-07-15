from pathlib import Path

from iaei.content_audit import audit_repository


def test_repository_public_content_is_audience_facing() -> None:
    root = Path(__file__).resolve().parents[1]
    assert audit_repository(root) == []
