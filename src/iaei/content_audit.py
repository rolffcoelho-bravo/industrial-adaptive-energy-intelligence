from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {
    ".cff",
    ".cmd",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".sql",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}

EXCLUDED_PREFIXES = (
    "data/raw/",
    "outputs/",
)

EXCLUDED_FILES = {
    "src/iaei/content_audit.py",
}

FORBIDDEN_FRAGMENTS = (
    "public-release rule",
    "public release sequence",
    "do not invite reviewers",
    "make the repository public",
    "switch to public before outreach",
    "personal application documents",
    "application correspondence",
    "target organization",
    "head of research",
    "approved by the head of research",
    "must contain exactly five pages",
    "exactly five pages",
    "exact page count",
    "before outreach",
    "perceived value",
)

EM_DASH = chr(0x2014)


@dataclass(frozen=True)
class ContentIssue:
    path: str
    line_number: int
    rule: str
    line: str


def _is_excluded(path: Path, root: Path) -> bool:
    relative_path = path.relative_to(root)
    relative = relative_path.as_posix()

    if relative in EXCLUDED_FILES:
        return True

    if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative_path.parts):
        return True

    return relative.startswith(EXCLUDED_PREFIXES)


def audit_repository(root: Path) -> list[ContentIssue]:
    issues: list[ContentIssue] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if _is_excluded(path, root):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        relative = path.relative_to(root).as_posix()

        for line_number, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()

            if EM_DASH in line:
                issues.append(
                    ContentIssue(
                        path=relative,
                        line_number=line_number,
                        rule="em_dash",
                        line=line.strip(),
                    )
                )

            for fragment in FORBIDDEN_FRAGMENTS:
                if fragment in lowered:
                    issues.append(
                        ContentIssue(
                            path=relative,
                            line_number=line_number,
                            rule=f"forbidden_phrase:{fragment}",
                            line=line.strip(),
                        )
                    )

    return issues
