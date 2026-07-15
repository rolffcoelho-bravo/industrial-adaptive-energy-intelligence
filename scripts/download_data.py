from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from iaei.data.intake import (
    ARCHIVE_URL,
    DATASET_CITATION,
    DATASET_DOI,
    DATASET_ID,
    DATASET_LICENSE,
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    build_snapshot_manifest,
    inspect_csv,
    validate_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "raw" / "uci_steel_energy"
CSV_PATH = SNAPSHOT_DIR / "Steel_industry_data.csv"
MANIFEST_PATH = ROOT / "data" / "manifests" / "uci_steel_energy_manifest.json"
LEGACY_MANIFEST_PATH = ROOT / "data" / "manifests" / "uci_851_manifest.json"
SOURCE_METADATA_PATH = SNAPSHOT_DIR / "source_metadata.json"
CHECKSUM_PATH = SNAPSHOT_DIR / "SHA256SUMS"
README_PATH = SNAPSHOT_DIR / "README.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive(destination: Path) -> None:
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"User-Agent": "industrial-adaptive-energy-intelligence/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, destination.open(
        "wb"
    ) as target:
        shutil.copyfileobj(response, target)


def extract_single_csv(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise RuntimeError(f"Expected exactly one CSV in UCI archive; found {csv_members}")
        with archive.open(csv_members[0]) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def write_supporting_files(
    manifest: dict[str, object],
    source_metadata_path: Path,
    checksum_path: Path,
    readme_path: Path,
    csv_path: Path,
) -> None:
    source_metadata = {
        "dataset_id": DATASET_ID,
        "dataset_name": "Steel Industry Energy Consumption",
        "official_dataset_page": "https://archive.ics.uci.edu/dataset/851/steel%2Bindustry%2Benergy%2Bconsumption",
        "official_archive_url": ARCHIVE_URL,
        "doi": DATASET_DOI,
        "license": DATASET_LICENSE,
        "citation": DATASET_CITATION,
        "source_status": "official",
        "snapshot_policy": "immutable_raw_snapshot",
        "retrieved_at_utc": manifest["downloaded_at_utc"],
        "timestamp_convention": manifest["timestamp_convention"],
    }
    source_metadata_path.write_text(
        json.dumps(source_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    checksum_path.write_text(
        f'{manifest["csv_sha256"]}  {csv_path.name}\n',
        encoding="utf-8",
    )
    readme_path.write_text(
        "\n".join(
            [
                "# UCI Steel Industry Energy Consumption snapshot",
                "",
                "This directory contains an immutable byte-for-byte CSV snapshot downloaded from the official UCI Machine Learning Repository.",
                "",
                f"- Dataset ID: `{DATASET_ID}`",
                f"- DOI: `{DATASET_DOI}`",
                f"- License: `{DATASET_LICENSE}`",
                f"- Expected observations: `{EXPECTED_ROWS}`",
                f"- Observed columns: `{len(EXPECTED_COLUMNS)}`",
                f"- SHA-256: `{manifest['csv_sha256']}`",
                "",
                "## Timestamp convention",
                "",
                str(manifest["timestamp_convention"]),
                "",
                "The raw CSV is never sorted or rewritten. Chronological timestamps are constructed only in memory and in downstream governed layers.",
                "",
                "## Scope",
                "",
                "The snapshot contains no proprietary company data.",
                "Cleaning, feature engineering, and imputation are prohibited in this raw directory.",
                "",
                "## Attribution",
                "",
                DATASET_CITATION,
                "",
            ]
        ),
        encoding="utf-8",
    )


def atomic_publish(staged_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    incoming_path = destination_path.with_name(destination_path.name + ".incoming")
    shutil.copy2(staged_path, incoming_path)
    os.replace(incoming_path, destination_path)


def validate_supporting_files() -> None:
    required = [SOURCE_METADATA_PATH, CHECKSUM_PATH, README_PATH]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Snapshot supporting files are missing: {missing}")

    expected_checksum = f"{sha256(CSV_PATH)}  {CSV_PATH.name}"
    observed_checksum = CHECKSUM_PATH.read_text(encoding="utf-8").strip()
    if observed_checksum != expected_checksum:
        raise RuntimeError(
            "SHA256SUMS does not match the committed CSV: "
            f"expected {expected_checksum}; observed {observed_checksum}"
        )


def refresh_snapshot() -> dict[str, object]:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="iaei-uci-851-") as temporary:
        staging_root = Path(temporary) / "publication"
        staged_snapshot_dir = staging_root / "data" / "raw" / "uci_steel_energy"
        staged_manifest_dir = staging_root / "data" / "manifests"
        staged_snapshot_dir.mkdir(parents=True, exist_ok=True)
        staged_manifest_dir.mkdir(parents=True, exist_ok=True)

        archive_path = Path(temporary) / "uci_851.zip"
        staged_csv = staged_snapshot_dir / CSV_PATH.name
        staged_manifest = staged_manifest_dir / MANIFEST_PATH.name
        staged_source_metadata = staged_snapshot_dir / SOURCE_METADATA_PATH.name
        staged_checksum = staged_snapshot_dir / CHECKSUM_PATH.name
        staged_readme = staged_snapshot_dir / README_PATH.name

        download_archive(archive_path)
        extract_single_csv(archive_path, staged_csv)

        # Validate the source bytes before publishing anything into the repository.
        inspect_csv(staged_csv)
        manifest = build_snapshot_manifest(
            csv_path=staged_csv,
            root=staging_root,
            downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
            archive_sha256=sha256(archive_path),
        )
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_supporting_files(
            manifest=manifest,
            source_metadata_path=staged_source_metadata,
            checksum_path=staged_checksum,
            readme_path=staged_readme,
            csv_path=staged_csv,
        )
        validate_snapshot(staged_csv, staged_manifest, staging_root)

        # Publish evidence files first and the manifest last.
        atomic_publish(staged_csv, CSV_PATH)
        atomic_publish(staged_source_metadata, SOURCE_METADATA_PATH)
        atomic_publish(staged_checksum, CHECKSUM_PATH)
        atomic_publish(staged_readme, README_PATH)
        atomic_publish(staged_manifest, MANIFEST_PATH)

    if LEGACY_MANIFEST_PATH.exists():
        LEGACY_MANIFEST_PATH.unlink()

    validate_snapshot(CSV_PATH, MANIFEST_PATH, ROOT)
    validate_supporting_files()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire or verify the governed UCI-851 raw snapshot"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--refresh",
        action="store_true",
        help="Download the official archive and rebuild the snapshot",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Verify the committed snapshot without network access",
    )
    args = parser.parse_args()

    if args.verify:
        result = validate_snapshot(CSV_PATH, MANIFEST_PATH, ROOT)
        validate_supporting_files()
        print(
            "UCI-851 snapshot verification: PASS | "
            f"rows={result['row_count']} | sha256={result['csv_sha256']} | "
            f"effective_sample={result['sample_start']}..{result['sample_end']}"
        )
        return

    manifest = refresh_snapshot()
    print(
        "UCI-851 official snapshot created and verified | "
        f"rows={manifest['row_count']} | sha256={manifest['csv_sha256']} | "
        f"effective_sample={manifest['sample_start']}..{manifest['sample_end']}"
    )


if __name__ == "__main__":
    main()
