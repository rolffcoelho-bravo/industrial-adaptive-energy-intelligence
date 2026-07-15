from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "data" / "manifests"
URL = "https://archive.ics.uci.edu/static/public/851/steel%2Bindustry%2Benergy%2Bconsumption.zip"
ARCHIVE = RAW / "uci_851_steel_energy.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=90) as response, ARCHIVE.open("wb") as target:
        shutil.copyfileobj(response, target)

    with zipfile.ZipFile(ARCHIVE) as archive:
        archive.extractall(RAW)

    csv_files = sorted(RAW.rglob("*.csv"))
    if len(csv_files) != 1:
        raise RuntimeError(f"Expected one CSV in UCI archive; found {csv_files}")

    manifest = {
        "dataset_id": "uci-851",
        "source_url": URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": sha256(ARCHIVE),
        "csv_path": str(csv_files[0].relative_to(ROOT)),
        "csv_sha256": sha256(csv_files[0]),
        "license": "CC BY 4.0",
        "doi": "10.24432/C52G8C",
    }
    path = MANIFESTS / "uci_851_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded real UCI data and wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
