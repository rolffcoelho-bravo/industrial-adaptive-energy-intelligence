from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pandas as pd


NORMALIZED_TEXT_HASH_CONTRACT = "utf8_lf_sha256_v1"
LOGICAL_FRAME_HASH_CONTRACT = "iaei_logical_frame_v1"
LOGICAL_FRAME_FLOAT_DECIMALS = 12


def normalized_text_sha256(path: Path) -> str:
    """Hash UTF-8 text after canonical LF normalization."""
    text = path.read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def logical_frame_sha256(frame: pd.DataFrame) -> str:
    """Return a platform-stable logical fingerprint for a table."""
    digest = hashlib.sha256()
    digest.update(b"IAEI_LOGICAL_FRAME_V1")
    digest.update(struct.pack("<Q", len(frame)))
    digest.update(struct.pack("<Q", len(frame.columns)))

    for column in frame.columns:
        column_name = str(column).encode("utf-8")
        digest.update(struct.pack("<Q", len(column_name)))
        digest.update(column_name)

        series = frame[column]

        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            digest.update(b"DATETIME_NS")
            datetimes = pd.to_datetime(series, errors="raise")
            nulls = datetimes.isna().to_numpy(dtype=np.uint8)
            values = (
                datetimes.fillna(pd.Timestamp(0))
                .astype("int64")
                .to_numpy(dtype="<i8")
            )
            digest.update(nulls.tobytes())
            digest.update(values.tobytes())
            continue

        if pd.api.types.is_bool_dtype(series.dtype):
            digest.update(b"BOOLEAN")
            nulls = series.isna().to_numpy(dtype=np.uint8)
            values = (
                series.fillna(False)
                .astype(bool)
                .to_numpy(dtype=np.uint8)
            )
            digest.update(nulls.tobytes())
            digest.update(values.tobytes())
            continue

        if pd.api.types.is_integer_dtype(series.dtype):
            digest.update(b"INTEGER_64")
            nulls = series.isna().to_numpy(dtype=np.uint8)
            values = (
                series.fillna(0)
                .astype("int64")
                .to_numpy(dtype="<i8")
            )
            digest.update(nulls.tobytes())
            digest.update(values.tobytes())
            continue

        if pd.api.types.is_numeric_dtype(series.dtype):
            digest.update(b"FLOAT_64_ROUNDED_12")
            numeric = pd.to_numeric(series, errors="raise")
            nulls = numeric.isna().to_numpy(dtype=np.uint8)
            values = (
                numeric.fillna(0.0)
                .astype("float64")
                .to_numpy(dtype="<f8")
            )
            values = np.round(
                values,
                decimals=LOGICAL_FRAME_FLOAT_DECIMALS,
            )
            values[values == 0.0] = 0.0
            digest.update(nulls.tobytes())
            digest.update(values.tobytes())
            continue

        digest.update(b"UTF8")

        for value in series.astype("object"):
            if pd.isna(value):
                digest.update(struct.pack("<q", -1))
                continue

            encoded = str(value).encode("utf-8")
            digest.update(struct.pack("<q", len(encoded)))
            digest.update(encoded)

    return digest.hexdigest()
