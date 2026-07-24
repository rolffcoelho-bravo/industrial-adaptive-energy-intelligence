from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from iaei.visualization.style import (
    PALETTE,
    add_figure_header,
    publication_style,
    save_publication_figure,
    style_axis,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render(path: Path) -> Path:
    x = np.linspace(0.0, 24.0, 2_000)
    primary = 42.0 + 8.0 * np.sin(x)
    reference = 46.0 + 6.0 * np.cos(x / 1.4)

    with publication_style():
        fig, ax = plt.subplots(figsize=(11.2, 6.4))
        add_figure_header(
            fig,
            "Deterministic visualization test",
            "Controlled in-memory fixture for rendering validation only.",
        )
        ax.plot(
            x,
            primary,
            color=PALETTE["teal"],
            label="Candidate",
        )
        ax.plot(
            x,
            reference,
            color=PALETTE["slate"],
            label="Reference",
        )
        ax.fill_between(
            x,
            primary,
            reference,
            color=PALETTE["blue"],
            alpha=0.12,
        )
        ax.set_xlabel("Controlled index")
        ax.set_ylabel("Controlled value")
        ax.legend(loc="upper right")
        style_axis(ax)
        fig.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.84,
            bottom=0.12,
        )

        return save_publication_figure(
            fig,
            path,
            figure_id="Style test",
            source="Controlled in-memory fixture",
            sample="Not a public research result",
            evidence_id="style-contract-v1",
        )


def test_style_forces_headless_backend() -> None:
    assert matplotlib.get_backend().lower() == "agg"


def test_style_render_is_byte_deterministic(tmp_path: Path) -> None:
    first = _render(tmp_path / "first.png")
    second = _render(tmp_path / "second.png")

    assert _sha256(first) == _sha256(second)
    assert first.read_bytes() == second.read_bytes()


def test_style_render_passes_publication_dimensions(
    tmp_path: Path,
) -> None:
    output = _render(tmp_path / "dimensions.png")

    assert output.stat().st_size >= 30_000

    with Image.open(output) as image:
        width, height = image.size
        assert width >= 2_400
        assert height >= 1_200
        assert image.mode in {"RGB", "RGBA"}


def test_style_contains_no_clock_timestamp() -> None:
    source = (
        Path("src/iaei/visualization/style.py")
        .read_text(encoding="utf-8")
    )

    assert "datetime.now" not in source
    assert "utcnow" not in source
    assert "Generated:" not in source


def test_style_uses_deterministic_png_metadata() -> None:
    source = (
        Path("src/iaei/visualization/style.py")
        .read_text(encoding="utf-8")
    )

    assert "metadata={" in source
    assert "compress_level" in source
    assert "optimize" in source
