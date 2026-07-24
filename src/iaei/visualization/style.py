from __future__ import annotations

import gc
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image as PILImage


PALETTE = {
    "ink": "#17212B",
    "muted": "#5B6670",
    "grid": "#D7DEE5",
    "navy": "#234E70",
    "blue": "#3B82A0",
    "teal": "#2A9D8F",
    "amber": "#E9A23B",
    "vermillion": "#D95D39",
    "red": "#B33A3A",
    "slate": "#7A8793",
    "light": "#F4F7F9",
    "white": "#FFFFFF",
}


@contextmanager
def publication_style() -> Iterator[None]:
    """Apply the repository-controlled institutional figure style."""

    settings = {
        "figure.facecolor": PALETTE["white"],
        "axes.facecolor": PALETTE["white"],
        "savefig.facecolor": PALETTE["white"],
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["grid"],
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "text.color": PALETTE["ink"],
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.16,
    }

    with mpl.rc_context(settings):
        yield


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)


def add_figure_header(
    fig: Figure,
    title: str,
    subtitle: str,
) -> None:
    fig.suptitle(
        title,
        x=0.055,
        y=0.975,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.925,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=PALETTE["muted"],
    )


def save_publication_figure(
    fig: Figure,
    output_path: Path,
    *,
    figure_id: str,
    source: str,
    sample: str,
    evidence_id: str | None = None,
) -> Path:
    """Save one governed publication figure with deterministic metadata."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    footer_fields = [
        figure_id,
        f"Source: {source}",
        f"Sample: {sample}",
    ]
    if evidence_id:
        footer_fields.append(f"Evidence: {evidence_id}")

    fig.text(
        0.055,
        0.018,
        "  |  ".join(footer_fields),
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=PALETTE["muted"],
    )

    try:
        fig.savefig(
            output_path,
            dpi=300,
            facecolor=PALETTE["white"],
            bbox_inches="tight",
            pad_inches=0.18,
            metadata={
                "Software": "industrial-adaptive-energy-intelligence",
                "Title": figure_id,
            },
            pil_kwargs={
                "compress_level": 9,
                "optimize": False,
            },
        )
    finally:
        fig.clear()
        plt.close(fig)
        gc.collect()

    if not output_path.exists():
        raise ValueError(f"Rendered figure is missing: {output_path}")

    if output_path.stat().st_size < 30_000:
        raise ValueError(
            "Rendered figure is below the publication-size gate: "
            f"{output_path}"
        )

    with PILImage.open(output_path) as image:
        width, height = image.size

    if width < 2_400 or height < 1_200:
        raise ValueError(
            "Rendered figure is below the dimension gate: "
            f"{output_path} ({width}x{height})"
        )

    return output_path
