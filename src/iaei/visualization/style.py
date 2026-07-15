from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


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
    """Apply a repository-controlled, light-background publication style."""

    settings = {
        "figure.facecolor": PALETTE["white"],
        "axes.facecolor": PALETTE["white"],
        "savefig.facecolor": PALETTE["white"],
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 15,
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


def add_figure_header(fig: Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.055, y=0.975, ha="left", va="top", fontsize=15, fontweight="bold")
    fig.text(0.055, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color=PALETTE["muted"])


def save_publication_figure(
    fig: Figure,
    output_path: Path,
    *,
    figure_id: str,
    source: str,
    sample: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    footer = f"{figure_id}  |  Source: {source}  |  Sample: {sample}  |  Generated: {generated}"
    fig.text(0.055, 0.018, footer, ha="left", va="bottom", fontsize=6.8, color=PALETTE["muted"])
    fig.savefig(output_path, dpi=300, facecolor=PALETTE["white"], bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    if not output_path.exists() or output_path.stat().st_size < 30_000:
        raise ValueError(f"Rendered figure is missing or below the publication-size gate: {output_path}")
    return output_path
