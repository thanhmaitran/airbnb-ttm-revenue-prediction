"""Publication-style plotting helpers shared by every figure in
reports/figures/. Centralising style (font, DPI, palette) here is what
Stage 7 of the roadmap calls making figures 'publication style' instead of
default-matplotlib style.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

DPI = 150
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#8172B2"]


def set_publication_style() -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.axisbelow": True,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    })


def save_figure(fig, path: str | Path, tight: bool = True) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
