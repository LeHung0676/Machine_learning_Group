"""Tiện ích vẽ hình dùng chung: thiết lập style thống nhất và lưu hình cho báo cáo/slide/poster."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR

PALETTE = ["#2a6fdb", "#e4572e", "#29a19c", "#f2a541", "#7d5ba6", "#5b5f97"]


def set_style() -> None:
    """Style chung cho mọi hình của dự án."""
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })


def save_fig(fig: plt.Figure, name: str, dpi: int = 150) -> None:
    """Lưu hình vào reports/figures/<name>.png (dùng lại cho báo cáo, slide, poster)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Đã lưu hình: {path.relative_to(FIGURES_DIR.parents[1])}")
