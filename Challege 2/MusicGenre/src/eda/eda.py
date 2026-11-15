"""Class EDA cho Music Genre: thống kê và biểu đồ khám phá."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Tên thể loại theo bảng mã của bộ dữ liệu gốc (MachineHack Music Genre).
# Hãy đối chiếu với đề bài của lớp trước khi dùng trong báo cáo.
GENRES = {0: "Acoustic/Folk", 1: "Alt Music", 2: "Blues", 3: "Bollywood", 4: "Country",
          5: "HipHop", 6: "Indie Alt", 7: "Instrumental", 8: "Metal", 9: "Pop", 10: "Rock"}


class EDA:
    def __init__(self, df: pd.DataFrame, target: str = "Class", fig_dir=None):
        self.df = df
        self.target = target
        self.fig_dir = Path(fig_dir) if fig_dir else None

    def overview(self) -> pd.DataFrame:
        return pd.DataFrame({
            "dtype": self.df.dtypes.astype(str),
            "n_missing": self.df.isna().sum(),
            "pct_missing": self.df.isna().mean().mul(100).round(2),
            "n_unique": self.df.nunique(),
        }).sort_values("pct_missing", ascending=False)

    def missing_report(self) -> pd.Series:
        m = self.df.isna().mean().mul(100)
        return m[m > 0].sort_values(ascending=False).round(2)

    def missing_by_class(self, col: str) -> pd.Series:
        """Tỉ lệ thiếu của một cột theo từng lớp: thiếu ngẫu nhiên hay có quy luật?"""
        return self.df.groupby(self.target)[col].apply(lambda s: s.isna().mean()).round(3)

    def class_distribution(self) -> pd.DataFrame:
        c = self.df[self.target].value_counts().sort_index()
        return pd.DataFrame({"genre": [GENRES.get(i, str(i)) for i in c.index],
                             "count": c.values, "pct": (c / c.sum() * 100).round(1).values},
                            index=c.index)

    def plot_class_distribution(self):
        d = self.class_distribution()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar([f"{i}\n{g}" for i, g in zip(d.index, d["genre"])], d["count"], color="#2F6FD0")
        ax.set_ylabel("Số bài hát"); ax.set_title("Phân bố 11 lớp thể loại")
        ax.tick_params(axis="x", labelsize=8)
        self._save(fig, "class_distribution.png")

    def duration_units(self, col: str = "duration_in min/ms", threshold: float = 30) -> dict:
        """Phát hiện cột thời lượng trộn hai đơn vị."""
        d = self.df[col]
        small = d[d < threshold]
        return {"n_minutes": int(len(small)), "pct_minutes": round(len(small) / len(d) * 100, 1),
                "max_small": float(small.max()), "min_large": float(d[d >= threshold].min())}

    def duration_unit_by_class(self, col: str = "duration_in min/ms", threshold: float = 30) -> pd.Series:
        return self.df.groupby(self.target)[col].apply(lambda s: (s < threshold).mean()).round(3)

    def key_values(self) -> pd.Series:
        """Đếm giá trị của key, kể cả NA. Nếu không có 0 thì NA chính là 0."""
        return self.df["key"].value_counts(dropna=False).sort_index()

    def duplicate_tracks(self) -> dict:
        """Cùng (nghệ sĩ, tên bài) xuất hiện nhiều lần, có khi với nhãn khác nhau."""
        g = self.df.groupby(["Artist Name", "Track Name"])[self.target]
        size, nun = g.size(), g.nunique()
        return {"pairs_repeated": int((size > 1).sum()),
                "pairs_multi_label": int((nun > 1).sum()),
                "rows_in_repeated_pairs": int(size[size > 1].sum())}

    def boxplot_by_class(self, col: str):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        classes = sorted(self.df[self.target].unique())
        groups = [self.df.loc[self.df[self.target] == c, col].dropna() for c in classes]
        ax.boxplot(groups, showfliers=False)
        # Đặt nhãn sau khi vẽ: tham số nhãn của boxplot đổi tên giữa các bản matplotlib
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels(classes)
        ax.set_xlabel("Lớp"); ax.set_ylabel(col); ax.set_title(f"{col} theo lớp")
        self._save(fig, f"box_{col}.png")

    def _save(self, fig, name: str):
        if self.fig_dir is not None:
            self.fig_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.fig_dir / name, bbox_inches="tight", dpi=120)
        plt.show()
