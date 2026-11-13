"""Class EDA: biểu đồ và phân tích thống kê cho House Prices."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew

sns.set_theme(style="whitegrid")


class EDA:
    def __init__(self, df: pd.DataFrame, target: str = "SalePrice", fig_dir=None):
        self.df = df
        self.target = target
        self.fig_dir = Path(fig_dir) if fig_dir else None

    # ---------- bảng tổng quan ----------
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

    def plot_missing(self):
        m = self.missing_report()
        if m.empty:
            print("Không có cột nào thiếu dữ liệu.")
            return
        fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(m))))
        sns.barplot(x=m.values, y=m.index, ax=ax, color="steelblue")
        ax.set_xlabel("% thiếu")
        ax.set_title("Tỉ lệ thiếu dữ liệu theo cột")
        self._save(fig, "missing_by_column.png")

    def skew_report(self, threshold: float = 0.75) -> pd.Series:
        num = self.df.select_dtypes("number").drop(columns=[self.target], errors="ignore")
        s = num.apply(lambda x: skew(x.dropna()))
        return s[s.abs() > threshold].sort_values(key=abs, ascending=False).round(2)

    # ---------- biến mục tiêu ----------
    def target_distribution(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(self.df[self.target], kde=True, ax=ax[0])
        ax[0].set_title(f"{self.target} (gốc), skew={skew(self.df[self.target]):.2f}")
        logged = np.log1p(self.df[self.target])
        sns.histplot(logged, kde=True, ax=ax[1], color="darkorange")
        ax[1].set_title(f"log1p({self.target}), skew={skew(logged):.2f}")
        fig.tight_layout()
        self._save(fig, "target_distribution.png")

    # ---------- quan hệ với biến mục tiêu ----------
    def top_correlations(self, k: int = 15) -> pd.Series:
        corr = self.df.select_dtypes("number").corr()[self.target]
        return corr.drop(self.target).sort_values(key=abs, ascending=False).head(k).round(3)

    def correlation_heatmap(self, k: int = 12):
        cols = list(self.top_correlations(k).index) + [self.target]
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(self.df[cols].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"Tương quan: {k} cột số liên quan nhất tới {self.target}")
        self._save(fig, "correlation_heatmap.png")

    def scatter_vs_target(self, col: str, annotate_id: bool = False):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=self.df, x=col, y=self.target, alpha=0.6, ax=ax)
        if annotate_id and "Id" in self.df.columns:
            # Đánh dấu các điểm nghi ngờ là ngoại lai để dễ đọc biểu đồ
            sus = self.df.nlargest(5, col)
            for _, r in sus.iterrows():
                ax.annotate(int(r["Id"]), (r[col], r[self.target]), fontsize=8)
        ax.set_title(f"{col} vs {self.target}")
        self._save(fig, f"scatter_{col}.png")

    def boxplot_by_category(self, col: str):
        order = (self.df.groupby(col)[self.target].median()
                 .sort_values().index)
        fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(order)), 5))
        sns.boxplot(data=self.df, x=col, y=self.target, order=order, ax=ax)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"{self.target} theo {col}")
        fig.tight_layout()
        self._save(fig, f"box_{col}.png")

    # ---------- phát hiện ngoại lai ----------
    def find_outliers(self, col: str = "GrLivArea",
                      area_gt: float = 4000, price_lt: float = 300000) -> pd.DataFrame:
        """Trả về các dòng diện tích rất lớn nhưng giá lại thấp."""
        mask = (self.df[col] > area_gt) & (self.df[self.target] < price_lt)
        cols = [c for c in ["Id", col, "OverallQual", "SaleCondition", self.target]
                if c in self.df.columns]
        return self.df.loc[mask, cols]

    def _save(self, fig, name: str):
        if self.fig_dir is not None:
            self.fig_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.fig_dir / name, bbox_inches="tight", dpi=120)
        plt.show()