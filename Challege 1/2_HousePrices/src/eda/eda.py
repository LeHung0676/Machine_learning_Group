"""Class EDA: biểu đồ và phân tích thống kê."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDA:
    def __init__(self, df: pd.DataFrame, target: str = "SalePrice", fig_dir=None):
        self.df = df
        self.target = target
        self.fig_dir = fig_dir

    def overview(self) -> pd.DataFrame:
        return pd.DataFrame({
            "dtype": self.df.dtypes,
            "n_missing": self.df.isna().sum(),
            "pct_missing": self.df.isna().mean() * 100,
            "n_unique": self.df.nunique(),
        }).sort_values("pct_missing", ascending=False)

    def missing_report(self) -> pd.Series:
        m = self.df.isna().mean().mul(100)
        return m[m > 0].sort_values(ascending=False)

    def target_distribution(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(self.df[self.target], kde=True, ax=ax[0])
        sns.histplot(np.log1p(self.df[self.target]), kde=True, ax=ax[1])
        ax[1].set_title(f"log1p({self.target})")
        self._save(fig, "target_distribution.png")

    def top_correlations(self, k: int = 10) -> pd.Series:
        corr = self.df.select_dtypes("number").corr()[self.target]
        return corr.drop(self.target).abs().sort_values(ascending=False).head(k)

    def correlation_heatmap(self, k: int = 10):
        cols = list(self.top_correlations(k).index) + [self.target]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        self._save(fig, "correlation_heatmap.png")

    def scatter_vs_target(self, col: str):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=self.df, x=col, y=self.target, ax=ax)
        self._save(fig, f"scatter_{col}.png")

    def _save(self, fig, name: str):
        if self.fig_dir is not None:
            fig.savefig(f"{self.fig_dir}/{name}", bbox_inches="tight", dpi=120)
        plt.show()


