"""Tạo và chọn đặc trưng."""
import pandas as pd


class FeatureEngineer:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                           + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
        df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"]
                              + df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"])
        df["HasPool"] = (df["PoolArea"] > 0).astype(int)
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
        # Các cột số thực chất là phân loại
        for c in ["MSSubClass", "MoSold", "YrSold"]:
            if c in df: df[c] = df[c].astype(str)
        return df

    def select(self, df: pd.DataFrame, drop: list[str] | None = None) -> pd.DataFrame:
        drop = drop or ["Utilities", "Street", "PoolQC"]
        return df.drop(columns=[c for c in drop if c in df])
