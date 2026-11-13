"""Tạo và chọn đặc trưng cho House Prices.

Điểm quan trọng: nhiều cột chữ trong bộ này có THỨ BẬC
(Ex > Gd > TA > Fa > Po). Nếu dùng one-hot thì mất hoàn toàn thông tin
thứ bậc đó. Vì vậy ta mã hóa chúng thành số theo đúng trật tự.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Thang chất lượng chung, dùng cho nhiều cột
QUAL = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

ORDINAL_MAPS: dict[str, dict] = {
    "ExterQual": QUAL, "ExterCond": QUAL,
    "BsmtQual": QUAL, "BsmtCond": QUAL,
    "HeatingQC": QUAL, "KitchenQual": QUAL,
    "FireplaceQu": QUAL, "GarageQual": QUAL, "GarageCond": QUAL,
    "PoolQC": QUAL,
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4,
                   "Min2": 5, "Min1": 6, "Typ": 7},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
}

# Cột số nhưng thực chất là mã phân loại, không có quan hệ lớn/nhỏ
AS_CATEGORY = ["MSSubClass", "MoSold"]


class FeatureEngineer:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    # ---------- bước 1: mã hóa thứ bậc ----------
    def encode_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, mapping in ORDINAL_MAPS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                # Giá trị lạ không có trong mapping -> điền 0 thay vì để NaN
                if df[col].isna().any():
                    df[col] = df[col].fillna(0)
                df[col] = df[col].astype(int)
        return df

    # ---------- bước 2: tạo đặc trưng mới ----------
    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.encode_ordinal(df)

        # --- Diện tích tổng: mạnh hơn từng cột riêng lẻ ---
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        df["TotalFinSF"] = df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["GrLivArea"]
        df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"]
                              + df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"])

        # --- Phòng tắm: gộp 4 cột rời thành 1 con số có nghĩa ---
        df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                           + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])

        # --- Tuổi nhà: quan trọng hơn năm xây tuyệt đối ---
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
        df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
        df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
        df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
        # Tuổi âm do lỗi dữ liệu -> kéo về 0
        for c in ["HouseAge", "RemodAge", "GarageAge"]:
            df[c] = df[c].clip(lower=0)

        # --- Điểm tổng hợp: nhân chất lượng với tình trạng ---
        df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
        df["ExterGrade"] = df["ExterQual"] * df["ExterCond"]
        df["GarageGrade"] = df["GarageQual"] * df["GarageCond"]
        df["BsmtGrade"] = df["BsmtQual"] * df["BsmtCond"]
        df["KitchenScore"] = df["KitchenAbvGr"] * df["KitchenQual"]
        df["FireplaceScore"] = df["Fireplaces"] * df["FireplaceQu"]
        df["GarageScore"] = df["GarageArea"] * df["GarageQual"]
        df["PoolScore"] = df["PoolArea"] * df["PoolQC"]

        # --- Có / không có: giúp model tuyến tính bắt bước nhảy ---
        df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
        df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
        df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
        df["HasPool"] = (df["PoolArea"] > 0).astype(int)
        df["HasPorch"] = (df["TotalPorchSF"] > 0).astype(int)

        # --- Tỉ lệ ---
        df["LivAreaPerRoom"] = df["GrLivArea"] / df["TotRmsAbvGrd"].clip(lower=1)
        df["BsmtFinRatio"] = (df["BsmtFinSF1"] + df["BsmtFinSF2"]) / df["TotalBsmtSF"].replace(0, np.nan)
        df["BsmtFinRatio"] = df["BsmtFinRatio"].fillna(0)
        df["LotAreaRatio"] = df["GrLivArea"] / df["LotArea"].clip(lower=1)

        # --- Cột số nhưng là mã phân loại ---
        for c in AS_CATEGORY:
            if c in df.columns:
                df[c] = df[c].astype(str)

        return df

    # ---------- bước 3: bỏ cột không cần ----------
    def select(self, df: pd.DataFrame, drop: list[str] | None = None) -> pd.DataFrame:
        if drop is None:
            drop = self.cfg.get("features", {}).get("drop", ["Street", "PoolArea", "MiscVal"])
        return df.drop(columns=[c for c in drop if c in df.columns])