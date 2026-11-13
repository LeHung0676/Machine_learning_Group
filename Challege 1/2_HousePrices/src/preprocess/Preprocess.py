"""Xử lý missing, duplicate, outlier cho bộ House Prices.

Nguyên tắc quan trọng: NA trong bộ này phần lớn KHÔNG phải lỗi dữ liệu,
mà có nghĩa "căn nhà không có hạng mục đó" (theo data_description.txt).
Vì vậy tuyệt đối không điền mean/median cho các cột trong NONE_COLS/ZERO_COLS.
"""
from __future__ import annotations

import pandas as pd

# NA = "không có hạng mục này" -> điền chuỗi "None"
NONE_COLS = [
    "Alley", "MasVnrType",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature",
]

# NA = "không có" -> điền 0
ZERO_COLS = [
    "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
    "GarageCars", "GarageArea",
]

# NA là thiếu thật -> điền mode học từ tập train
MODE_COLS = ["Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"]

# Cột gần như chỉ có một giá trị duy nhất -> bỏ, không mang thông tin
DROP_COLS = ["Utilities"]


class Cleaner:
    """Làm sạch dữ liệu. Học tham số từ train, áp dụng cho cả train và test."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.modes_: dict[str, object] = {}
        self.lot_frontage_by_nb_: pd.Series | None = None
        self.lot_frontage_global_: float = 0.0
        self.mszoning_by_nb_: pd.Series | None = None
        self.report_: dict = {}

    # ---------- fit ----------
    def fit(self, train: pd.DataFrame) -> "Cleaner":
        """Học mọi giá trị điền từ TẬP TRAIN, tránh rò rỉ thông tin từ test."""
        self.lot_frontage_by_nb_ = train.groupby("Neighborhood")["LotFrontage"].median()
        self.lot_frontage_global_ = float(train["LotFrontage"].median())
        self.mszoning_by_nb_ = train.groupby("Neighborhood")["MSZoning"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else "RL"
        )
        for c in MODE_COLS:
            if c in train.columns and not train[c].mode().empty:
                self.modes_[c] = train[c].mode().iloc[0]
        return self

    # ---------- transform ----------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Bỏ cột không mang thông tin
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        # 2. NA có nghĩa "không có"
        for c in NONE_COLS:
            if c in df.columns:
                df[c] = df[c].fillna("None")
        for c in ZERO_COLS:
            if c in df.columns:
                df[c] = df[c].fillna(0)

        # 3. Functional: data_description ghi rõ "assume typical" khi thiếu
        if "Functional" in df.columns:
            df["Functional"] = df["Functional"].fillna("Typ")

        # 4. LotFrontage: điền theo median của cùng khu dân cư.
        #    Các nhà cùng Neighborhood có mặt tiền tương tự nhau,
        #    nên cách này sát thực tế hơn median toàn bộ.
        if "LotFrontage" in df.columns:
            df["LotFrontage"] = df["LotFrontage"].fillna(
                df["Neighborhood"].map(self.lot_frontage_by_nb_)
            ).fillna(self.lot_frontage_global_)

        # 5. MSZoning: điền theo mode của cùng khu dân cư
        if "MSZoning" in df.columns:
            df["MSZoning"] = df["MSZoning"].fillna(
                df["Neighborhood"].map(self.mszoning_by_nb_)
            )

        # 6. GarageYrBlt: nhà không có garage thì điền bằng năm xây nhà,
        #    KHÔNG điền 0 vì sẽ làm cột tuổi garage sai lệch rất nặng.
        if "GarageYrBlt" in df.columns:
            df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])
            # Sửa lỗi gốc trong dữ liệu: có 1 dòng ở test ghi GarageYrBlt = 2207
            bad = df["GarageYrBlt"] > df["YrSold"]
            df.loc[bad, "GarageYrBlt"] = df.loc[bad, "YearBuilt"]

        # 7. Các cột thiếu thật còn lại -> mode học từ train
        for c, v in self.modes_.items():
            if c in df.columns:
                df[c] = df[c].fillna(v)

        # 8. Lưới an toàn: nếu còn NA nào chưa lường trước
        leftover = [c for c in df.columns if df[c].isna().any()]
        if leftover:
            for c in leftover:
                if df[c].dtype == "O":
                    df[c] = df[c].fillna(df[c].mode().iloc[0])
                else:
                    df[c] = df[c].fillna(df[c].median())
        self.report_["filled_by_fallback"] = leftover

        return df

    # ---------- outlier ----------
    def remove_outliers(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """CHỈ áp dụng cho train. Không bao giờ bỏ dòng của test."""
        before = len(df)
        removed = []
        for r in self.cfg["clean"].get("outlier_rules", []):
            mask = (df[r["col"]] > r["gt"]) & (df[target] < r["target_lt"])
            if "Id" in df.columns:
                removed += df.loc[mask, "Id"].tolist()
            df = df[~mask]
        self.report_["outliers_removed_ids"] = removed
        self.report_["rows_before"] = before
        self.report_["rows_after"] = len(df)
        return df.reset_index(drop=True)

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in df.columns if c != "Id"]
        n = len(df)
        df = df.drop_duplicates(subset=cols).reset_index(drop=True)
        self.report_["duplicates_removed"] = n - len(df)
        return df

    # ---------- entry point dùng trong Pipeline ----------
    def run(self, train: pd.DataFrame, test: pd.DataFrame, target: str):
        train = self.drop_duplicates(train)
        train = self.remove_outliers(train, target)
        self.fit(train)
        return self.transform(train), self.transform(test)

    def summary(self) -> str:
        """Dùng cho note kỹ thuật trong docs/."""
        r = self.report_
        return (
            f"- Dòng trước khi làm sạch: {r.get('rows_before')}\n"
            f"- Dòng sau khi làm sạch: {r.get('rows_after')}\n"
            f"- Trùng lặp đã bỏ: {r.get('duplicates_removed')}\n"
            f"- Id ngoại lai đã bỏ: {r.get('outliers_removed_ids')}\n"
            f"- Cột phải dùng lưới an toàn: {r.get('filled_by_fallback')}\n"
        )