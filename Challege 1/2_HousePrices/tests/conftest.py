"""Fixture dùng chung: một DataFrame nhỏ mô phỏng bộ House Prices."""
import numpy as np
import pandas as pd
import pytest

from src.preprocess.Clean import Cleaner
from src.preprocess.Feature import FeatureEngineer

CFG = {
    "project": {"seed": 42},
    "clean": {"outlier_rules": [{"col": "GrLivArea", "gt": 4000, "target_lt": 300000}]},
    "preprocess": {"scaler": "robust", "skew_threshold": 0.75, "skew_min_unique": 3},
    "features": {"drop": ["Street", "PoolArea", "MiscVal"]},
}

NUM = {
    "MSSubClass":    [20, 60, 20, 70, 20, 60],
    "LotFrontage":   [65.0, np.nan, 80.0, 60.0, np.nan, 75.0],
    "LotArea":       [8450, 9600, 11250, 9550, 14260, 10000],
    "OverallQual":   [7, 6, 7, 7, 8, 5],
    "OverallCond":   [5, 8, 5, 5, 5, 6],
    "YearBuilt":     [2003, 1976, 2001, 1915, 2000, 1993],
    "YearRemodAdd":  [2003, 1976, 2002, 1970, 2000, 1995],
    "MasVnrArea":    [196.0, 0.0, 162.0, np.nan, 350.0, 0.0],
    "BsmtFinSF1":    [706, 978, 486, 216, 655, 0],
    "BsmtFinSF2":    [0, 0, 0, 0, 0, 0],
    "BsmtUnfSF":     [150, 284, 434, 540, 490, 0],
    "TotalBsmtSF":   [856, 1262, 920, 756, 1145, 0],
    "1stFlrSF":      [856, 1262, 920, 961, 1145, 900],
    "2ndFlrSF":      [854, 0, 866, 756, 0, 0],
    "GrLivArea":     [1710, 1262, 1786, 1717, 4500, 900],
    "BsmtFullBath":  [1, 0, 1, 1, 1, 0],
    "BsmtHalfBath":  [0, 1, 0, 0, 0, 0],
    "FullBath":      [2, 2, 2, 1, 2, 1],
    "HalfBath":      [1, 0, 1, 0, 1, 0],
    "BedroomAbvGr":  [3, 3, 3, 3, 4, 2],
    "KitchenAbvGr":  [1, 1, 1, 1, 1, 1],
    "TotRmsAbvGrd":  [8, 6, 6, 7, 9, 4],
    "Fireplaces":    [0, 1, 1, 1, 1, 0],
    "GarageYrBlt":   [2003.0, 1976.0, 2001.0, np.nan, 2000.0, 2207.0],
    "GarageCars":    [2, 2, 2, 0, 3, 1],
    "GarageArea":    [548.0, 460.0, 608.0, 0.0, 836.0, np.nan],
    "WoodDeckSF":    [0, 298, 0, 0, 192, 0],
    "OpenPorchSF":   [61, 0, 42, 35, 84, 0],
    "EnclosedPorch": [0, 0, 0, 272, 0, 0],
    "3SsnPorch":     [0, 0, 0, 0, 0, 0],
    "ScreenPorch":   [0, 0, 0, 0, 0, 0],
    "PoolArea":      [0, 0, 0, 0, 0, 0],
    "MiscVal":       [0, 0, 0, 0, 0, 0],
    "MoSold":        [2, 5, 9, 2, 12, 6],
    "YrSold":        [2008, 2007, 2008, 2006, 2008, 2009],
}

CAT = {
    "MSZoning":      ["RL", "RL", "RL", "RL", "RL", None],
    "Street":        ["Pave"] * 6,
    "Alley":         [None, None, None, None, None, "Grvl"],
    "LotShape":      ["Reg", "Reg", "IR1", "IR1", "IR1", "Reg"],
    "LandSlope":     ["Gtl"] * 6,
    "Utilities":     ["AllPub"] * 6,
    "Neighborhood":  ["CollgCr", "Veenker", "CollgCr", "Crawfor", "NoRidge", "Veenker"],
    "BldgType":      ["1Fam"] * 6,
    "HouseStyle":    ["2Story", "1Story", "2Story", "2Story", "1Story", "1Story"],
    "Exterior1st":   ["VinylSd", "MetalSd", "VinylSd", "Wd Sdng", "VinylSd", None],
    "Exterior2nd":   ["VinylSd", "MetalSd", "VinylSd", "Wd Shng", "VinylSd", None],
    "MasVnrType":    ["BrkFace", None, "BrkFace", None, "BrkFace", None],
    "ExterQual":     ["Gd", "TA", "Gd", "TA", "Gd", "TA"],
    "ExterCond":     ["TA"] * 6,
    "Foundation":    ["PConc", "CBlock", "PConc", "BrkTil", "PConc", "CBlock"],
    "BsmtQual":      ["Gd", "Gd", "Gd", "TA", "Gd", None],
    "BsmtCond":      ["TA", "TA", "TA", "Gd", "TA", None],
    "BsmtExposure":  ["No", "Gd", "Mn", "No", "Av", None],
    "BsmtFinType1":  ["GLQ", "ALQ", "GLQ", "ALQ", "GLQ", None],
    "BsmtFinType2":  ["Unf"] * 5 + [None],
    "HeatingQC":     ["Ex", "Ex", "Ex", "Gd", "Ex", "TA"],
    "CentralAir":    ["Y", "Y", "Y", "Y", "Y", "N"],
    "Electrical":    ["SBrkr", "SBrkr", "SBrkr", "SBrkr", None, "SBrkr"],
    "KitchenQual":   ["Gd", "TA", "Gd", "Gd", "Gd", None],
    "Functional":    ["Typ", "Typ", "Typ", "Typ", None, "Typ"],
    "FireplaceQu":   [None, "TA", "TA", "Gd", "TA", None],
    "GarageType":    ["Attchd", "Attchd", "Attchd", None, "Attchd", "Detchd"],
    "GarageFinish":  ["RFn", "RFn", "RFn", None, "Fin", "Unf"],
    "GarageQual":    ["TA", "TA", "TA", None, "TA", "TA"],
    "GarageCond":    ["TA", "TA", "TA", None, "TA", "TA"],
    "PavedDrive":    ["Y"] * 6,
    "PoolQC":        [None] * 6,
    "Fence":         [None, None, None, None, None, "MnPrv"],
    "MiscFeature":   [None] * 6,
    "SaleType":      ["WD", "WD", "WD", "WD", "WD", None],
    "SaleCondition": ["Normal", "Normal", "Normal", "Abnorml", "Normal", "Normal"],
}


def _make(first_id: int, with_target: bool) -> pd.DataFrame:
    df = pd.DataFrame({**NUM, **CAT})
    df.insert(0, "Id", range(first_id, first_id + len(df)))
    if with_target:
        df["SalePrice"] = [208500, 181500, 223500, 140000, 250000, 120000]
    return df


@pytest.fixture
def cfg():
    return CFG


@pytest.fixture
def train_df():
    """Id 1..6. Id 5 là ngoại lai: GrLivArea 4500 nhưng giá chỉ 250k."""
    return _make(1, with_target=True)


@pytest.fixture
def test_df():
    """Id 101..106, không có SalePrice."""
    return _make(101, with_target=False)


@pytest.fixture
def cleaned(cfg, train_df, test_df):
    return Cleaner(cfg).run(train_df, test_df, "SalePrice")


@pytest.fixture
def featured(cfg, cleaned):
    tr, te = cleaned
    fe = FeatureEngineer(cfg)
    Xtr = fe.select(fe.create(tr.drop(columns=["SalePrice", "Id"])))
    Xte = fe.select(fe.create(te.drop(columns=["Id"])))
    return Xtr, Xte
