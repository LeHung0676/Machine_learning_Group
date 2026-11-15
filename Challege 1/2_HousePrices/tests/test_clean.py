import pandas as pd
from src.preprocess.Clean import Cleaner

CFG = {"clean": {"outlier_rules": [{"col": "GrLivArea", "gt": 4000, "target_lt": 300000}]}}


def test_fill_missing_no_nan():
    df = pd.DataFrame({"PoolQC": [None, "Ex"], "GarageArea": [None, 200.0], "X": [1.0, None]})
    out = Cleaner(CFG).fill_missing(df)
    assert out.isna().sum().sum() == 0
    assert out.loc[0, "PoolQC"] == "None"


def test_remove_outliers():
    df = pd.DataFrame({"GrLivArea": [5000, 1000], "SalePrice": [100000, 200000]})
    assert len(Cleaner(CFG).remove_outliers(df, "SalePrice")) == 1
