"""Chuẩn hóa, scale, xử lý skew, encode."""
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


class Preprocessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg["preprocess"]
        self.scaler = SCALERS[self.cfg["scaler"]]()
        self.skewed_cols: list[str] = []
        self.num_cols: list[str] = []
        self.columns: list[str] = []

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.num_cols = X.select_dtypes("number").columns.tolist()
        sk = X[self.num_cols].apply(lambda s: skew(s.dropna()))
        self.skewed_cols = sk[sk.abs() > self.cfg["skew_threshold"]].index.tolist()
        X = self._log_skewed(X)
        X = pd.get_dummies(X, drop_first=False)
        self.columns = X.columns.tolist()
        X[self.num_cols] = self.scaler.fit_transform(X[self.num_cols])
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._log_skewed(X)
        X = pd.get_dummies(X).reindex(columns=self.columns, fill_value=0)
        X[self.num_cols] = self.scaler.transform(X[self.num_cols])
        return X

    def _log_skewed(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.skewed_cols:
            X[c] = np.log1p(X[c].clip(lower=0))
        return X
