"""Voting, Stacking, Bagging, Blending."""
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


class Blending:
    """Trung bình có trọng số các dự đoán."""
    def __init__(self, weights: dict):
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}

    def predict(self, preds: dict) -> np.ndarray:
        return sum(self.weights[k] * preds[k] for k in self.weights)


class Voting(Blending):
    """Trung bình đều các model."""
    def __init__(self, names: list[str]):
        super().__init__({n: 1.0 for n in names})


class Stacking:
    """Out-of-fold predictions -> meta model."""
    def __init__(self, base_factories: dict, meta=None, n_folds: int = 5, seed: int = 42):
        self.base_factories = base_factories
        self.meta = meta or Ridge(alpha=1.0)
        self.kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        self.fitted = {}

    def fit(self, X, y):
        oof = np.zeros((len(X), len(self.base_factories)))
        for j, (name, f) in enumerate(self.base_factories.items()):
            for tr, va in self.kf.split(X):
                oof[va, j] = f().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
            self.fitted[name] = f().fit(X, y)
        self.meta.fit(oof, y)
        return self

    def predict(self, X):
        base = np.column_stack([m.predict(X) for m in self.fitted.values()])
        return self.meta.predict(base)


def bagging(estimator, n_estimators: int = 10, seed: int = 42):
    return BaggingRegressor(estimator=estimator, n_estimators=n_estimators, random_state=seed)
