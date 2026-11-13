"""Model LightGBM."""
import joblib


class LightGBMModel:
    name = "lightgbm"

    def __init__(self, cfg: dict):
        from lightgbm import LGBMRegressor  # import khi dùng, tránh lỗi nếu chưa cài
        p = cfg["models"]["lightgbm"]
        self.model = LGBMRegressor(**p, random_state=cfg["project"]["seed"], verbose=-1)

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is not None:
            from lightgbm import early_stopping
            self.model.fit(X, y, eval_set=[(X_val, y_val)],
                           callbacks=[early_stopping(100, verbose=False)])
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self
