"""Model Lasso Regression."""
import joblib
from sklearn.linear_model import Lasso


class LassoModel:
    name = "lasso"

    def __init__(self, cfg: dict):
        p = cfg["models"]["lasso"]
        self.model = Lasso(alpha=p["alpha"], max_iter=p["max_iter"],
                           random_state=cfg["project"]["seed"])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self
