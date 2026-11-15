"""Histogram Gradient Boosting: cây tăng cường tuần tự, model mạnh nhất của nhóm."""
from sklearn.ensemble import HistGradientBoostingClassifier

from src.model.base import BaseModel


class HistGBModel(BaseModel):
    name = "histgb"

    def build_estimator(self):
        p = self.params
        return HistGradientBoostingClassifier(
            max_iter=p.get("max_iter", 300),
            learning_rate=p.get("learning_rate", 0.06),
            random_state=self.seed,
        )
