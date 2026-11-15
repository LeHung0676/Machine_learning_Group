"""Extra Trees: như Random Forest nhưng chọn ngưỡng cắt ngẫu nhiên, đa dạng hơn."""
from sklearn.ensemble import ExtraTreesClassifier

from src.model.base import BaseModel


class ExtraTreesModel(BaseModel):
    name = "extratrees"

    def build_estimator(self):
        p = self.params
        return ExtraTreesClassifier(
            n_estimators=p.get("n_estimators", 400),
            min_samples_leaf=p.get("min_samples_leaf", 2),
            n_jobs=-1, random_state=self.seed,
        )
