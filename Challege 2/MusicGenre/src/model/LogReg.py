"""Hồi quy logistic đa lớp trên đặc trưng đã chuẩn hóa: model tuyến tính đối chứng."""
from sklearn.linear_model import LogisticRegression

from src.model.base import BaseModel


class LogRegModel(BaseModel):
    name = "logreg"
    scale = True

    def build_estimator(self):
        p = self.params
        return LogisticRegression(C=p.get("C", 1.0), max_iter=p.get("max_iter", 2000))
