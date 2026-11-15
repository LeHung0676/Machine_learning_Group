"""LightGBM. Import đặt trong hàm để ai chưa cài vẫn chạy được các model khác."""
from src.model.base import BaseModel


class LightGBMModel(BaseModel):
    name = "lightgbm"

    def build_estimator(self):
        from lightgbm import LGBMClassifier
        p = self.params
        return LGBMClassifier(
            n_estimators=p.get("n_estimators", 600),
            learning_rate=p.get("learning_rate", 0.05),
            num_leaves=p.get("num_leaves", 31),
            subsample=p.get("subsample", 0.8), subsample_freq=1,
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=self.seed, verbose=-1,
        )
