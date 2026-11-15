"""XGBoost. Import đặt trong hàm để ai chưa cài vẫn chạy được các model khác."""
from src.model.base import BaseModel


class XGBoostModel(BaseModel):
    name = "xgboost"

    def build_estimator(self):
        from xgboost import XGBClassifier
        p = self.params
        return XGBClassifier(
            n_estimators=p.get("n_estimators", 600),
            learning_rate=p.get("learning_rate", 0.05),
            max_depth=p.get("max_depth", 6),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=self.seed, n_jobs=-1, verbosity=0,
        )
