"""CatBoost. Import đặt trong hàm để ai chưa cài vẫn chạy được các model khác."""
from src.model.base import BaseModel


class CatBoostModel(BaseModel):
    name = "catboost"

    def build_estimator(self):
        from catboost import CatBoostClassifier
        p = self.params
        return CatBoostClassifier(
            iterations=p.get("iterations", 800),
            learning_rate=p.get("learning_rate", 0.08),
            depth=p.get("depth", 6),
            random_seed=self.seed, verbose=0, allow_writing_files=False,
        )
