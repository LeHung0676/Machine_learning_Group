"""Luồng: Clean -> Feature -> Preprocess -> Model -> Evaluate -> Ensemble -> Submit."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.data_loader import load_config, load_raw, get_path, save_df
from src.preprocess.Clean import Cleaner
from src.preprocess.Feature import FeatureEngineer
from src.preprocess.Preprocess import Preprocessor
from src.model.Lasso import LassoModel
from src.model.LightGBM import LightGBMModel
from src.evaluation.Evaluation import Evaluator
from src.ensemble.Ensemble import Blending
from src.tracking.tracking import Tracker

MODELS = {"lasso": LassoModel, "lightgbm": LightGBMModel}


class HousePricePipeline:
    def __init__(self, config_path=None, member: str = "team"):
        self.cfg = load_config(config_path) if config_path else load_config()
        self.member = member

    def run(self, model_names=("lasso", "lightgbm")):
        cfg, d = self.cfg, self.cfg["data"]
        tracker = Tracker(get_path(cfg, "experiments"), "pipeline", self.member)
        tracker.log_params(**cfg["models"], ensemble=cfg["ensemble"])

        # 1. Load + Clean
        train, test = load_raw(cfg)
        train, test = Cleaner(cfg).run(train, test, d["target"])
        save_df(train, cfg, "interim", "train_clean.csv")
        save_df(test, cfg, "interim", "test_clean.csv")

        y = np.log1p(train[d["target"]]) if d["log_target"] else train[d["target"]]
        test_ids = test[d["id_col"]]
        X = train.drop(columns=[d["target"], d["id_col"]])
        X_test = test.drop(columns=[d["id_col"]])

        # 2. Feature
        fe = FeatureEngineer(cfg)
        X, X_test = fe.select(fe.create(X)), fe.select(fe.create(X_test))

        # 3. Preprocess
        pp = Preprocessor(cfg)
        X = pp.fit_transform(X)
        X_test = pp.transform(X_test)
        save_df(X.assign(**{d["target"]: y.values}), cfg, "processed", "train_processed.csv")
        save_df(X_test, cfg, "processed", "test_processed.csv")

        # 4. Train + Evaluate
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=cfg["split"]["test_size"], random_state=cfg["project"]["seed"])
        val_preds, test_preds = {}, {}
        for name in model_names:
            m = MODELS[name](cfg).fit(X_tr, y_tr)
            val_preds[name] = m.predict(X_va)
            tracker.log_metrics(prefix=f"{name}_", **Evaluator.regression_metrics(y_va, val_preds[name]))
            full = MODELS[name](cfg).fit(X, y)
            full.save(get_path(cfg, "models") / f"{name}.joblib")
            test_preds[name] = full.predict(X_test)

        # 5. Ensemble
        blend = Blending({k: v for k, v in cfg["ensemble"]["weights"].items() if k in model_names})
        tracker.log_metrics(prefix="ensemble_", **Evaluator.regression_metrics(y_va, blend.predict(val_preds)))
        final = blend.predict(test_preds)
        if d["log_target"]:
            final = np.expm1(final)

        # 6. Submission
        out = get_path(cfg, "submission") / "submission.csv"
        pd.DataFrame({d["id_col"]: test_ids, d["target"]: final}).to_csv(out, index=False)
        tracker.save()
        return out
