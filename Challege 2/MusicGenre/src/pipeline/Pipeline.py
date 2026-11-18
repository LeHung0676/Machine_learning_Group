"""Luồng đầy đủ: Clean -> Feature -> Preprocess -> OOF từng model -> Ensemble -> Submit."""
from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pandas as pd

from src.ensemble import Ensemble
from src.evaluation.Evaluation import Evaluator
from src.preprocess.Clean import Cleaner
from src.preprocess.Feature import FeatureEngineer
from src.preprocess.Preprocess import Preprocessor
from src.tracking.tracking import Tracker
from src.utils.data_loader import get_path, load_config, load_raw, save_df

# Mỗi model một file trong src/model/. Model cần thư viện ngoài chỉ được đăng ký
# khi thư viện đã cài (kiểm tra bằng find_spec vì các file đó import trong hàm).
MODELS: dict[str, type] = {}


def _register(name, module_path, class_name, requires=None):
    if requires is not None and importlib.util.find_spec(requires) is None:
        return
    MODELS[name] = getattr(importlib.import_module(module_path), class_name)


_register("histgb",       "src.model.HistGB",       "HistGBModel")
_register("randomforest", "src.model.RandomForest", "RandomForestModel")
_register("extratrees",   "src.model.ExtraTrees",   "ExtraTreesModel")
_register("logreg",       "src.model.LogReg",       "LogRegModel")
_register("text",         "src.model.TextModel",    "TextModel")
_register("lightgbm",     "src.model.LightGBM",     "LightGBMModel", requires="lightgbm")
_register("xgboost",      "src.model.XGBoost",      "XGBoostModel",  requires="xgboost")
_register("catboost",     "src.model.CatBoost",     "CatBoostModel", requires="catboost")


def available_models() -> list[str]:
    return list(MODELS)


class MusicGenrePipeline:
    def __init__(self, config_path=None, member: str = "team"):
        self.cfg = load_config(config_path)
        self.member = member
        self.cleaner = None
        self.pre = None

    def prepare(self):
        cfg, d = self.cfg, self.cfg["data"]
        train, test = load_raw(cfg)

        self.cleaner = Cleaner(cfg)
        train, test = self.cleaner.run(train, test, d["target"])
        save_df(train, cfg, "interim", "train_clean.csv")
        save_df(test, cfg, "interim", "test_clean.csv")

        y = train[d["target"]].astype(int).reset_index(drop=True)
        test_ids = test[d["id_col"]].reset_index(drop=True)
        fe = FeatureEngineer(cfg)
        X = fe.create(train.drop(columns=[d["target"], d["id_col"]]))
        X_test = fe.create(test.drop(columns=[d["id_col"]]))

        self.pre = Preprocessor(cfg)
        X = self.pre.fit_transform(X).reset_index(drop=True)
        X_test = self.pre.transform(X_test).reset_index(drop=True)
        if list(X.columns) != list(X_test.columns):
            raise RuntimeError("Train và test lệch cột sau tiền xử lý")

        save_df(X.assign(**{d["target"]: y.values}), cfg, "processed", "train_processed.csv")
        save_df(X_test, cfg, "processed", "test_processed.csv")
        return X, y, X_test, test_ids

    def run(self, model_names=None):
        cfg, d = self.cfg, self.cfg["data"]
        seed, n_folds = cfg["project"]["seed"], cfg["split"]["n_folds"]
        tracker = Tracker(get_path(cfg, "experiments"), "pipeline", self.member)

        if model_names is None:
            model_names = cfg.get("run", {}).get("models") or available_models()
        missing = [m for m in model_names if m not in MODELS]
        model_names = [m for m in model_names if m in MODELS]
        if missing:
            tracker.logger.warning(f"Bỏ qua model chưa khả dụng (thiếu thư viện?): {missing}")
        if not model_names:
            raise RuntimeError(f"Không có model nào khả dụng. Đã yêu cầu: {missing}")
        tracker.log_params(models=model_names, n_folds=n_folds, ensemble=cfg["ensemble"],
                           params={m: cfg["models"].get(m) for m in model_names})

        X, y, X_test, test_ids = self.prepare()
        tracker.logger.info(f"Dữ liệu sau xử lý: train {X.shape}, test {X_test.shape}")

        results, oof, test_proba = {}, {}, {}
        for name in model_names:
            r = Evaluator.oof_predict(lambda n=name: MODELS[n](cfg, self.pre), X, y, X_test,
                                      n_folds=n_folds, seed=seed)
            results[name], oof[name], test_proba[name] = r, r["oof"], r["test_proba"]
            tracker.log_metrics(prefix=f"{name}_", accuracy=r["accuracy"], acc_std=r["acc_std"],
                                f1_macro=r["f1_macro"], log_loss=r["log_loss"])
        classes = results[model_names[0]]["classes"]
        tracker.log_table("model_comparison", Evaluator.compare(results))

        if len(model_names) > 1:
            tracker.log_table("ensemble_comparison", Ensemble.compare_methods(oof, y, classes, seed))
            ens = Ensemble.build(cfg["ensemble"]["method"], model_names, cfg).fit(oof, y)
            if cfg["ensemble"]["method"] == "stacking":
                final_oof = ens.oof_proba(oof, y, seed=seed)
            else:
                final_oof = ens.predict_proba(oof)
            final_test = ens.predict_proba(test_proba)
            m = Evaluator.metrics(y, final_oof, classes)
            tracker.log_metrics(prefix="ensemble_", method=cfg["ensemble"]["method"], **m,
                                weights={k: round(float(v), 4) for k, v in (ens.weights or {}).items()})
        else:
            final_oof, final_test = oof[model_names[0]], test_proba[model_names[0]]

        tracker.log_table("per_class", Evaluator.per_class(y, final_oof, classes))
        Evaluator.plot_confusion(y, final_oof, classes, fig_dir=get_path(cfg, "figures"))
        np.save(tracker.dir / "oof_final.npy", final_oof)

        for name in model_names:      # huấn luyện lại trên toàn bộ train để lưu
            MODELS[name](cfg, self.pre).fit(X, y).save(get_path(cfg, "models") / f"{name}.joblib")

        pred = classes[final_test.argmax(axis=1)]
        sub = pd.DataFrame({d["id_col"]: test_ids, d["target"]: pred})
        if len(sub) != len(X_test):
            raise RuntimeError("Số dòng file nộp không khớp số dòng test")
        out = get_path(cfg, "submission") / "submission.csv"
        sub.to_csv(out, index=False)
        tracker.log_metrics(submission_rows=len(sub),
                            pred_distribution=sub[d["target"]].value_counts().sort_index().to_dict())
        tracker.save()
        return out
