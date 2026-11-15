"""Luồng đầy đủ: Clean -> Feature -> Preprocess -> CV từng model -> Ensemble -> Submit.

Điểm khác bản đầu: mọi model được đánh giá bằng K-fold sinh dự đoán OOF,
thay vì một lần chia train/test. Với 1460 dòng, một lần chia cho con số
may rủi tới 0.01 RMSE, đủ để cả nhóm kết luận sai về đặc trưng nào tốt.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from src.ensemble import Ensemble
from src.evaluation.Evaluation import Evaluator, rmse
from src.preprocess.Clean import Cleaner
from src.preprocess.Feature import FeatureEngineer
from src.preprocess.Preprocess import Preprocessor
from src.tracking.tracking import Tracker
from src.utils.data_loader import get_path, load_config, load_raw, save_df

# ---------- Bảng đăng ký model ----------
# Mỗi model nằm một file riêng trong src/model/, giống quy ước của repo mẫu.
# Model cần thư viện ngoài (lightgbm, xgboost, catboost) chỉ được đăng ký khi
# thư viện đó đã cài. Phải kiểm tra bằng find_spec chứ không dùng try/except
# quanh câu import, vì các file đó cố tình import thư viện bên trong hàm,
# nên câu import ở đây luôn thành công dù thư viện chưa có.
MODELS: dict[str, type] = {}


def _register(name: str, module_path: str, class_name: str, requires: str | None = None):
    if requires is not None and importlib.util.find_spec(requires) is None:
        return
    MODELS[name] = getattr(importlib.import_module(module_path), class_name)


_register("lasso",        "src.model.Lasso",        "LassoModel")
_register("elasticnet",   "src.model.ElasticNet",   "ElasticNetModel")
_register("kernelridge",  "src.model.KernelRidge",  "KernelRidgeModel")
_register("randomforest", "src.model.RandomForest", "RandomForestModel")
_register("lightgbm",     "src.model.LightGBM",     "LightGBMModel",  requires="lightgbm")
_register("xgboost",      "src.model.XGBoost",      "XGBoostModel",   requires="xgboost")
_register("catboost",     "src.model.CatBoost",     "CatBoostModel",  requires="catboost")


def available_models() -> list[str]:
    """Danh sách model dùng được trên máy hiện tại."""
    return list(MODELS)


class HousePricePipeline:
    def __init__(self, config_path=None, member: str = "team"):
        self.cfg = load_config(config_path)
        self.member = member
        self.cleaner = None
        self.preprocessor = None

    # ---------- chuẩn bị dữ liệu ----------
    def prepare(self):
        cfg, d = self.cfg, self.cfg["data"]
        train, test = load_raw(cfg)

        self.cleaner = Cleaner(cfg)
        train, test = self.cleaner.run(train, test, d["target"])
        save_df(train, cfg, "interim", "train_clean.csv")
        save_df(test, cfg, "interim", "test_clean.csv")

        y = np.log1p(train[d["target"]]) if d["log_target"] else train[d["target"]].astype(float)
        test_ids = test[d["id_col"]].reset_index(drop=True)
        X = train.drop(columns=[d["target"], d["id_col"]])
        X_test = test.drop(columns=[d["id_col"]])

        fe = FeatureEngineer(cfg)
        # .copy() để gom lại bộ nhớ: Feature tạo gần 30 cột bằng cách gán lần lượt,
        # khiến pandas cảnh báo DataFrame bị phân mảnh ở các bước sau.
        X = fe.select(fe.create(X)).copy()
        X_test = fe.select(fe.create(X_test)).copy()

        self.preprocessor = Preprocessor(cfg)
        X = self.preprocessor.fit_transform(X)
        X_test = self.preprocessor.transform(X_test)

        if list(X.columns) != list(X_test.columns):
            raise RuntimeError("Train và test lệch cột sau tiền xử lý")

        save_df(X.assign(**{d["target"]: y.values}), cfg, "processed", "train_processed.csv")
        save_df(X_test, cfg, "processed", "test_processed.csv")
        return X.reset_index(drop=True), y.reset_index(drop=True), X_test, test_ids

    # ---------- chạy toàn bộ ----------
    def run(self, model_names=None):
        cfg, d = self.cfg, self.cfg["data"]
        # Không truyền thì lấy từ config, không có trong config thì lấy tất cả
        if model_names is None:
            model_names = cfg.get("run", {}).get("models") or available_models()
        tracker = Tracker(get_path(cfg, "experiments"), "pipeline", self.member)

        missing = [m for m in model_names if m not in MODELS]
        model_names = [m for m in model_names if m in MODELS]
        if missing:
            tracker.logger.warning(
                f"Bỏ qua model chưa khả dụng (thiếu thư viện?): {missing}")
        if not model_names:
            raise RuntimeError(f"Không có model nào khả dụng. Đã yêu cầu: {missing}")
        tracker.log_params(models=model_names,
                           params={m: cfg["models"].get(m) for m in model_names},
                           ensemble=cfg["ensemble"], n_folds=cfg["split"]["n_folds"])

        X, y, X_test, test_ids = self.prepare()
        tracker.logger.info(f"Dữ liệu sau xử lý: train {X.shape}, test {X_test.shape}")

        # --- cross-validation từng model, thu dự đoán OOF ---
        results, oof, test_preds = {}, {}, {}
        for name in model_names:
            r = Evaluator.oof_predict(
                lambda n=name: MODELS[n](cfg), X, y, X_test,
                n_folds=cfg["split"]["n_folds"], seed=cfg["project"]["seed"])
            results[name], oof[name], test_preds[name] = r, r["oof"], r["test_pred"]
            tracker.log_metrics(prefix=f"{name}_", rmse_oof=r["rmse_oof"],
                                rmse_cv_mean=r["rmse_mean"], rmse_cv_std=r["rmse_std"])

        tracker.log_table("model_comparison", Evaluator.compare(results))

        # --- kết hợp model ---
        if len(model_names) > 1:
            tracker.log_table("ensemble_comparison", Ensemble.compare_methods(oof, y))
            ens = Ensemble.build(cfg["ensemble"]["method"], model_names, cfg)
            ens.fit(oof, y)
            final_log = ens.predict(test_preds)
            tracker.log_metrics(prefix="ensemble_",
                                method=cfg["ensemble"]["method"],
                                rmse_oof=rmse(y, ens.predict(oof)),
                                weights={k: round(float(v), 4)
                                         for k, v in (ens.weights or {}).items()})
        else:
            final_log = test_preds[model_names[0]]

        # --- huấn luyện lại trên toàn bộ train để lưu model ---
        for name in model_names:
            m = MODELS[name](cfg).fit(X, y)
            m.save(get_path(cfg, "models") / f"{name}.joblib")
            try:
                imp = m.feature_importance(X.columns)
                imp.head(50).to_csv(tracker.dir / f"importance_{name}.csv")
                Evaluator.plot_feature_importance(
                    imp, title=f"Độ quan trọng đặc trưng: {name}",
                    fig_dir=get_path(cfg, "figures"), fname=f"importance_{name}.png")
            except AttributeError:
                pass

        # --- biểu đồ chẩn đoán trên OOF của model tốt nhất ---
        best = min(results, key=lambda k: results[k]["rmse_oof"])
        Evaluator.plot_pred_vs_actual(y, oof[best], fig_dir=get_path(cfg, "figures"),
                                      fname=f"pred_vs_actual_{best}.png")
        Evaluator.plot_residuals(y, oof[best], fig_dir=get_path(cfg, "figures"),
                                 fname=f"residuals_{best}.png")

        # --- file nộp Kaggle ---
        final = np.expm1(final_log) if d["log_target"] else final_log
        final = np.clip(final, 1000, None)   # giá nhà không thể âm
        out = get_path(cfg, "submission") / "submission.csv"
        sub = pd.DataFrame({d["id_col"]: test_ids, d["target"]: final})
        if len(sub) != len(X_test):
            raise RuntimeError("Số dòng file nộp không khớp số dòng test")
        sub.to_csv(out, index=False)

        tracker.log_metrics(submission_rows=len(sub),
                            price_min=float(final.min()), price_max=float(final.max()))
        tracker.save()
        return out