"""Đánh giá model phân loại: metrics, cross-validation sinh OOF, biểu đồ.

OOF (out-of-fold): mỗi dòng train được dự đoán bởi model CHƯA từng thấy dòng
đó. Dùng StratifiedKFold để mỗi fold giữ đúng tỉ lệ 11 lớp, vì dữ liệu lệch
lớp mạnh (lớp đông nhất gấp gần 13 lần lớp ít nhất).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold


class Evaluator:
    @staticmethod
    def metrics(y_true, proba, classes) -> dict:
        classes = np.asarray(classes)
        pred = classes[np.asarray(proba).argmax(axis=1)]
        return {
            "accuracy": float(accuracy_score(y_true, pred)),
            "f1_macro": float(f1_score(y_true, pred, average="macro")),
            "log_loss": float(log_loss(y_true, proba, labels=classes)),
        }

    @staticmethod
    def oof_predict(model_factory, X, y, X_test=None, n_folds: int = 5, seed: int = 42) -> dict:
        """Trả về xác suất OOF (n, n_lớp) và xác suất test bình quân qua các fold."""
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        classes = np.sort(y.unique())
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        oof = np.zeros((len(X), len(classes)))
        test_parts, fold_acc = [], []
        for tr, va in skf.split(X, y):
            m = model_factory().fit(X.iloc[tr], y.iloc[tr])
            if list(m.classes_) != list(classes):
                raise RuntimeError("Fold thiếu lớp, không ghép được xác suất")
            oof[va] = m.predict_proba(X.iloc[va])
            fold_acc.append(accuracy_score(y.iloc[va], classes[oof[va].argmax(axis=1)]))
            if X_test is not None:
                test_parts.append(m.predict_proba(X_test))

        out = Evaluator.metrics(y, oof, classes)
        out.update({
            "oof": oof,
            "test_proba": np.mean(test_parts, axis=0) if test_parts else None,
            "classes": classes,
            "fold_acc": [float(a) for a in fold_acc],
            "acc_mean": float(np.mean(fold_acc)),
            "acc_std": float(np.std(fold_acc)),
        })
        return out

    @staticmethod
    def compare(results: dict) -> pd.DataFrame:
        rows = [{"model": n, "accuracy": round(r["accuracy"], 4),
                 "acc_std": round(r["acc_std"], 4),
                 "f1_macro": round(r["f1_macro"], 4),
                 "log_loss": round(r["log_loss"], 4)} for n, r in results.items()]
        return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)

    @staticmethod
    def per_class(y_true, proba, classes) -> pd.DataFrame:
        classes = np.asarray(classes)
        pred = classes[np.asarray(proba).argmax(axis=1)]
        p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=classes, zero_division=0)
        return pd.DataFrame({"class": classes, "precision": p.round(3), "recall": r.round(3),
                             "f1": f.round(3), "support": s})

    @staticmethod
    def confusion(y_true, proba, classes, normalize: bool = True) -> np.ndarray:
        classes = np.asarray(classes)
        pred = classes[np.asarray(proba).argmax(axis=1)]
        return confusion_matrix(y_true, pred, labels=classes, normalize="true" if normalize else None)

    @staticmethod
    def plot_confusion(y_true, proba, classes, fig_dir=None, fname="confusion_matrix.png", title=""):
        import matplotlib.pyplot as plt
        cm = Evaluator.confusion(y_true, proba, classes)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes); ax.set_yticklabels(classes)
        ax.set_xlabel("Lớp dự đoán"); ax.set_ylabel("Lớp thật")
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j] >= 0.005:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] > 0.5 else "black")
        ax.set_title(title or "Ma trận nhầm lẫn (chuẩn hóa theo hàng)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        if fig_dir is not None:
            from pathlib import Path
            d = Path(fig_dir); d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / fname, bbox_inches="tight", dpi=120)
        plt.close(fig)
