"""Pipeline tiền xử lý dùng chung cho các mô hình.

Nguyên tắc chống rò rỉ dữ liệu (data leakage):
- Mọi bước "học" tham số từ dữ liệu (trung bình/độ lệch chuẩn của StandardScaler, danh mục của OneHotEncoder)
  chỉ được `fit` trên tập train, rồi `transform` cho validation/test.
- Đặt tiền xử lý **bên trong** `Pipeline` cùng với mô hình → khi cross-validation, scaler được fit lại trên
  phần train của từng fold, không bao giờ nhìn thấy fold validation.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ["day_of_week"]


def build_preprocessor(feature_cols: list[str], kind: str = "linear") -> ColumnTransformer | str:
    """Tạo bộ tiền xử lý theo loại mô hình.

    - kind="linear": mô hình nhạy với thang đo (Linear, Ridge, Lasso, MLP)
        * StandardScaler cho các biến số → mọi đặc trưng cùng thang, hệ số so sánh được, regularization công bằng
        * OneHotEncoder cho `day_of_week` → tránh áp đặt thứ tự giả "Chủ Nhật (6) > Thứ Hai (0)"
    - kind="tree": mô hình cây (Random Forest, Gradient Boosting) không cần chuẩn hóa vì chỉ so sánh ngưỡng,
      và tách được biến thứ tự → giữ nguyên ("passthrough").
    """
    if kind == "tree":
        return "passthrough"
    if kind != "linear":
        raise ValueError("kind phải là 'linear' hoặc 'tree'")

    cat = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    num = [c for c in feature_cols if c not in cat]
    transformers = [("num", StandardScaler(), num)]
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_pipeline(model, feature_cols: list[str], kind: str = "linear") -> Pipeline:
    """Ghép tiền xử lý + mô hình thành một Pipeline duy nhất."""
    return Pipeline([("prep", build_preprocessor(feature_cols, kind)), ("model", model)])


if __name__ == "__main__":
    from sklearn.linear_model import Ridge

    from src.data_loader import load_raw_data
    from src.features import make_xy
    from src.split import time_split

    train, val, _ = time_split(load_raw_data())
    X_tr, y_tr = make_xy(train)
    X_va, _ = make_xy(val)
    pipe = make_pipeline(Ridge(), list(X_tr.columns), kind="linear").fit(X_tr, y_tr)
    Z = pipe.named_steps["prep"].transform(X_va)
    print("Số cột sau tiền xử lý:", Z.shape[1])
    print("Trung bình T1 sau chuẩn hóa – train ≈ 0, validation khác 0 (đúng, vì scaler chỉ học từ train):")
    print(" train:", round(pipe.named_steps["prep"].transform(X_tr)[:, 0].mean(), 3), "| val:", round(Z[:, 0].mean(), 3))
