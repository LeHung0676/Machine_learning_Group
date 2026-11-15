"""Model Kernel Ridge với kernel đa thức.

Ridge thường nhưng ánh xạ đặc trưng lên không gian bậc 2, nên bắt được
tương tác giữa các cặp đặc trưng (ví dụ diện tích nhân chất lượng)
mà model tuyến tính thuần không thấy.
"""
from __future__ import annotations

from sklearn.kernel_ridge import KernelRidge

from src.model.base import BaseModel


class KernelRidgeModel(BaseModel):
    name = "kernelridge"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        p = cfg["models"].get("kernelridge", {})
        self.model = KernelRidge(
            alpha=p.get("alpha", 0.5),
            kernel=p.get("kernel", "polynomial"),
            degree=p.get("degree", 2),
            coef0=p.get("coef0", 2.5),
        )

    def feature_importance(self, feature_names=None):
        # KernelRidge làm việc trong không gian kernel, không có hệ số
        # ứng với từng đặc trưng gốc, nên không báo được độ quan trọng.
        raise AttributeError("KernelRidge không cung cấp độ quan trọng đặc trưng")
