"""Load dữ liệu và tiện ích xử lý file."""
from pathlib import Path

import pandas as pd
import yaml

# ROOT là thư mục gốc dự án (2_HousePrices), suy ra từ vị trí file này.
# Nhờ vậy mọi đường dẫn đúng dù bạn chạy từ notebook, từ tests hay từ terminal.
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = ROOT / "config" / "global_config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Gộp hai dict lồng nhau. Giá trị trong override thắng."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Đọc file cấu hình.

    path=None thì dùng config/global_config.yaml. Phải xử lý None tường minh
    vì Pipeline gọi load_config(config_path) với config_path có thể là None.

    Nếu file có khóa `extends: global_config.yaml` thì nó được gộp lên trên
    file gốc. Nhờ vậy mỗi thành viên giữ một file config nhỏ chỉ ghi phần
    mình đổi, thay vì bốn người tranh nhau sửa global_config.yaml.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_name = cfg.pop("extends", None)
    if base_name:
        base = load_config(path.parent / base_name)
        cfg = _deep_merge(base, cfg)
    return cfg


def get_path(cfg: dict, key: str) -> Path:
    """Trả về đường dẫn tuyệt đối theo cfg['paths'][key], tạo thư mục nếu chưa có."""
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = get_path(cfg, "raw")
    train_file = raw / cfg["data"]["train_file"]
    test_file = raw / cfg["data"]["test_file"]
    for f in (train_file, test_file):
        if not f.exists():
            raise FileNotFoundError(
                f"Thiếu {f.name} trong {raw}. Tải train.csv và test.csv từ Kaggle "
                f"rồi đặt vào data/raw/."
            )
    return pd.read_csv(train_file), pd.read_csv(test_file)


def save_df(df: pd.DataFrame, cfg: dict, stage: str, name: str) -> Path:
    """stage: 'interim' | 'processed'"""
    out = get_path(cfg, stage) / name
    df.to_csv(out, index=False)
    return out


def load_df(cfg: dict, stage: str, name: str) -> pd.DataFrame:
    return pd.read_csv(get_path(cfg, stage) / name)
