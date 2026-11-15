"""Load dữ liệu và tiện ích xử lý file."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "global_config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Đọc config. path=None dùng global_config.yaml; hỗ trợ khóa `extends`."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_name = cfg.pop("extends", None)
    if base_name:
        cfg = _deep_merge(load_config(path.parent / base_name), cfg)
    return cfg


def get_path(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = get_path(cfg, "raw")
    files = [raw / cfg["data"]["train_file"], raw / cfg["data"]["test_file"]]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Thiếu {f.name} trong {raw}. Đặt train.csv và test.csv vào data/raw/.")
    return pd.read_csv(files[0]), pd.read_csv(files[1])


def save_df(df: pd.DataFrame, cfg: dict, stage: str, name: str) -> Path:
    out = get_path(cfg, stage) / name
    df.to_csv(out, index=False)
    return out
