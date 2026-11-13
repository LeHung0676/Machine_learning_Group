"""Load dữ liệu và tiện ích xử lý file."""
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = ROOT / "config" / "global_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = get_path(cfg, "raw")
    train = pd.read_csv(raw / cfg["data"]["train_file"])
    test = pd.read_csv(raw / cfg["data"]["test_file"])
    return train, test


def save_df(df: pd.DataFrame, cfg: dict, stage: str, name: str) -> Path:
    """stage: 'interim' | 'processed'"""
    out = get_path(cfg, stage) / name
    df.to_csv(out, index=False)
    return out


def load_df(cfg: dict, stage: str, name: str) -> pd.DataFrame:
    return pd.read_csv(get_path(cfg, stage) / name)
