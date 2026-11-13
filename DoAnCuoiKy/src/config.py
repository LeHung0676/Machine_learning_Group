"""Cấu hình chung: đường dẫn, tên cột, hằng số dùng trong toàn dự án."""
from pathlib import Path

# ---- Đường dẫn ----
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "energydata_complete.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# ---- Cột dữ liệu ----
DATE_COL = "date"
TARGET = "Appliances"

TEMP_COLS = [f"T{i}" for i in range(1, 10)]
HUMIDITY_COLS = [f"RH_{i}" for i in range(1, 10)]
WEATHER_COLS = ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
RANDOM_COLS = ["rv1", "rv2"]
OTHER_COLS = ["lights"]

# ---- Hằng số thực nghiệm ----
RANDOM_STATE = 42


def ensure_dirs() -> None:
    """Tạo các thư mục đầu ra nếu chưa có."""
    for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
