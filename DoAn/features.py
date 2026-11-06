"""Module dung chung: nap du lieu, xay dac trung, chia du lieu theo thoi gian."""
import numpy as np
import pandas as pd

DATA = "/home/claude/project/data/energydata_complete.csv"
SEED = 42

INDOOR_T = [f"T{i}" for i in range(1, 10)]
INDOOR_RH = [f"RH_{i}" for i in range(1, 10)]
WEATHER = ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
TIME = ["NSM", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
LIGHTS = ["lights"]
CONTROL = ["rv1"]          # bien ngau nhien giu lai lam doi chung (control variable)


def load_raw():
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_features(df):
    """Tao dac trung thoi gian. Khong dung lag cua bien muc tieu:
    bai toan duoc dinh nghia la NOWCASTING tu mang cam bien, khong phai du bao chuoi."""
    out = df.copy()
    dt = out["date"]
    # NSM = so giay tinh tu nua dem (dung trong Candanedo et al. 2017)
    out["NSM"] = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
    # Ma hoa tuan hoan: tranh viec 23h va 0h bi coi la cach xa nhau
    out["hour_sin"] = np.sin(2 * np.pi * out["NSM"] / 86400)
    out["hour_cos"] = np.cos(2 * np.pi * out["NSM"] / 86400)
    dow = dt.dt.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (dow >= 5).astype(int)
    # rv2 bi loai: trung lap hoan toan voi rv1 (r = 1.0)
    out = out.drop(columns=["rv2"])
    return out


def feature_set(include_lights=True, include_control=True):
    cols = INDOOR_T + INDOOR_RH + WEATHER + TIME
    if include_lights:
        cols = LIGHTS + cols
    if include_control:
        cols = cols + CONTROL
    return cols


def chrono_split(df, train_frac=0.70, val_frac=0.15):
    """Chia theo thu tu thoi gian: train | val | test khong chong lan."""
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()
