"""Fixture dùng chung: dữ liệu nhỏ mô phỏng bộ Music Genre (11 lớp)."""
import numpy as np
import pandas as pd
import pytest

from src.preprocess.Clean import Cleaner
from src.preprocess.Feature import FeatureEngineer
from src.preprocess.Preprocess import Preprocessor

CFG = {
    "project": {"seed": 42},
    "data": {"target": "Class", "id_col": "Id", "n_classes": 11},
    "split": {"n_folds": 3},
    "clean": {"duration_col": "duration_in min/ms", "minute_threshold": 30},
    "features": {"name_features": True, "target_encode": ["Artist Name", "pair"]},
    "models": {"histgb": {"max_iter": 30}, "logreg": {"C": 1.0, "max_iter": 500},
               "randomforest": {"n_estimators": 30}, "extratrees": {"n_estimators": 30},
               "text": {"C": 5.0, "min_df": 1}},
    "ensemble": {"method": "blending"},
}


def _make(n_per_class: int, first_id: int, with_target: bool, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(11):
        for i in range(n_per_class):
            rows.append({
                "Artist Name": f"Artist {c}-{i % 4}",          # mỗi lớp có 4 nghệ sĩ riêng
                "Track Name": f"Song {c} {i} (feat. X)" if i % 5 == 0 else f"Bài hát {c} {i}",
                "Popularity": rng.integers(1, 100) if i % 7 else np.nan,
                "danceability": rng.uniform(0.1, 0.9),
                "energy": np.clip(c / 11 + rng.normal(0, 0.05), 0.01, 1),
                "key": rng.integers(1, 12) if i % 6 else np.nan,   # không bao giờ có 0
                "loudness": rng.uniform(-30, 0),
                "mode": rng.integers(0, 2),
                "speechiness": rng.uniform(0.02, 0.5),
                "acousticness": rng.uniform(0, 1),
                "instrumentalness": rng.uniform(1e-6, 0.9) if i % 4 else np.nan,
                "liveness": rng.uniform(0.01, 0.9),
                "valence": rng.uniform(0.02, 0.98),
                "tempo": rng.uniform(60, 200),
                "duration_in min/ms": rng.uniform(2, 6) if c in (0, 3) else rng.uniform(120000, 360000),
                "time_signature": 4,
                "Class": c,
            })
    df = pd.DataFrame(rows)
    df.insert(0, "Id", range(first_id, first_id + len(df)))
    return df if with_target else df.drop(columns=["Class"])


@pytest.fixture
def cfg():
    return CFG


@pytest.fixture
def train_df():
    return _make(12, 1, True, seed=0)          # 132 dòng


@pytest.fixture
def test_df():
    return _make(4, 1001, False, seed=1)       # 44 dòng


@pytest.fixture
def cleaned(cfg, train_df, test_df):
    return Cleaner(cfg).run(train_df, test_df, "Class")


@pytest.fixture
def prepared(cfg, cleaned):
    """Trả về (X, y, X_test, pre) giống Pipeline.prepare()."""
    tr, te = cleaned
    fe = FeatureEngineer(cfg)
    y = tr["Class"].astype(int).reset_index(drop=True)
    X = fe.create(tr.drop(columns=["Class", "Id"]))
    Xt = fe.create(te.drop(columns=["Id"]))
    pre = Preprocessor(cfg)
    return pre.fit_transform(X), y, pre.transform(Xt), pre
