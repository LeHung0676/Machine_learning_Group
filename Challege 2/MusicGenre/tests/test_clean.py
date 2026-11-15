"""Test cho Cleaner."""
import numpy as np

from src.preprocess.Clean import Cleaner


def test_khong_con_missing(cleaned):
    tr, te = cleaned
    assert tr.isna().sum().sum() == 0
    assert te.isna().sum().sum() == 0


def test_khong_bo_dong_cua_test(test_df, cleaned):
    """Bỏ dòng ở test sẽ làm file nộp Kaggle sai số dòng."""
    _, te = cleaned
    assert len(te) == len(test_df)


def test_key_thieu_nghia_la_cung_do(train_df, cleaned):
    tr, _ = cleaned
    was_na = train_df["key"].isna().values
    assert was_na.any()
    assert (tr.loc[was_na, "key"] == 0).all()
    assert (tr.loc[~was_na, "key"] >= 1).all()


def test_instrumentalness_thieu_nghia_la_0_va_co_co(train_df, cleaned):
    tr, _ = cleaned
    was_na = train_df["instrumentalness"].isna().values
    assert (tr.loc[was_na, "instrumentalness"] == 0).all()
    assert (tr.loc[was_na, "instr_missing"] == 1).all()
    assert (tr.loc[~was_na, "instr_missing"] == 0).all()


def test_quy_doi_thoi_luong_ve_mili_giay(train_df, cleaned):
    tr, _ = cleaned
    assert "duration_in min/ms" not in tr.columns
    was_min = (train_df["duration_in min/ms"] < 30).values
    assert was_min.any() and (~was_min).any()
    assert (tr.loc[was_min, "duration_is_min"] == 1).all()
    assert np.allclose(tr.loc[was_min, "duration_ms"],
                       train_df.loc[was_min, "duration_in min/ms"] * 60_000)
    assert np.allclose(tr.loc[~was_min, "duration_ms"],
                       train_df.loc[~was_min, "duration_in min/ms"])
    assert tr["duration_ms"].min() > 30_000     # không còn giá trị ở đơn vị phút


def test_popularity_dien_bang_trung_vi_cua_train(cfg, train_df, test_df):
    """Trung vị phải học từ train, không được nhìn test."""
    c = Cleaner(cfg)
    _, te = c.run(train_df, test_df, "Class")
    assert c.popularity_median_ == train_df["Popularity"].median()
    was_na = test_df["Popularity"].isna().values
    assert (te.loc[was_na, "Popularity"] == c.popularity_median_).all()
    assert (te.loc[was_na, "pop_missing"] == 1).all()


def test_chi_bo_dong_trung_hoan_toan(cfg, train_df, test_df):
    import pandas as pd
    doubled = pd.concat([train_df, train_df.tail(3)], ignore_index=True)
    c = Cleaner(cfg)
    tr, _ = c.run(doubled, test_df, "Class")
    assert len(tr) == len(train_df)
    assert c.report_["duplicates_removed"] == 3
