"""Test cho FeatureEngineer."""
from src.preprocess.Feature import PAIR_SEP, FeatureEngineer


def test_tao_khoa_pair(cfg, cleaned):
    tr, _ = cleaned
    out = FeatureEngineer(cfg).create(tr)
    r = out.iloc[0]
    assert r["pair"] == r["Artist Name"] + PAIR_SEP + r["Track Name"]


def test_dac_trung_ten_bai(cfg, cleaned):
    tr, _ = cleaned
    out = FeatureEngineer(cfg).create(tr)
    feat = out[out["Track Name"].str.contains("feat")]
    assert len(feat) > 0 and (feat["has_feat"] == 1).all()
    assert (out.loc[~out["Track Name"].str.contains("feat"), "has_feat"] == 0).all()
    assert (out["track_len"] == out["Track Name"].str.len()).all()


def test_ti_le_ky_tu_ngoai_ascii(cfg, cleaned):
    tr, _ = cleaned
    out = FeatureEngineer(cfg).create(tr)
    viet = out[out["Track Name"].str.startswith("Bài hát")]
    eng = out[out["Track Name"].str.startswith("Song")]
    assert (viet["non_ascii_ratio"] > 0).all()
    assert (eng["non_ascii_ratio"] == 0).all()


def test_khong_sinh_them_missing(cfg, cleaned):
    tr, te = cleaned
    fe = FeatureEngineer(cfg)
    assert fe.create(tr).isna().sum().sum() == 0
    assert fe.create(te).isna().sum().sum() == 0


def test_tat_dac_trung_ten_bai_bang_config(cfg, cleaned):
    tr, _ = cleaned
    off = dict(cfg, features=dict(cfg["features"], name_features=False))
    out = FeatureEngineer(off).create(tr)
    assert "track_len" not in out.columns and "pair" in out.columns
