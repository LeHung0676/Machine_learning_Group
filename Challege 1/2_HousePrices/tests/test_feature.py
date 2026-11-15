"""Test cho FeatureEngineer (người 2)."""
from src.preprocess.Feature import FeatureEngineer


def test_ma_hoa_thu_bac_thanh_so(cfg, cleaned):
    tr, _ = cleaned
    out = FeatureEngineer(cfg).create(tr)
    assert out["ExterQual"].dtype.kind in "iu"
    assert out.loc[out["Id"] == 1, "ExterQual"].iloc[0] == 4   # Gd
    assert out.loc[out["Id"] == 6, "BsmtQual"].iloc[0] == 0    # None


def test_giu_dung_thu_bac(cfg, cleaned):
    """Gd phải lớn hơn TA, nếu dùng one-hot thì mất thông tin này."""
    tr, _ = cleaned
    out = FeatureEngineer(cfg).create(tr)
    gd = out.loc[out["Id"] == 1, "ExterQual"].iloc[0]
    ta = out.loc[out["Id"] == 2, "ExterQual"].iloc[0]
    assert gd > ta


def test_total_sf(cfg, cleaned):
    tr, _ = cleaned
    out = FeatureEngineer(cfg).create(tr)
    assert out.loc[out["Id"] == 1, "TotalSF"].iloc[0] == 856 + 856 + 854


def test_total_bath(cfg, cleaned):
    tr, _ = cleaned
    out = FeatureEngineer(cfg).create(tr)
    assert out.loc[out["Id"] == 1, "TotalBath"].iloc[0] == 2 + 0.5 + 1


def test_tuoi_khong_bao_gio_am(cfg, cleaned):
    tr, te = cleaned
    fe = FeatureEngineer(cfg)
    for df in (fe.create(tr), fe.create(te)):
        for c in ["HouseAge", "RemodAge", "GarageAge"]:
            assert (df[c] >= 0).all(), c


def test_mssubclass_thanh_phan_loai(cfg, cleaned):
    tr, _ = cleaned
    out = FeatureEngineer(cfg).create(tr)
    from pandas.api.types import is_numeric_dtype
    assert not is_numeric_dtype(out["MSSubClass"])
    assert not is_numeric_dtype(out["MoSold"])


def test_khong_sinh_them_missing(cfg, cleaned):
    """BsmtFinRatio chia cho 0 rất dễ tạo ra NaN."""
    tr, te = cleaned
    fe = FeatureEngineer(cfg)
    assert fe.create(tr).isna().sum().sum() == 0
    assert fe.create(te).isna().sum().sum() == 0


def test_nha_khong_co_tang_ham_thi_ratio_bang_0(cfg, cleaned):
    tr, _ = cleaned
    out = FeatureEngineer(cfg).create(tr)
    assert out.loc[out["Id"] == 6, "BsmtFinRatio"].iloc[0] == 0


def test_cac_cot_co_khong(cfg, cleaned):
    tr, _ = cleaned
    out = FeatureEngineer(cfg).create(tr)
    assert out.loc[out["Id"] == 6, "HasBsmt"].iloc[0] == 0
    assert out.loc[out["Id"] == 1, "HasBsmt"].iloc[0] == 1
    assert out.loc[out["Id"] == 2, "Has2ndFloor"].iloc[0] == 0


def test_select_bo_cot_theo_config(cfg, cleaned):
    tr, _ = cleaned
    fe = FeatureEngineer(cfg)
    out = fe.select(fe.create(tr))
    for c in ["Street", "PoolArea", "MiscVal"]:
        assert c not in out.columns
