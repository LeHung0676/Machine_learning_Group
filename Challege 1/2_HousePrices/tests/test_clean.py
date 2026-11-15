"""Test cho Cleaner (người 1)."""
from src.preprocess.Clean import Cleaner


def test_khong_con_missing(cleaned):
    tr, te = cleaned
    assert tr.isna().sum().sum() == 0
    assert te.isna().sum().sum() == 0


def test_bo_dung_outlier_cua_train(cfg, train_df, cleaned):
    tr, _ = cleaned
    assert 5 not in tr["Id"].values
    assert len(tr) == len(train_df) - 1


def test_khong_bao_gio_bo_dong_cua_test(test_df, cleaned):
    """Bỏ dòng ở test sẽ làm file nộp Kaggle sai số dòng."""
    _, te = cleaned
    assert len(te) == len(test_df)


def test_na_nghia_la_khong_co(cleaned):
    tr, _ = cleaned
    assert (tr["PoolQC"] == "None").all()
    assert tr.loc[tr["Id"] == 1, "FireplaceQu"].iloc[0] == "None"


def test_cot_so_khong_co_thi_dien_0(cleaned):
    _, te = cleaned
    assert te.loc[te["Id"] == 106, "GarageArea"].iloc[0] == 0


def test_garage_yr_blt_dien_bang_nam_xay_nha(cleaned):
    """Điền 0 sẽ làm tuổi garage thành hơn 2000 năm."""
    tr, _ = cleaned
    row = tr[tr["Id"] == 4].iloc[0]
    assert row["GarageYrBlt"] == row["YearBuilt"] == 1915


def test_sua_garage_yr_blt_sai_trong_du_lieu_goc(cleaned):
    tr, _ = cleaned
    assert tr.loc[tr["Id"] == 6, "GarageYrBlt"].iloc[0] == 1993


def test_bo_cot_utilities(cleaned):
    tr, te = cleaned
    assert "Utilities" not in tr.columns
    assert "Utilities" not in te.columns


def test_functional_mac_dinh_la_typ(cleaned):
    _, te = cleaned
    assert te.loc[te["Id"] == 105, "Functional"].iloc[0] == "Typ"


def test_lot_frontage_theo_median_khu_dan_cu(cleaned):
    """Id 2 thuộc Veenker, nhà còn lại của Veenker có mặt tiền 75."""
    tr, _ = cleaned
    assert tr.loc[tr["Id"] == 2, "LotFrontage"].iloc[0] == 75.0


def test_lot_frontage_du_phong_khi_khu_khong_co_trong_train(cleaned):
    """NoRidge bị loại khỏi train vì là outlier -> dùng median toàn bộ."""
    _, te = cleaned
    assert te.loc[te["Id"] == 105, "LotFrontage"].iloc[0] == 70.0


def test_bo_dong_trung_lap(cfg, train_df, test_df):
    import pandas as pd
    doubled = pd.concat([train_df, train_df.tail(1)], ignore_index=True)
    doubled.loc[doubled.index[-1], "Id"] = 999
    c = Cleaner(cfg)
    assert len(c.drop_duplicates(doubled)) == len(train_df)
    assert c.report_["duplicates_removed"] == 1
