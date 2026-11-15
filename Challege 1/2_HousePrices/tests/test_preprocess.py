"""Test cho Preprocessor (người 2)."""
from src.preprocess.Preprocess import Preprocessor


def test_train_va_test_cung_bo_cot(cfg, featured):
    """Lệch cột là nguyên nhân số một làm pipeline chết khi dự đoán."""
    tr, te = featured
    pp = Preprocessor(cfg)
    Xtr = pp.fit_transform(tr)
    Xte = pp.transform(te)
    assert list(Xtr.columns) == list(Xte.columns)


def test_khong_con_missing(cfg, featured):
    tr, te = featured
    pp = Preprocessor(cfg)
    Xtr = pp.fit_transform(tr)
    Xte = pp.transform(te)
    assert Xtr.isna().sum().sum() == 0
    assert Xte.isna().sum().sum() == 0


def test_tat_ca_la_so(cfg, featured):
    tr, _ = featured
    Xtr = Preprocessor(cfg).fit_transform(tr)
    assert Xtr.select_dtypes(exclude="number").empty


def test_gia_tri_la_trong_test_khong_sinh_cot_moi(cfg, featured):
    tr, te = featured
    te = te.copy()
    te.loc[te.index[0], "Neighborhood"] = "KhuChuaTungThay"
    pp = Preprocessor(cfg)
    Xtr = pp.fit_transform(tr)
    Xte = pp.transform(te)
    assert list(Xtr.columns) == list(Xte.columns)


def test_scaler_chi_hoc_tu_train(cfg, featured):
    """Nếu fit cả test thì thông tin test rò rỉ vào huấn luyện."""
    tr, te = featured
    pp = Preprocessor(cfg)
    pp.fit_transform(tr)
    center_sau_train = pp.scaler.center_.copy()
    pp.transform(te)
    assert (pp.scaler.center_ == center_sau_train).all()
