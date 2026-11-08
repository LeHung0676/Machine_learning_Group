"""Buoc 3: Baseline, mo hinh chinh, tinh chinh sieu tham so. Danh gia tren VALIDATION.
TEST chi duoc cham DUY NHAT MOT LAN o cuoi cung cho mo hinh da chon."""
import sys, time, json
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/claude/project/src")

from features import load_raw, build_features, feature_set, chrono_split, SEED
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TAB = "/home/claude/project/outputs/tables"
rng = np.random.RandomState(SEED)


def metrics(y_true, y_pred):
    return dict(
        MAE=mean_absolute_error(y_true, y_pred),
        RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        R2=r2_score(y_true, y_pred),
        MAPE=float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    )


# ====================== DU LIEU ======================
df = build_features(load_raw())
FEATS = feature_set(include_lights=True, include_control=True)
TARGET = "Appliances"

tr, va, te = chrono_split(df)
Xtr, ytr = tr[FEATS], tr[TARGET].values
Xva, yva = va[FEATS], va[TARGET].values
Xte, yte = te[FEATS], te[TARGET].values

print("=" * 78)
print("CHIA DU LIEU THEO THU TU THOI GIAN (khong xao tron)")
print("=" * 78)
for name, part in [("TRAIN", tr), ("VAL", va), ("TEST", te)]:
    print(f"{name:<6} n={len(part):>6}  {part.date.min()}  ->  {part.date.max()}"
          f"   y_mean={part[TARGET].mean():7.2f}")
print(f"\nSo dac trung: {len(FEATS)}")
print("Danh sach:", FEATS)

# ====================== BASELINE + MO HINH MAC DINH ======================
tscv = TimeSeriesSplit(n_splits=5)

models = {
    "B0. Dummy (trung binh)": DummyRegressor(strategy="mean"),
    "B1. Dummy (trung vi)": DummyRegressor(strategy="median"),
    "B2. Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0, random_state=SEED))]),
    "B3. Ridge + log(y)": TransformedTargetRegressor(
        regressor=Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0, random_state=SEED))]),
        func=np.log1p, inverse_func=np.expm1),
    "M1. KNN (k=10)": Pipeline([("sc", StandardScaler()),
                                ("m", KNeighborsRegressor(n_neighbors=10, n_jobs=-1))]),
    "M2. Random Forest (mac dinh)": RandomForestRegressor(
        n_estimators=300, random_state=SEED, n_jobs=-1),
    "M3. HistGradientBoosting (mac dinh)": HistGradientBoostingRegressor(random_state=SEED),
}

rows = []
fitted = {}
for name, mdl in models.items():
    t0 = time.time()
    mdl.fit(Xtr, ytr)
    dt = time.time() - t0
    mtr = metrics(ytr, mdl.predict(Xtr))
    mva = metrics(yva, mdl.predict(Xva))
    fitted[name] = mdl
    rows.append(dict(model=name, fit_s=round(dt, 1),
                     **{f"train_{k}": v for k, v in mtr.items()},
                     **{f"val_{k}": v for k, v in mva.items()}))
    print(f"\n{name}  ({dt:.1f}s)")
    print(f"   TRAIN  MAE={mtr['MAE']:7.2f}  RMSE={mtr['RMSE']:7.2f}  R2={mtr['R2']:+.4f}")
    print(f"   VAL    MAE={mva['MAE']:7.2f}  RMSE={mva['RMSE']:7.2f}  R2={mva['R2']:+.4f}")

res = pd.DataFrame(rows).round(4)
res.to_csv(f"{TAB}/01_baseline_default_models.csv", index=False)

# ====================== THAM CHIEU: PERSISTENCE ======================
# Ngoai pham vi bai toan nowcasting (dung y(t-1)) - chi de do luong
# bao nhieu phuong sai co the lay "mien phi" tu tu tuong quan.
y_all = df[TARGET].values
va_idx = va.index.values
pers_pred = y_all[va_idx - 1]
mp = metrics(yva, pers_pred)
print(f"\n[THAM CHIEU - ngoai pham vi] Persistence y(t-1): "
      f"MAE={mp['MAE']:.2f} RMSE={mp['RMSE']:.2f} R2={mp['R2']:+.4f}")
json.dump(mp, open(f"{TAB}/02_persistence_reference.json", "w"), indent=2)

# ====================== TINH CHINH SIEU THAM SO ======================
print("\n" + "=" * 78)
print("TINH CHINH SIEU THAM SO (RandomizedSearchCV + TimeSeriesSplit tren TRAIN)")
print("=" * 78)

searches = {}

rf_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 15, 25, 40],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": [0.3, 0.5, 0.7, "sqrt"],
}
searches["Random Forest"] = (
    RandomForestRegressor(random_state=SEED, n_jobs=-1), rf_grid, 20)

hgb_grid = {
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "max_iter": [300, 600, 900],
    "max_leaf_nodes": [31, 63, 127, 255],
    "min_samples_leaf": [10, 20, 40],
    "l2_regularization": [0.0, 0.1, 1.0],
}
searches["HistGradientBoosting"] = (
    HistGradientBoostingRegressor(random_state=SEED, early_stopping=False), hgb_grid, 25)

best = {}
for name, (est, grid, n_iter) in searches.items():
    t0 = time.time()
    rs = RandomizedSearchCV(est, grid, n_iter=n_iter, cv=tscv,
                            scoring="neg_root_mean_squared_error",
                            random_state=SEED, n_jobs=-1, refit=True, verbose=0)
    rs.fit(Xtr, ytr)
    dt = time.time() - t0
    mtr = metrics(ytr, rs.best_estimator_.predict(Xtr))
    mva = metrics(yva, rs.best_estimator_.predict(Xva))
    best[name] = rs
    print(f"\n{name}  ({dt:.0f}s, {n_iter} cau hinh x 5 folds)")
    print("   Tham so tot nhat:", rs.best_params_)
    print(f"   CV RMSE (TimeSeriesSplit tren train) = {-rs.best_score_:.2f}")
    print(f"   TRAIN  MAE={mtr['MAE']:7.2f}  RMSE={mtr['RMSE']:7.2f}  R2={mtr['R2']:+.4f}")
    print(f"   VAL    MAE={mva['MAE']:7.2f}  RMSE={mva['RMSE']:7.2f}  R2={mva['R2']:+.4f}")
    rows.append(dict(model=f"{name} (da tuning)", fit_s=round(dt, 1),
                     **{f"train_{k}": v for k, v in mtr.items()},
                     **{f"val_{k}": v for k, v in mva.items()}))
    pd.DataFrame(rs.cv_results_).to_csv(
        f"{TAB}/cv_results_{name.replace(' ', '_')}.csv", index=False)

final_tab = pd.DataFrame(rows).round(4)
final_tab.to_csv(f"{TAB}/03_all_models_val.csv", index=False)
print("\n" + "=" * 78)
print("BANG TONG HOP (sap xep theo VAL RMSE)")
print("=" * 78)
print(final_tab.sort_values("val_RMSE")[
    ["model", "train_MAE", "train_RMSE", "train_R2", "val_MAE", "val_RMSE", "val_R2"]
].to_string(index=False))

# Luu mo hinh tot nhat de buoc sau dung lai
import joblib
champ_name = final_tab.sort_values("val_RMSE").iloc[0]["model"]
print(f"\n==> Mo hinh tot nhat tren VAL: {champ_name}")
for n, rs in best.items():
    joblib.dump(rs.best_estimator_, f"/home/claude/project/outputs/model_{n.replace(' ', '_')}.joblib")
json.dump({n: rs.best_params_ for n, rs in best.items()},
          open(f"{TAB}/04_best_params.json", "w"), indent=2, default=str)
print("==> Da luu mo hinh va tham so.")
