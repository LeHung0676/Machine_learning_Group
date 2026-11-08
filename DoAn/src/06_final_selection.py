"""Buoc 6: Lua chon mo hinh cuoi cung tren bo dac trung tot nhat.
Ket hop phat hien tu buoc 5 (dac trung HIEU khang troi dat) voi cac ho mo hinh.
Moi lua chon deu dua tren VALIDATION; TEST khong duoc dung o day.
"""
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
sys.path.insert(0, "/home/claude/project/src")

from features import (load_raw, build_features, chrono_split, SEED,
                      INDOOR_T, INDOOR_RH, WEATHER, TIME, LIGHTS, CONTROL)
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "font.size": 10,
                     "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
FIG = "/home/claude/project/outputs/figures"
TAB = "/home/claude/project/outputs/tables"


def metrics(y, p):
    return dict(MAE=mean_absolute_error(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                R2=r2_score(y, p), MAPE=float(np.mean(np.abs((y - p) / y)) * 100))


df = build_features(load_raw())
for i in range(1, 10):
    df[f"dT{i}"] = df[f"T{i}"] - df["T_out"]
    df[f"dRH{i}"] = df[f"RH_{i}"] - df["RH_out"]
DIFF = [f"dT{i}" for i in range(1, 10)] + [f"dRH{i}" for i in range(1, 10)]

SETS = {
    "A. Tuyet doi": LIGHTS + INDOOR_T + INDOOR_RH + WEATHER + TIME + CONTROL,
    "B. Hieu (khang troi dat)": LIGHTS + DIFF + WEATHER + TIME + CONTROL,
}
TARGET = "Appliances"
tr, va, te = chrono_split(df)
tscv = TimeSeriesSplit(n_splits=5)

# --- Tinh chinh alpha cho Ridge bang TimeSeriesSplit tren TRAIN ---
print("=" * 78)
print("TINH CHINH alpha CHO RIDGE (TimeSeriesSplit tren TRAIN)")
print("=" * 78)
ridge_alpha = {}
for sname, cols in SETS.items():
    base = TransformedTargetRegressor(
        regressor=Pipeline([("sc", StandardScaler()), ("m", Ridge(random_state=SEED))]),
        func=np.log1p, inverse_func=np.expm1)
    gs = GridSearchCV(base, {"regressor__m__alpha": [0.01, 0.1, 1, 10, 100, 500, 1000, 5000]},
                      cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(tr[cols], tr[TARGET])
    ridge_alpha[sname] = gs.best_params_["regressor__m__alpha"]
    print(f"  {sname:<28} alpha* = {ridge_alpha[sname]:<8} CV RMSE = {-gs.best_score_:.2f}")

# --- So sanh cac ho mo hinh tren ca hai bo dac trung ---
print("\n" + "=" * 78)
print("SO SANH MO HINH x BO DAC TRUNG (danh gia tren VALIDATION)")
print("=" * 78)

rows = []
best = (None, None, -1e9)
for sname, cols in SETS.items():
    cands = {
        "Ridge + log(y)": TransformedTargetRegressor(
            regressor=Pipeline([("sc", StandardScaler()),
                                ("m", Ridge(alpha=ridge_alpha[sname], random_state=SEED))]),
            func=np.log1p, inverse_func=np.expm1),
        "Random Forest (tuned)": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=8, max_features=0.3,
            max_depth=40, random_state=SEED, n_jobs=-1),
        "Random Forest (chinh quy manh)": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=40, max_features=0.3,
            random_state=SEED, n_jobs=-1),
        "HistGB (chinh quy)": HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, random_state=SEED, early_stopping=False),
        "HistGB + log(y)": TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
                l2_regularization=1.0, random_state=SEED, early_stopping=False),
            func=np.log1p, inverse_func=np.expm1),
    }
    for mname, mdl in cands.items():
        mdl.fit(tr[cols], tr[TARGET])
        rtr = metrics(tr[TARGET].values, mdl.predict(tr[cols]))
        rva = metrics(va[TARGET].values, mdl.predict(va[cols]))
        gap = rtr["R2"] - rva["R2"]
        rows.append(dict(feature_set=sname, model=mname,
                         train_R2=rtr["R2"], val_MAE=rva["MAE"], val_RMSE=rva["RMSE"],
                         val_R2=rva["R2"], val_MAPE=rva["MAPE"], overfit_gap=gap))
        print(f"  {sname:<26} | {mname:<32} train R2={rtr['R2']:+.4f} "
              f"val R2={rva['R2']:+.4f} MAE={rva['MAE']:6.2f} RMSE={rva['RMSE']:6.2f} "
              f"gap={gap:.3f}")
        if rva["R2"] > best[2]:
            best = ((sname, cols), (mname, mdl), rva["R2"])

tab = pd.DataFrame(rows).round(4)
tab.to_csv(f"{TAB}/12_final_model_selection.csv", index=False)

(sname, cols), (mname, champ), bscore = best
print("\n" + "=" * 78)
print(f"MO HINH CUOI CUNG: {mname}  |  bo dac trung: {sname}  |  val R2 = {bscore:.4f}")
print("=" * 78)
joblib.dump({"model": champ, "cols": cols, "name": mname, "fset": sname},
            "/home/claude/project/outputs/champion.joblib")
json.dump({"model": mname, "feature_set": sname, "val_R2": bscore,
           "ridge_alpha": ridge_alpha}, open(f"{TAB}/13_champion.json", "w"),
          indent=2, default=str)

# --- Bieu do so sanh ---
fig, ax = plt.subplots(figsize=(10, 5))
piv = tab.pivot(index="model", columns="feature_set", values="val_R2")
x = np.arange(len(piv)); w = .36
ax.bar(x - w/2, piv.iloc[:, 0], w, label=piv.columns[0], color="#a0aec0")
ax.bar(x + w/2, piv.iloc[:, 1], w, label=piv.columns[1], color="#2b6cb0")
ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=18, ha="right", fontsize=9)
ax.axhline(0, color="k", lw=.9)
ax.set_ylabel("R² trên validation"); ax.legend(fontsize=9)
ax.set_title("H15. So sánh mô hình × bộ đặc trưng (chia theo thứ tự thời gian)",
             fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/h15_model_comparison.png", bbox_inches="tight"); plt.close(fig)

# --- Overfitting gap ---
fig, ax = plt.subplots(figsize=(10, 4.6))
sub = tab[tab["feature_set"] == sname].sort_values("overfit_gap")
x = np.arange(len(sub)); w = .36
ax.bar(x - w/2, sub["train_R2"], w, label="R² train", color="#c05621")
ax.bar(x + w/2, sub["val_R2"], w, label="R² validation", color="#2b6cb0")
ax.set_xticks(x); ax.set_xticklabels(sub["model"], rotation=18, ha="right", fontsize=9)
ax.axhline(0, color="k", lw=.9); ax.legend(fontsize=9)
ax.set_ylabel("R²")
ax.set_title("H16. Khoảng cách train–validation: dấu hiệu overfitting", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/h16_overfit_gap.png", bbox_inches="tight"); plt.close(fig)

print("\nBang day du:")
print(tab.sort_values("val_R2", ascending=False).to_string(index=False))
