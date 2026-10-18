"""
House Prices: Advanced Regression Techniques
=============================================
Lab 02 - CRISP-DM pipeline
Sinh vien thuc hien:
  - 3124411174 Vo Tien Nam Tat
  - 3124411175 Pham Anh Minh

Cac buoc: Data Understanding -> Data Preparation -> Modeling -> Evaluation
Dataset: Ames Housing (Kaggle "House Prices - Advanced Regression Techniques")
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# 1. DATA UNDERSTANDING
# ---------------------------------------------------------------------------
train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))

print("Train shape:", train.shape)
print("Test shape:", test.shape)

results = {}
results["train_shape"] = list(train.shape)
results["test_shape"] = list(test.shape)
results["saleprice_describe"] = train["SalePrice"].describe().to_dict()

# Target distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(train["SalePrice"], kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("Phan phoi SalePrice (goc)")
sns.histplot(np.log1p(train["SalePrice"]), kde=True, ax=axes[1], color="#DD8452")
axes[1].set_title("Phan phoi log(1+SalePrice)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "target_distribution.png"), dpi=150)
plt.close()

# Missing values overview (train)
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_pct = (missing / len(train) * 100).round(2)
results["missing_top"] = missing_pct.to_dict()

fig, ax = plt.subplots(figsize=(9, 6))
missing_pct.head(20).plot(kind="barh", ax=ax, color="#55A868")
ax.invert_yaxis()
ax.set_xlabel("% gia tri thieu")
ax.set_title("Top 20 cot co du lieu thieu nhieu nhat (train)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "missing_values.png"), dpi=150)
plt.close()

# Correlation with SalePrice (numeric features)
num_train = train.select_dtypes(include="number")
corr = num_train.corr()["SalePrice"].sort_values(ascending=False)
results["top_corr_with_saleprice"] = corr.head(11).to_dict()

fig, ax = plt.subplots(figsize=(9, 6))
top_corr_features = corr.index[1:11]  # exclude SalePrice itself
sns.heatmap(train[top_corr_features.tolist() + ["SalePrice"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Ma tran tuong quan - Top 10 bien so co tuong quan cao nhat voi SalePrice")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 2. DATA PREPARATION
# ---------------------------------------------------------------------------
train_id = train["Id"]
test_id = test["Id"]
y = train["SalePrice"].copy()

# Drop columns with >70% missing (Alley, PoolQC, Fence, MiscFeature) + Id
cols_to_drop = ["Id", "Alley", "PoolQC", "Fence", "MiscFeature"]
train_feat = train.drop(columns=cols_to_drop + ["SalePrice"])
test_feat = test.drop(columns=cols_to_drop)

# Concat to keep encoding consistent between train & test
all_data = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)

num_cols = all_data.select_dtypes(include="number").columns
cat_cols = all_data.select_dtypes(exclude="number").columns

# Fill numeric NA with mean, categorical NA with mode
for c in num_cols:
    all_data[c] = all_data[c].fillna(all_data[c].mean())
for c in cat_cols:
    all_data[c] = all_data[c].fillna(all_data[c].mode()[0])

results["n_numeric_features"] = len(num_cols)
results["n_categorical_features"] = len(cat_cols)

# One-hot encode categorical columns
all_data_encoded = pd.get_dummies(all_data, columns=cat_cols, drop_first=True)
results["n_features_after_encoding"] = all_data_encoded.shape[1]

# Split back
n_train = train_feat.shape[0]
X = all_data_encoded.iloc[:n_train, :].reset_index(drop=True)
X_kaggle_test = all_data_encoded.iloc[n_train:, :].reset_index(drop=True)

# Log-transform target (reduces skew, standard practice for this dataset)
y_log = np.log1p(y)

# Train / validation split (80/20) for model evaluation
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.2, random_state=RANDOM_STATE
)
y_val = np.expm1(y_val_log)

# ---------------------------------------------------------------------------
# 3. MODELING
# ---------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10, random_state=RANDOM_STATE),
    "Lasso Regression": Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000),
    "Random Forest": RandomForestRegressor(
        n_estimators=400, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
    ),
}

model_metrics = {}
predictions_val = {}

for name, model in models.items():
    model.fit(X_train, y_train_log)
    pred_log = model.predict(X_val)
    pred = np.expm1(pred_log)
    predictions_val[name] = pred

    rmse_log = np.sqrt(mean_squared_error(y_val_log, pred_log))
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    mae = mean_absolute_error(y_val, pred)
    r2 = r2_score(y_val, pred)

    # 5-fold CV RMSE (on log target, full training data)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        model, X, y_log, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1
    )
    cv_rmse = -cv_scores.mean()

    model_metrics[name] = {
        "RMSE_log": round(rmse_log, 4),
        "RMSE_USD": round(rmse, 2),
        "MAE_USD": round(mae, 2),
        "R2": round(r2, 4),
        "CV_RMSE_log_5fold": round(cv_rmse, 4),
    }
    print(name, model_metrics[name])

results["model_metrics"] = model_metrics

# Best model by validation RMSE (log scale)
best_model_name = min(model_metrics, key=lambda k: model_metrics[k]["RMSE_log"])
results["best_model"] = best_model_name
print("Best model:", best_model_name)

# ---------------------------------------------------------------------------
# 4. EVALUATION - plots
# ---------------------------------------------------------------------------
# Actual vs Predicted for best model
best_pred = predictions_val[best_model_name]
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(y_val, best_pred, alpha=0.5, color="#4C72B0", edgecolor="none")
lims = [min(y_val.min(), best_pred.min()), max(y_val.max(), best_pred.max())]
ax.plot(lims, lims, "r--", linewidth=1.5)
ax.set_xlabel("Gia thuc te (SalePrice)")
ax.set_ylabel("Gia du doan")
ax.set_title(f"Thuc te vs Du doan - {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "actual_vs_predicted.png"), dpi=150)
plt.close()

# Model comparison bar chart (RMSE)
fig, ax = plt.subplots(figsize=(8, 5))
names = list(model_metrics.keys())
rmse_vals = [model_metrics[n]["RMSE_log"] for n in names]
bars = ax.bar(names, rmse_vals, color=sns.color_palette("viridis", len(names)))
ax.set_ylabel("RMSE (log scale, tap validation)")
ax.set_title("So sanh RMSE giua cac mo hinh")
plt.xticks(rotation=20, ha="right")
for b, v in zip(bars, rmse_vals):
    ax.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=150)
plt.close()

# Feature importance (best tree-based model, else Random Forest)
importance_model_name = best_model_name if best_model_name in ("Random Forest", "Gradient Boosting") else "Random Forest"
importance_model = models[importance_model_name]
importances = pd.Series(importance_model.feature_importances_, index=X.columns)
top_importances = importances.sort_values(ascending=False).head(15)
results["top_feature_importances"] = top_importances.to_dict()

fig, ax = plt.subplots(figsize=(9, 7))
top_importances.sort_values().plot(kind="barh", ax=ax, color="#C44E52")
ax.set_title(f"Top 15 dac trung quan trong nhat - {importance_model_name}")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. DEPLOYMENT (mo phong) - Predict on Kaggle test set with best model, retrain on full train
# ---------------------------------------------------------------------------
final_model = models[best_model_name]
final_model.fit(X, y_log)
test_pred_log = final_model.predict(X_kaggle_test)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame({"Id": test_id, "SalePrice": test_pred})
submission.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)

# Save all results to JSON for the report writer
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\nDone. Outputs saved to:", OUT_DIR)
