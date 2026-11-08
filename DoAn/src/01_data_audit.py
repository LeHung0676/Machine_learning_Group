"""Buoc 1: Kiem tra toan ven du lieu va thong ke mo ta."""
import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

df = pd.read_csv("/home/claude/project/data/energydata_complete.csv")
print("=" * 70)
print("SHAPE:", df.shape)
print("=" * 70)
print("\n--- DTYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
mv = df.isna().sum()
print("Tong so o thieu:", mv.sum())
print(mv[mv > 0] if mv.sum() > 0 else "Khong co gia tri thieu nao.")

print("\n--- DUPLICATED FULL ROWS ---")
print("So dong trung lap hoan toan:", df.duplicated().sum())

# Parse datetime
df["date"] = pd.to_datetime(df["date"])
print("\n--- TIME RANGE ---")
print("Tu:", df["date"].min(), " den:", df["date"].max())
print("So dong:", len(df))
print("Trung lap timestamp:", df["date"].duplicated().sum())

# Kiem tra tinh lien tuc cua chuoi thoi gian
diffs = df["date"].diff().dropna()
print("\n--- KHOANG CACH GIUA CAC BAN GHI ---")
print(diffs.value_counts().head(10))
expected = pd.Timedelta("10min")
gaps = diffs[diffs != expected]
print("So khoang cach KHAC 10 phut:", len(gaps))
if len(gaps) > 0:
    print(gaps.value_counts().head())

# Kiem tra thu tu thoi gian
print("Du lieu da sap xep tang dan theo thoi gian:", df["date"].is_monotonic_increasing)

span = (df["date"].max() - df["date"].min())
print("Khoang thoi gian bao phu:", span, "=", span.total_seconds() / 86400, "ngay")
print("So ban ghi ly thuyet neu day du:", int(span.total_seconds() / 600) + 1)

print("\n--- THONG KE MO TA (bien muc tieu va lights) ---")
print(df[["Appliances", "lights"]].describe())

print("\n--- PHAN PHOI Appliances ---")
ap = df["Appliances"]
for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 0.999]:
    print(f"  q{q:<6} = {ap.quantile(q):8.1f}")
print(f"  mean  = {ap.mean():.2f}, std = {ap.std():.2f}")
print(f"  skewness = {ap.skew():.3f}, kurtosis = {ap.kurt():.3f}")
print(f"  So gia tri duy nhat: {ap.nunique()}")
print("  Top 10 gia tri pho bien nhat:")
print(ap.value_counts().head(10).to_string())

print("\n--- PHAN PHOI lights ---")
print(df["lights"].value_counts().sort_index().to_string())
print("Ty le lights == 0:", (df["lights"] == 0).mean())

print("\n--- THONG KE MO TA TAT CA BIEN SO ---")
num = df.select_dtypes(include=[np.number])
print(num.describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(3).to_string())

print("\n--- KIEM TRA rv1 / rv2 (bien ngau nhien) ---")
print("Tuong quan rv1 vs rv2:", df["rv1"].corr(df["rv2"]))
print("rv1 == rv2 hoan toan?:", np.allclose(df["rv1"], df["rv2"]))
print("rv1 corr voi Appliances:", df["rv1"].corr(df["Appliances"]))

print("\n--- KIEM TRA GIA TRI BAT THUONG VE VAT LY ---")
temp_cols = [c for c in df.columns if c.startswith("T") and c not in ("Tdewpoint",)] + ["T_out"]
temp_cols = sorted(set(temp_cols))
rh_cols = [c for c in df.columns if c.startswith("RH")]
print("Cot nhiet do:", temp_cols)
print("Cot do am:", rh_cols)
print("\nRH ngoai khoang [0,100]:")
for c in rh_cols:
    bad = ((df[c] < 0) | (df[c] > 100)).sum()
    print(f"  {c}: {bad} gia tri, range=[{df[c].min():.2f}, {df[c].max():.2f}]")
