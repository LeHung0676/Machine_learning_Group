"""Buoc 2: EDA va truc quan hoa."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})
FIG = "/home/claude/project/outputs/figures"
TAB = "/home/claude/project/outputs/tables"

df = pd.read_csv("/home/claude/project/data/energydata_complete.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---- Dac trung thoi gian phuc vu EDA ----
df["hour"] = df["date"].dt.hour
df["dow"] = df["date"].dt.dayofweek          # 0=Mon
df["is_weekend"] = (df["dow"] >= 5).astype(int)
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.date
df["tod"] = df["hour"] + df["date"].dt.minute / 60

# =====================================================================
# H1. Phan phoi bien muc tieu (goc vs log)
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
axes[0].hist(df["Appliances"], bins=80, color="#2b6cb0", edgecolor="white", linewidth=.3)
axes[0].set_title("Appliances (thang gốc)\nđộ lệch = %.2f" % df["Appliances"].skew())
axes[0].set_xlabel("Wh / 10 phút"); axes[0].set_ylabel("Tần suất")

logy = np.log1p(df["Appliances"])
axes[1].hist(logy, bins=80, color="#276749", edgecolor="white", linewidth=.3)
axes[1].set_title("log1p(Appliances)\nđộ lệch = %.2f" % logy.skew())
axes[1].set_xlabel("log1p(Wh)")

axes[2].boxplot(df["Appliances"], vert=True, widths=.5,
                flierprops=dict(marker=".", markersize=2, alpha=.25, markerfacecolor="#c53030",
                                markeredgecolor="none"))
axes[2].set_title("Biểu đồ hộp")
axes[2].set_ylabel("Wh / 10 phút")
axes[2].set_xticks([])
fig.suptitle("H1. Phân phối biến mục tiêu", y=1.02, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/h1_target_distribution.png", bbox_inches="tight"); plt.close(fig)

# Thong ke outlier theo IQR
q1, q3 = df["Appliances"].quantile([.25, .75])
iqr = q3 - q1
up = q3 + 1.5 * iqr
n_out = (df["Appliances"] > up).sum()
print(f"[H1] IQR upper fence = {up:.1f} Wh -> {n_out} diem ({n_out/len(df)*100:.2f}%) vuot nguong")
print(f"[H1] Nang luong trong cac diem do chiem {df.loc[df['Appliances']>up,'Appliances'].sum()/df['Appliances'].sum()*100:.1f}% tong nang luong")

# =====================================================================
# H2. Chuoi thoi gian: toan bo + zoom 1 tuan
# =====================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 6))
axes[0].plot(df["date"], df["Appliances"], lw=.35, color="#2b6cb0")
axes[0].set_title("Toàn bộ chuỗi (11/01/2016 – 27/05/2016)")
axes[0].set_ylabel("Wh / 10 phút")

wk = df[(df["date"] >= "2016-02-01") & (df["date"] < "2016-02-08")]
axes[1].plot(wk["date"], wk["Appliances"], lw=1, color="#2b6cb0")
axes[1].set_title("Phóng to: 01/02 – 07/02/2016 (một tuần)")
axes[1].set_ylabel("Wh / 10 phút")
fig.suptitle("H2. Chuỗi thời gian tiêu thụ điện", y=1.0, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/h2_timeseries.png", bbox_inches="tight"); plt.close(fig)

# =====================================================================
# H3. Chu ky: theo gio trong ngay, theo thu, ngay thuong vs cuoi tuan
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))

g = df.groupby("hour")["Appliances"]
m, lo, hi = g.mean(), g.quantile(.25), g.quantile(.75)
axes[0].plot(m.index, m.values, "o-", color="#2b6cb0", ms=4)
axes[0].fill_between(m.index, lo, hi, alpha=.2, color="#2b6cb0")
axes[0].set_title("Theo giờ trong ngày"); axes[0].set_xlabel("Giờ"); axes[0].set_ylabel("Appliances (Wh)")
axes[0].set_xticks(range(0, 24, 3))

names = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
g2 = df.groupby("dow")["Appliances"].mean()
cols = ["#2b6cb0"] * 5 + ["#c05621"] * 2
axes[1].bar(range(7), g2.values, color=cols)
axes[1].set_xticks(range(7)); axes[1].set_xticklabels(names)
axes[1].set_title("Theo thứ trong tuần"); axes[1].set_ylabel("Appliances (Wh)")

for wkd, lab, c in [(0, "Ngày thường", "#2b6cb0"), (1, "Cuối tuần", "#c05621")]:
    s = df[df["is_weekend"] == wkd].groupby("hour")["Appliances"].mean()
    axes[2].plot(s.index, s.values, "o-", label=lab, color=c, ms=4)
axes[2].legend(); axes[2].set_title("Ngày thường so với cuối tuần")
axes[2].set_xlabel("Giờ"); axes[2].set_ylabel("Appliances (Wh)"); axes[2].set_xticks(range(0, 24, 3))
fig.suptitle("H3. Cấu trúc chu kỳ của biến mục tiêu", y=1.03, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/h3_cyclical.png", bbox_inches="tight"); plt.close(fig)

print("\n[H3] TB theo gio - thap nhat:", m.idxmin(), "h =", round(m.min(), 1),
      "| cao nhat:", m.idxmax(), "h =", round(m.max(), 1))
print("[H3] TB ngay thuong =", round(df[df.is_weekend == 0].Appliances.mean(), 1),
      "| cuoi tuan =", round(df[df.is_weekend == 1].Appliances.mean(), 1))

# =====================================================================
# H4. Ma tran tuong quan
# =====================================================================
feat = [c for c in df.columns if c not in ("date", "hour", "dow", "is_weekend", "month", "day", "tod")]
corr = df[feat].corr()

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(feat))); ax.set_xticklabels(feat, rotation=90, fontsize=7)
ax.set_yticks(range(len(feat))); ax.set_yticklabels(feat, fontsize=7)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=.75, label="Pearson r")
ax.set_title("H4. Ma trận tương quan Pearson", fontweight="bold", pad=12)
fig.tight_layout(); fig.savefig(f"{FIG}/h4_corr_matrix.png", bbox_inches="tight"); plt.close(fig)

# Tuong quan voi target
ct = corr["Appliances"].drop("Appliances").sort_values(key=abs, ascending=False)
print("\n[H4] Tuong quan voi Appliances (sap theo |r|):")
print(ct.round(4).to_string())
ct.to_frame("pearson_r").to_csv(f"{TAB}/corr_with_target.csv")

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#c53030" if v < 0 else "#2b6cb0" for v in ct.values]
ax.barh(range(len(ct)), ct.values, color=colors)
ax.set_yticks(range(len(ct))); ax.set_yticklabels(ct.index, fontsize=8)
ax.invert_yaxis(); ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("Hệ số tương quan Pearson với Appliances")
ax.set_title("H5. Tương quan tuyến tính với biến mục tiêu", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/h5_corr_with_target.png", bbox_inches="tight"); plt.close(fig)

# Da cong tuyen giua cac cam bien
high = []
for i in range(len(feat)):
    for j in range(i + 1, len(feat)):
        r = corr.iloc[i, j]
        if abs(r) > 0.9:
            high.append((feat[i], feat[j], round(r, 4)))
print("\n[H4] Cap bien co |r| > 0.9 (da cong tuyen):")
for a, b, r in high:
    print(f"   {a:<12} <-> {b:<12} r={r}")

# =====================================================================
# H6. Tu tuong quan (chung minh rui ro random split)
# =====================================================================
y = df["Appliances"].values
lags = [1, 2, 3, 6, 12, 18, 36, 72, 144, 288, 1008]
acf = []
for L in lags:
    acf.append(np.corrcoef(y[:-L], y[L:])[0, 1])
print("\n[H6] Tu tuong quan cua Appliances:")
for L, a in zip(lags, acf):
    print(f"   lag {L:>5} buoc = {L*10/60:>7.2f} gio : r = {a:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
maxlag = 300
ac = [np.corrcoef(y[:-L], y[L:])[0, 1] for L in range(1, maxlag)]
ax.plot(np.arange(1, maxlag) * 10 / 60, ac, color="#2b6cb0", lw=1.2)
ax.axhline(0, color="k", lw=.8)
ax.set_xlabel("Độ trễ (giờ)"); ax.set_ylabel("Tự tương quan")
ax.set_title("H6. Tự tương quan của Appliances\n"
             "r = %.2f ở độ trễ 10 phút → chia ngẫu nhiên sẽ gây rò rỉ dữ liệu" % ac[0],
             fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/h6_autocorrelation.png", bbox_inches="tight"); plt.close(fig)

# =====================================================================
# H7. Troi dat phan phoi theo thoi gian (mua dong -> mua xuan)
# =====================================================================
daily = df.groupby("day").agg(App=("Appliances", "mean"), T_out=("T_out", "mean")).reset_index()
daily["day"] = pd.to_datetime(daily["day"])
fig, ax = plt.subplots(figsize=(12, 3.8))
ax.plot(daily["day"], daily["App"], color="#2b6cb0", lw=1.2, label="Appliances trung bình ngày")
ax.set_ylabel("Appliances trung bình (Wh)", color="#2b6cb0")
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(daily["day"], daily["T_out"], color="#c05621", lw=1.2, label="Nhiệt độ ngoài trời")
ax2.set_ylabel("T_out trung bình (°C)", color="#c05621")
ax.set_title("H7. Trôi dạt theo thời gian: nhiệt độ ngoài trời tăng dần (đông → xuân)",
             fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/h7_drift.png", bbox_inches="tight"); plt.close(fig)

mth = df.groupby("month")["Appliances"].agg(["mean", "std", "count"])
print("\n[H7] Thong ke theo thang:")
print(mth.round(2).to_string())
print("\n[H7] T_out TB theo thang:")
print(df.groupby("month")["T_out"].mean().round(2).to_string())

# =====================================================================
# H8. lights: kiem tra tinh hop le lam dac trung
# =====================================================================
print("\n[H8] lights:")
print("   Ty le bang 0:", round((df.lights == 0).mean(), 4))
print("   Pearson r voi Appliances:", round(df.lights.corr(df.Appliances), 4))
print("   Spearman r voi Appliances:", round(df.lights.corr(df.Appliances, method='spearman'), 4))
print("   TB Appliances khi lights=0:", round(df[df.lights == 0].Appliances.mean(), 1))
print("   TB Appliances khi lights>0:", round(df[df.lights > 0].Appliances.mean(), 1))

# =====================================================================
# H9. rv1/rv2 - bien ngau nhien doi chung
# =====================================================================
print("\n[H9] rv1 vs rv2 identical:", np.allclose(df.rv1, df.rv2))
print("   r(rv1, Appliances) =", round(df.rv1.corr(df.Appliances), 5))

# =====================================================================
# Luu bang thong ke mo ta
# =====================================================================
desc = df[feat].describe().T
desc["missing"] = df[feat].isna().sum()
desc["skew"] = df[feat].skew()
desc.round(3).to_csv(f"{TAB}/descriptive_stats.csv")
print("\n==> Da luu figures va tables.")
