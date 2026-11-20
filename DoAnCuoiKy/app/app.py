"""Demo Streamlit – Dự báo điện năng thiết bị gia dụng 10 phút tới.

Chạy từ thư mục DoAnCuoiKy/:
    pip install -r requirements.txt
    streamlit run app/app.py

Yêu cầu: đã có models/best_model.joblib (chạy notebooks/09_final_evaluation.ipynb nếu chưa có).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import DATE_COL, TARGET  # noqa: E402
from src.data_loader import load_raw_data  # noqa: E402
from src.evaluate import regression_metrics  # noqa: E402
from src.predict import N_HISTORY, load_model, predict_next, predict_series  # noqa: E402
from src.split import time_split  # noqa: E402

st.set_page_config(page_title="Dự báo điện năng thiết bị gia dụng", page_icon="⚡", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_data():
    df = load_raw_data()
    _, _, test = time_split(df)
    return df, test[DATE_COL].min()


try:
    model, info = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
features = info["features"]
data, test_start = get_data()

st.title("⚡ Dự báo điện năng thiết bị gia dụng – 10 phút tới")
st.caption(
    "Đồ án cuối kỳ môn Máy học · Dữ liệu UCI *Appliances Energy Prediction* · "
    f"Mô hình: {info.get('model', '?')} với {len(features)} đặc trưng (thời gian + tiêu thụ 10–60 phút trước)"
)

tab_day, tab_try, tab_info = st.tabs(["📈 Xem lại một ngày", "🧮 Tự nhập số liệu", "ℹ️ Về mô hình"])

# ---------------------------------------------------------------------------------------------------------------
# Tab 1: chọn một ngày, so sánh thực tế – dự báo – persistence
# ---------------------------------------------------------------------------------------------------------------
with tab_day:
    min_day = data[DATE_COL].min().date() + pd.Timedelta(days=1)
    max_day = data[DATE_COL].max().date() - pd.Timedelta(days=1)
    day = st.date_input(
        "Chọn ngày (từ 07/05/2016 trở đi là tập test – mô hình chưa từng thấy)",
        value=(test_start + pd.Timedelta(days=1)).date(), min_value=min_day, max_value=max_day,
    )
    start = pd.Timestamp(day)
    end = start + pd.Timedelta(days=1)
    window = data[(data[DATE_COL] >= start - pd.Timedelta(minutes=10 * N_HISTORY)) & (data[DATE_COL] < end)]
    pred = predict_series(model, features, window)
    view = pd.DataFrame({
        "Thực tế": window.set_index(DATE_COL)[TARGET],
        "Dự báo (mô hình)": pred,
        "Persistence (= 10 phút trước)": window.set_index(DATE_COL)[TARGET].shift(1),
    }).loc[start:end - pd.Timedelta(minutes=1)]

    if start >= test_start.normalize():
        st.success("Ngày thuộc **tập test** – kết quả phản ánh khả năng dự báo thật.")
    else:
        st.warning("Ngày thuộc giai đoạn **huấn luyện** – mô hình đã thấy dữ liệu này, kết quả sẽ đẹp hơn thực tế.")

    st.line_chart(view, height=380)

    m_model = regression_metrics(view["Thực tế"], view["Dự báo (mô hình)"])
    m_pers = regression_metrics(view["Thực tế"], view["Persistence (= 10 phút trước)"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE mô hình (Wh)", f"{m_model['RMSE']:.1f}",
              delta=f"{m_model['RMSE'] - m_pers['RMSE']:+.1f} so với persistence", delta_color="inverse")
    c2.metric("MAE mô hình (Wh)", f"{m_model['MAE']:.1f}",
              delta=f"{m_model['MAE'] - m_pers['MAE']:+.1f} so với persistence", delta_color="inverse")
    c3.metric("Tổng tiêu thụ thực tế (kWh)", f"{view['Thực tế'].sum() / 1000:.2f}")
    c4.metric("Đỉnh trong ngày (Wh)", f"{view['Thực tế'].max():.0f}",
              help=f"lúc {view['Thực tế'].idxmax():%H:%M}")

# ---------------------------------------------------------------------------------------------------------------
# Tab 2: tự nhập 60 phút lịch sử
# ---------------------------------------------------------------------------------------------------------------
with tab_try:
    st.markdown("Nhập tiêu thụ của **6 mốc 10 phút gần nhất** và thời điểm cần dự báo.")
    c_date, c_time = st.columns(2)
    d = c_date.date_input("Ngày", value=pd.Timestamp("2016-05-10").date(), key="try_day")
    t = c_time.time_input("Giờ cần dự báo", value=pd.Timestamp("18:00").time(), step=600)
    when = pd.Timestamp.combine(d, t)

    defaults = [60, 60, 70, 90, 150, 240]
    cols = st.columns(N_HISTORY)
    recent = []
    for i, c in enumerate(cols):
        minutes = 10 * (N_HISTORY - i)
        recent.append(c.number_input(f"{minutes} phút trước", min_value=0, max_value=1200,
                                     value=defaults[i], step=10, key=f"lag{i}"))

    y_hat = predict_next(model, features, when, recent)
    st.metric(f"Dự báo tiêu thụ lúc {when:%H:%M %d/%m/%Y}", f"{y_hat:.0f} Wh",
              delta=f"{y_hat - recent[-1]:+.0f} Wh so với 10 phút trước")
    hist = pd.Series(list(recent) + [y_hat],
                     index=pd.date_range(end=when, periods=N_HISTORY + 1, freq="10min"), name="Wh")
    st.bar_chart(hist, height=260)
    st.caption("Cột cuối là giá trị dự báo. Thử tăng/giảm các giá trị gần nhất để thấy mô hình phản ứng thế nào "
               "khi một thiết bị vừa được bật hoặc tắt.")

# ---------------------------------------------------------------------------------------------------------------
# Tab 3: thông tin mô hình
# ---------------------------------------------------------------------------------------------------------------
with tab_info:
    tm = info.get("test_metrics", {})
    if tm:
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE trên test", f"{tm['RMSE']:.1f} Wh")
        c2.metric("MAE trên test", f"{tm['MAE']:.1f} Wh")
        c3.metric("R² trên test", f"{tm['R2']:.2f}")
    st.markdown(
        "- **Bài toán:** dự báo điện năng thiết bị (Wh) trong 10 phút tới.\n"
        "- **Đặc trưng:** giờ trong ngày (dạng sin/cos), thứ, cuối tuần, tiêu thụ 10/20/30/60 phút trước, "
        "trung bình 30/60 phút, mức thay đổi gần nhất.\n"
        "- **Huấn luyện:** train + validation (11/01 – 07/05/2016); **test:** 07/05 – 27/05/2016.\n"
        "- Chi tiết thực nghiệm: notebook 07–09 trong thư mục `notebooks/`."
    )
    with st.expander("best_model_info.json"):
        st.json(info)
