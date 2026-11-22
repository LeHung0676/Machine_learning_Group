# Dự đoán điện năng tiêu thụ của thiết bị gia dụng (Appliances Energy Prediction)

Đồ án cuối kỳ môn **Máy học** – 

## 1. Giới thiệu đề tài

Bài toán: dự đoán lượng điện năng tiêu thụ của các thiết bị gia dụng (`Appliances`, đơn vị Wh, đo mỗi 10 phút)
của một ngôi nhà thụ động năng lượng thấp tại Bỉ, dựa trên:

- Nhiệt độ và độ ẩm trong 9 khu vực của ngôi nhà (cảm biến không dây ZigBee),
- Dữ liệu thời tiết từ trạm khí tượng sân bay Chièvres gần nhất,
- Đặc trưng thời gian (giờ trong ngày, ngày trong tuần, ...).

Loại bài toán: **Hồi quy (Regression)** có giám sát.

## 2. Dữ liệu

- Nguồn: UCI Machine Learning Repository – *Appliances Energy Prediction*
  <https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction>
- Bài báo gốc: Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). *Data driven prediction models of energy use
  of appliances in a low-energy house*. Energy and Buildings, 140, 81–97.
- Kích thước: 19.735 mẫu × 29 cột, giai đoạn 11/01/2016 – 27/05/2016.
- Chi tiết: xem [`data/README.md`](data/README.md).

## 3. Cấu trúc thư mục

```
.
├── app/app.py           # Demo Streamlit: dự báo 10 phút tới
├── data/
│   ├── raw/             # Dữ liệu gốc energydata_complete.csv (không chỉnh sửa)
│   └── processed/       # Dữ liệu sau tiền xử lý (tự sinh, không đưa lên git)
├── docs/                # Kế hoạch, mô tả bài toán, tài liệu nhóm
├── models/              # best_model.joblib + best_model_info.json (do notebook 09 sinh)
├── notebooks/           # Notebook EDA và thực nghiệm
│   ├── NHAT_KY_CHAY.md  # Nhật ký kiểm thử notebook
│   ├── archive/         # Các phiên bản EDA cũ (02_eda_v1 … v4) đã gộp vào 02_eda.ipynb
│   └── test_runs/       # Bản lưu các lần chạy kiểm thử (<notebook>[_vN]_runXX.ipynb)
├── reports/
│   ├── figures/         # Hình ảnh xuất ra cho báo cáo/slide/poster
│   └── tables/          # Bảng kết quả .csv/.json của notebook 07–09
├── src/                 # Mã nguồn Python dùng chung
├── tests/               # Kiểm thử tự động (pytest)
├── requirements.txt
└── README.md
```

## 4. Cài đặt

```bash
git clone https://github.com/LeHung0676/Machine_learning_Group.git
cd Machine_learning_Group/DoAnCuoiKy
python -m venv .venv
# Windows: .venv\Scripts\activate    |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Hướng dẫn chạy

> Mọi lệnh chạy từ thư mục `DoAnCuoiKy/`.

```bash
# Kiểm tra nạp dữ liệu
python -m src.data_loader

# Chia train / validation / test theo thời gian
python -m src.split

# Xem các bộ đặc trưng
python -m src.features

# Dự báo thử bằng mô hình đã lưu
python -m src.predict

# Chạy kiểm thử (15 test)
python -m pytest tests/ -v

# Demo Streamlit (cần models/best_model.joblib)
streamlit run app/app.py
```

Chạy notebook **theo thứ tự số**: notebook 08 và 09 đọc siêu tham số HGB do notebook 07 lưu ở
`reports/tables/07_best_hgb_params.json`; notebook 09 lưu mô hình cho demo. Nếu nạp `best_model.joblib` báo lỗi phiên bản
scikit-learn, chạy lại notebook 09 để huấn luyện lại trên máy của bạn.

Notebook (mở bằng Jupyter, chạy *Run All*):

| Notebook | Nội dung |
|---|---|
| `01_data_overview.ipynb` | Tổng quan dữ liệu, kiểm tra chất lượng |
| `02_eda.ipynb` | **EDA chính (bản tổng hợp)**: phân phối mục tiêu, xu hướng thời gian, tương quan, đa cộng tuyến (VIF), ngoại lệ biến đầu vào và kết luận tiền xử lý |
| `03_preprocessing.ipynb` | Pipeline tiền xử lý, kiểm soát rò rỉ dữ liệu, chiến lược thực nghiệm |
| `04_baseline_linear.ipynb` | Baseline, Linear/Ridge/Lasso và phân tích hệ số |
| `05_tree_models.ipynb` | Random Forest mặc định và có điều chuẩn, chẩn đoán overfitting |
| `06_log_target.ipynb` | Thí nghiệm biến đổi `log1p` biến mục tiêu (Ridge, Random Forest): đánh đổi MAE và RMSE |
| `07_gradient_boosting_mlp.ipynb` | Gradient Boosting (tinh chỉnh bằng CV theo thời gian), MLP, so sánh các họ mô hình (RQ2) |
| `08_lag_features.ipynb` | Đặc trưng tiêu thụ 10–60 phút trước, baseline persistence, ablation cảm biến (RQ4) |
| `09_final_evaluation.ipynb` | Đánh giá **một lần** trên tập test, phân tích trôi dạt theo mùa, permutation importance (RQ3), lưu mô hình |
| `archive/02_eda_v1_phan1.ipynb` … `archive/02_eda_v4_phan1-3.ipynb` | Các phiên bản EDA trước (v1–v4), giữ để tham khảo; toàn bộ nội dung đã nằm trong `02_eda.ipynb` (v7) |

## 6. Kết quả chính (tập test 07/05 – 27/05/2016)

| Thiết lập | Mô hình | RMSE (Wh) | MAE (Wh) | R² |
|---|---|---|---|---|
| A – chỉ môi trường + thời gian | Baseline theo giờ | 84.9 | 46.5 | 0.13 |
| | Lasso | 84.2 | 49.0 | 0.14 |
| | Random Forest / HGB | 98.3 / 99.9 | 72.1 / 73.6 | âm |
| B – + tiêu thụ 10–60 phút trước | Persistence (= 10 phút trước) | 66.8 | 26.7 | 0.46 |
| | Lasso | 60.6 | 26.8 | 0.56 |
| | **HGB (mô hình triển khai)** | **58.9** | **25.2** | **0.58** |

- **RQ1:** chỉ dùng cảm biến môi trường thì gần như không hơn việc biết giờ trong ngày; có lịch sử tiêu thụ thì HGB hơn
  persistence 11.9%.
- **RQ2:** mô hình phi tuyến không vượt trội; với thiết lập A, chúng còn sụp đổ trên test do quan hệ cảm biến – điện năng
  thay đổi theo mùa (notebook 09, mục 2.1).
- **RQ3:** giờ trong ngày và nhiệt độ phòng giặt `T3` quan trọng nhất; 10/30 đặc trưng không hơn biến ngẫu nhiên `rv1`.
- **RQ4:** thêm tiêu thụ quá khứ gần giảm RMSE ~30%; khi đó cảm biến môi trường không còn đóng góp.

Bảng đầy đủ: `reports/tables/`. Hình cho báo cáo/slide/poster: `reports/figures/`.

## 7. Thành viên nhóm

| STT | Họ và tên | MSSV | Vai trò |
|-----|-----------|------|---------|
| 1 | Lê Long Hưng(Nhóm trưởng) | 3124411116| Quản lý repo, tích hợp, báo cáo |
| 2 | Lê Đức Anh| 3124411009| EDA & tiền xử lý |
| 3 | Tất Võ Tiến Nam |3124411174 | Feature engineering, baseline & mô hình tuyến tính |
| 4 | Phạm Anh Minh| 3124411166| Mô hình ensemble, tinh chỉnh, demo |

## 8. Ghi nhận nguồn hỗ trợ

Nhóm ghi nhận các nguồn hỗ trợ đã sử dụng:

- Thư viện: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit.

- Claude (Anthropic): lên kế hoạch; hỗ trợ viết notebook 07–09, `src/predict.py`, `app/app.py` và các test `tests/test_lag_features.py`, `tests/test_predict.py`.
