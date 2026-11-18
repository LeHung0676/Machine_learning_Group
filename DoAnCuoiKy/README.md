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
├── app/                 # Demo ứng dụng (Streamlit) – chưa triển khai
├── data/
│   ├── raw/             # Dữ liệu gốc energydata_complete.csv (không chỉnh sửa)
│   └── processed/       # Dữ liệu sau tiền xử lý (tự sinh, không đưa lên git)
├── docs/                # Kế hoạch, mô tả bài toán, tài liệu nhóm
├── models/              # Mô hình đã huấn luyện (.joblib)
├── notebooks/           # Notebook EDA và thực nghiệm
│   ├── NHAT_KY_CHAY.md  # Nhật ký kiểm thử notebook
│   ├── archive/         # Các phiên bản EDA cũ (02_eda_v1 … v4) đã gộp vào 02_eda.ipynb
│   └── test_runs/       # Bản lưu các lần chạy kiểm thử (<notebook>[_vN]_runXX.ipynb)
├── reports/figures/     # Hình ảnh xuất ra cho báo cáo/slide/poster
├── src/                 # Mã nguồn Python dùng chung
├── test/                # Kiểm thử tự động (pytest)
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

> Đang cập nhật theo tiến độ dự án. Mọi lệnh chạy từ thư mục `DoAnCuoiKy/`.

```bash
# Kiểm tra nạp dữ liệu
python -m src.data_loader

# Chia train / validation / test theo thời gian
python -m src.split

# Xem các bộ đặc trưng
python -m src.features

# Chạy kiểm thử
python -m pytest test/ -v
```

Notebook (mở bằng Jupyter, chạy *Run All*):

| Notebook | Nội dung |
|---|---|
| `01_data_overview.ipynb` | Tổng quan dữ liệu, kiểm tra chất lượng |
| `02_eda.ipynb` | **EDA chính (bản tổng hợp)**: phân phối mục tiêu, xu hướng thời gian, tương quan, đa cộng tuyến (VIF), ngoại lệ biến đầu vào và kết luận tiền xử lý |
| `03_preprocessing.ipynb` | Pipeline tiền xử lý, kiểm soát rò rỉ dữ liệu, chiến lược thực nghiệm |
| `04_baseline_linear.ipynb` | Baseline, Linear/Ridge/Lasso và phân tích hệ số |
| `06_log_target.ipynb` | Thí nghiệm biến đổi `log1p` biến mục tiêu (Ridge, Random Forest): đánh đổi MAE và RMSE |
| `archive/02_eda_v1_phan1.ipynb` … `archive/02_eda_v4_phan1-3.ipynb` | Các phiên bản EDA trước (v1–v4), giữ để tham khảo; toàn bộ nội dung đã nằm trong `02_eda.ipynb` (v7) |

## 6. Thành viên nhóm

| STT | Họ và tên | MSSV | Vai trò |
|-----|-----------|------|---------|
| 1 | Lê Long Hưng(Nhóm trưởng) | 3124411116| Quản lý repo, tích hợp, báo cáo |
| 2 | Lê Đức Anh| 3124411009| EDA & tiền xử lý |
| 3 | Tất Võ Tiến Nam |3124411174 | Feature engineering, baseline & mô hình tuyến tính |
| 4 | Phạm Anh Minh| 3124411166| Mô hình ensemble, tinh chỉnh, demo |

## 7. Ghi nhận nguồn hỗ trợ

 nhóm ghi nhận các nguồn hỗ trợ đã sử dụng:

- Thư viện: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn.

- Claude lên kế hoạch 
