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
├── app/                 # Demo ứng dụng (Streamlit)
├── data/
│   ├── raw/             # Dữ liệu gốc (không chỉnh sửa)
│   └── processed/       # Dữ liệu sau tiền xử lý
├── docs/                # Kế hoạch, mô tả bài toán, tài liệu nhóm
├── models/              # Mô hình đã huấn luyện (.joblib)
├── notebooks/           # Notebook EDA và thực nghiệm
├── reports/figures/     # Hình ảnh xuất ra cho báo cáo/slide/poster
├── src/                 # Mã nguồn Python dùng chung
├── requirements.txt
└── README.md
```

## 4. Cài đặt

```bash
git clone <link-repo>
cd <ten-repo>
python -m venv .venv
# Windows: .venv\Scripts\activate    |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Hướng dẫn chạy

> Đang cập nhật theo tiến độ dự án.

```bash
# Kiểm tra nạp dữ liệu
python -m src.data_loader
```

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

