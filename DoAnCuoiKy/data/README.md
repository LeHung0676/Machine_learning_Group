# Dữ liệu

## Nguồn

- **UCI Machine Learning Repository – Appliances Energy Prediction**
  <https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction>
- Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). *Data driven prediction models of energy use of appliances
  in a low-energy house*. Energy and Buildings, 140, 81–97.

## Vị trí file

| Thư mục | Nội dung |
|---|---|
| `raw/energydata_complete.csv` | Dữ liệu gốc, **không chỉnh sửa**. Đường dẫn khai báo ở `src/config.py` (`RAW_DATA_PATH`). |
| `processed/` | Dữ liệu sau tiền xử lý, do code tự sinh (`*.csv` đã được bỏ qua trong `.gitignore`). |

## Mô tả

- 19.735 dòng × 29 cột, mỗi dòng cách nhau **10 phút**, từ 11/01/2016 17:00 đến 27/05/2016 18:00.
- Không có giá trị thiếu, không trùng lặp, không đứt quãng thời gian (kiểm tra bằng `python -m src.data_loader`).

| Cột | Ý nghĩa | Đơn vị |
|---|---|---|
| `date` | Thời điểm đo | `YYYY-MM-DD HH:MM:SS` |
| `Appliances` | **Biến mục tiêu**: điện năng thiết bị gia dụng | Wh |
| `lights` | Điện năng đèn chiếu sáng | Wh |
| `T1`, `RH_1` | Nhiệt độ, độ ẩm khu bếp | °C, % |
| `T2`, `RH_2` | Phòng khách | °C, % |
| `T3`, `RH_3` | Phòng giặt | °C, % |
| `T4`, `RH_4` | Phòng làm việc | °C, % |
| `T5`, `RH_5` | Phòng tắm | °C, % |
| `T6`, `RH_6` | Bên ngoài nhà (phía bắc) | °C, % |
| `T7`, `RH_7` | Phòng ủi đồ | °C, % |
| `T8`, `RH_8` | Phòng con | °C, % |
| `T9`, `RH_9` | Phòng cha mẹ | °C, % |
| `T_out`, `RH_out` | Nhiệt độ, độ ẩm ngoài trời (trạm thời tiết Chièvres) | °C, % |
| `Press_mm_hg` | Áp suất | mm Hg |
| `Windspeed` | Tốc độ gió | m/s |
| `Visibility` | Tầm nhìn | km |
| `Tdewpoint` | Nhiệt độ điểm sương | °C |
| `rv1`, `rv2` | Hai biến **ngẫu nhiên** (giống hệt nhau), dùng làm biến đối chứng | – |

Lưu ý: file CSV gốc lưu số dạng chuỗi có khoảng trắng (vd. `"  60"`); `load_raw_data()` trong `src/data_loader.py`
đã tự ép kiểu số.
