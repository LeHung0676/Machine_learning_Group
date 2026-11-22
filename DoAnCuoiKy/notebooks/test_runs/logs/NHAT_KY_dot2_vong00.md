# Nhật ký kiểm thử đợt 2 – vòng 00 (chạy thử, không lưu notebook)

**Ngày:** 22/11/2026 · **Bắt đầu:** 15:04:40 · **Môi trường:** Python 3.13.9, scikit-learn 1.9.1, pandas 2.3.3, nbclient (kernel mới cho mỗi notebook)

| Notebook | File | Bắt đầu | Thời gian chạy | Ô lỗi | Cảnh báo | Hình | Giống lần 1 |
|---|---|---|---|---|---|---|---|
| `01_data_overview` | – | 15:04:40 | 2.0 s | 0 | 0 | 1 | – |
| `02_eda` | – | 15:04:42 | 5.6 s | 0 | 0 | 10 | – |
| `03_preprocessing` | – | 15:04:48 | 16.5 s | 0 | 0 | 1 | – |
| `04_baseline_linear` | – | 15:05:04 | 4.6 s | 0 | 0 | 3 | – |
| `05_tree_models` | – | 15:05:09 | 24.1 s | 0 | 0 | 1 | – |
| `06_log_target` | – | 15:05:33 | 9.4 s | 0 | 0 | 1 | – |
| `07_gradient_boosting_mlp` | – | 15:05:42 | 136.9 s | 0 | 0 | 2 | – |
| `08_lag_features` | – | 15:07:59 | 34.8 s | 1 | 0 | 1 | – |
| `09_final_evaluation` | – | 15:08:34 | 14.7 s | 0 | 0 | 3 | – |

## Lỗi

- `08_lag_features`, ô 4: `AttributeError: 'float' object has no attribute 'round'`

**Nguyên nhân:** `...corrwith(...).abs().max()` trả về `float` của Python, kiểu này không có phương thức `.round()`.
**Khắc phục:** đổi thành `round(..., 3)` trong ô 4 của `08_lag_features.ipynb` (kết quả in ra không đổi: 0.26). Kiểm tra lại ô này trước khi chạy vòng 1.
