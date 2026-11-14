# Nhật ký chạy notebook EDA

**Ngày:** 14/11/2026
**Notebook kiểm tra:** `01_eda.ipynb`, `02_eda .ipynb`
**Môi trường:** macOS, Python của Anaconda (`~/anaconda3/bin/python`, Python 3.13), kernel `python3`, nbconvert 7.16.6

---

## Lần thử 1: chạy bằng Python hệ thống (~23:20). LỖI

- **Hiện tượng:** `/usr/bin/python3` không chạy được, báo lỗi:
  `You have not agreed to the Xcode license agreements. Please run 'sudo xcodebuild -license'...`
- **Nguyên nhân:** Python mặc định của macOS cần Xcode Command Line Tools, mà máy chưa đồng ý license.
- **Khắc phục:** chuyển sang dùng Python và Jupyter của Anaconda (`~/anaconda3/bin/python`, `~/anaconda3/bin/jupyter`).

## Lần thử 2: chạy nguyên bản 2 notebook (~23:22). LỖI

- **Lệnh:** `jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb` (tương tự với `02_eda .ipynb`)
- **Hiện tượng:** cả 2 notebook dừng ngay **ô code đầu tiên** (`df = load_raw_data()`):
  ```
  FileNotFoundError: Không tìm thấy .../DoAnCuoiKy/data/raw/energydata_complete.csv.
  Tải dữ liệu tại https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction và đặt vào data/raw/.
  ```
- **Nguyên nhân:** `src/config.py` khai báo `RAW_DATA_PATH = DATA_DIR / "raw" / "energydata_complete.csv"`,
  nhưng file dữ liệu trên repo lại nằm ở `data/energydata_complete.csv` (không có thư mục con `raw/`).
- **Khắc phục tạm thời:** khi chạy, trỏ `load_raw_data()` tới `data/energydata_complete.csv`.
  Phần sửa này chỉ nạp vào kernel lúc chạy, **không ghi vào code** của notebook.
- **Trạng thái:** ⚠️ **CHƯA sửa trong code.** Người khác bấm *Run All* vẫn sẽ gặp lỗi này.
  Cần chọn một trong hai cách:
  1. chuyển `energydata_complete.csv` vào `data/raw/`, hoặc
  2. sửa `RAW_DATA_PATH` trong `src/config.py`.

## Lần thử 3: chạy bản sao sau khi trỏ lại đường dẫn (23:24). OK

- Chạy bản sao của 2 notebook (không đụng file gốc).
- `01_eda`: tất cả ô chạy xong. `02_eda`: tất cả ô chạy xong. Không có lỗi.
- **Đối chiếu số liệu với phần nhận xét trong notebook:**

  | Nhận xét viết trong notebook | Kết quả thực tế | Khớp |
  |---|---|---|
  | skewness ≈ 3.4; sau log1p ≈ 1.19 | 3.39; 1.19 | ✓ |
  | Trung bình ~98, trung vị 60, max 1080 | 97.7; 60; 1080 | ✓ |
  | ~10,8% ngoại lệ, ngưỡng IQR 175 Wh | 2138 mẫu (10.83%), 175 Wh | ✓ |
  | lights = 0 ~77%, tương quan ~0.2 | 77.3%; 0.197 | ✓ |
  | Đỉnh 18h ~190, đỉnh phụ 11h ~133, đêm ~50 | 190; 133; 48–58 | ✓ |
  | T2 cao nhất ~112, T3–T5 ~87–90 | T2 = 111.5; T3–T5 = 87.1 / 89.9 / 90.4 | ✓ (T5 nhỉnh hơn 90 một chút) |
  | Cuối tuần ~101, ngày thường ~97 | 100.6; 96.6 | ✓ |
  | Trung bình theo tuần 75–122 Wh | 75.5 – 122.0 | ✓ |

- Đã xem 2 hình (`02_target_distribution`, `02_heatmap_hour_weekday`): tiếng Việt hiển thị đúng, không lỗi font.
  4 hình còn lại chạy không lỗi nhưng chưa xem trực tiếp.

## Lần thử 4: chạy lặp 10 lần mỗi notebook (23:31–23:32). OK

Mỗi lần chạy lại từ đầu với kernel mới, lưu thành một file riêng, rồi so **từng output**
(chữ, bảng, và hash của từng hình) với lần chạy 1.

| Lần | File 01_eda | Thời điểm | Lỗi | Hình | Giống lần 1 |
|---|---|---|---|---|---|
| 1 | `01_eda_run01.ipynb` | 23:31:02 | 0 | 2 | ✓ |
| 2 | `01_eda_run02.ipynb` | 23:31:05 | 0 | 2 | ✓ |
| 3 | `01_eda_run03.ipynb` | 23:31:08 | 0 | 2 | ✓ |
| 4 | `01_eda_run04.ipynb` | 23:31:11 | 0 | 2 | ✓ |
| 5 | `01_eda_run05.ipynb` | 23:31:14 | 0 | 2 | ✓ |
| 6 | `01_eda_run06.ipynb` | 23:31:17 | 0 | 2 | ✓ |
| 7 | `01_eda_run07.ipynb` | 23:31:20 | 0 | 2 | ✓ |
| 8 | `01_eda_run08.ipynb` | 23:31:23 | 0 | 2 | ✓ |
| 9 | `01_eda_run09.ipynb` | 23:31:26 | 0 | 2 | ✓ |
| 10 | `01_eda_run10.ipynb` | 23:31:29 | 0 | 2 | ✓ |

| Lần | File 02_eda | Thời điểm | Lỗi | Hình | Giống lần 1 |
|---|---|---|---|---|---|
| 1 | `02_eda_run01.ipynb` | 23:31:33 | 0 | 6 | ✓ |
| 2 | `02_eda_run02.ipynb` | 23:31:36 | 0 | 6 | ✓ |
| 3 | `02_eda_run03.ipynb` | 23:31:40 | 0 | 6 | ✓ |
| 4 | `02_eda_run04.ipynb` | 23:31:43 | 0 | 6 | ✓ |
| 5 | `02_eda_run05.ipynb` | 23:31:46 | 0 | 6 | ✓ |
| 6 | `02_eda_run06.ipynb` | 23:31:50 | 0 | 6 | ✓ |
| 7 | `02_eda_run07.ipynb` | 23:31:53 | 0 | 6 | ✓ |
| 8 | `02_eda_run08.ipynb` | 23:31:57 | 0 | 6 | ✓ |
| 9 | `02_eda_run09.ipynb` | 23:32:00 | 0 | 6 | ✓ |
| 10 | `02_eda_run10.ipynb` | 23:32:03 | 0 | 6 | ✓ |

**Vì sao 10 lần giống hệt nhau:** hai notebook EDA chỉ đọc dữ liệu và tính thống kê, vẽ hình,
**không có bước ngẫu nhiên nào** (không chia train/test ngẫu nhiên, không huấn luyện mô hình).
Kết quả giống nhau là đúng như mong đợi và cho thấy notebook chạy ổn định, tái lập được.

**Chốt:** output của lần 10 được ghi vào 2 file gốc `01_eda.ipynb` và `02_eda .ipynb`.
Hình được lưu vào `reports/figures/` (6 file `02_*.png`).

---

## Việc còn tồn đọng

- [ ] **Sửa lỗi đường dẫn dữ liệu** (xem Lần thử 2). Sau khi sửa, chạy lại để xác nhận *Run All* chạy được mà không cần trỏ tay.
- [ ] `01_eda.ipynb` trùng nội dung với Phần 1 của `02_eda .ipynb`, và tiêu đề bên trong cũng ghi "02". Nên gộp hoặc bỏ một file.
- [ ] Tên file `02_eda .ipynb` có **dấu cách thừa** trước `.ipynb`. Nên đổi thành `02_eda.ipynb`.
- [ ] Đánh số notebook: đang có 2 file cùng số `01_` (`01_data_overview.ipynb`, `01_eda.ipynb`).
