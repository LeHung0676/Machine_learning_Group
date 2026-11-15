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

---
---

# Nhật ký chạy notebook `3.1_eda.ipynb`

**Ngày:** 15/11/2026
**Notebook kiểm tra:** `3.1_eda.ipynb` (EDA phần 1–3: phân phối biến mục tiêu, xu hướng thời gian, tương quan)
**Môi trường:** macOS, Python của Anaconda (`~/anaconda3/bin/python`, Python 3.13, pandas 2.3.3), kernel `python3`, chạy bằng `nbclient`
**Code trên repo lúc kiểm tra:** commit `580d6f8`

---

## Lần thử 1: chạy nguyên bản (13:55). LỖI

- **Hiện tượng:** dừng ngay ô code đầu tiên (`df = load_raw_data()`):
  ```
  FileNotFoundError: Không tìm thấy .../DoAnCuoiKy/data/raw/energydata_complete.csv.
  ```
- **Nguyên nhân:** đúng lỗi đường dẫn dữ liệu đã ghi ở nhật ký ngày 14/11, **vẫn chưa được sửa**
  (`data/raw/` chỉ có file `keep`, dữ liệu vẫn nằm ở `data/energydata_complete.csv`).
- **Khắc phục tạm thời:** trỏ `load_raw_data()` tới `data/energydata_complete.csv`, chỉ trong kernel lúc chạy,
  không ghi vào code.

## Lần thử 2: chạy sau khi trỏ lại đường dẫn (13:55). LỖI MỚI, NGHIÊM TRỌNG

- **Hiện tượng:** Phần 1 chạy xong (2 hình). Dừng ở ô `from src.features import add_time_features` (đầu Phần 2):
  ```
  File .../DoAnCuoiKy/src/features.py:10
      >>>>>>> 4d1608ea44602e52ed03a5da141d94c452ca596c
              ^
  SyntaxError: invalid decimal literal
  ```
- **Nguyên nhân:** `src/features.py` trên repo **còn sót dấu xung đột merge** (`<<<<<<< HEAD`, `=======`, `>>>>>>>`)
  ở 4 chỗ (dòng 1, 17, 72, 102). Lỗi đến từ commit `4a578ac` "Resolve merge conflicts" (15/11, 12:30):
  commit này gộp `a8f37ae` và `4d1608e` nhưng chưa thực sự gỡ xung đột.
- **Ảnh hưởng:** **mọi nơi import `src.features` đều hỏng**, gồm notebook này, `test/test_features.py`,
  và mọi code mô hình dùng `make_xy` / `get_feature_columns`.
- **Khắc phục tạm thời:** khi chạy, nạp bản `src/features.py` của commit `a8f37ae` (phía `HEAD` của xung đột).
  Bản này sạch, chạy được, và đầy đủ hơn phía kia (có thêm `get_feature_columns`, `make_xy`).
  Chỉ nạp trong kernel, **không sửa file `src/features.py`** trên repo.
- **Trạng thái:** ⚠️ **CHƯA sửa trong code.** Cần người phụ trách (`features.py`) gỡ xung đột và push lại.

## Lần thử 3: chạy với cả 2 cách khắc phục tạm (13:56). OK

- Tất cả 35 ô (13 ô code) chạy xong, không lỗi, tạo 7 hình.
- **Đối chiếu số liệu với phần nhận xét trong notebook:**

  | Nhận xét viết trong notebook | Kết quả thực tế | Khớp |
  |---|---|---|
  | Phần 1–2 (skewness, IQR, lights, theo giờ, theo thứ, theo tuần) | giống hệt lần kiểm tra 14/11 | ✓ |
  | \|Pearson\| lớn nhất ≈ 0,26 (`hour_sin`) | −0.260 (`hour_sin`) | ✓ |
  | Spearman: `hour_sin` −0,46; `T2` 0,31; `lights` 0,30 | −0.461; 0.305; 0.302 | ✓ |
  | Hầu hết cảm biến có \|r\| < 0,15 | 23/24 cảm biến < 0,15 (riêng `RH_out` = 0.152) | ✓ |
  | `rv1` gần như bằng 0 | Pearson −0.011, Spearman −0.009 | ✓ |
  | Thứ tự: thời gian → `lights`, `T2`, `RH_out`, `T1`, `T6` | thực tế: `hour_sin`, `nsm`, **`T2`, `lights`**, `RH_out`, **`hour_cos`**, `T1`, `T6` | ⚠️ lệch nhẹ |

  Ghi chú dòng cuối: `T2` (0.305) đứng **trên** `lights` (0.302), và `hour_cos` (đặc trưng thời gian) nằm sau `RH_out`.
  Kết luận chung của nhận xét không đổi, chỉ nên sửa lại thứ tự cho chính xác.
- Đã xem hình mới `02_corr_with_target.png`: hiển thị đúng 15 đặc trưng, tiếng Việt không lỗi font.

## Lần thử 4: chạy lặp 10 lần (13:56:49–13:57:27). OK

Mỗi lần chạy lại từ đầu với kernel mới (cùng 2 cách khắc phục tạm ở trên), lưu thành một file riêng, rồi so
**toàn bộ output chữ/bảng** và **hash của từng hình** với lần chạy 1.

| Lần | File | Thời điểm | Thời gian chạy | Lỗi | Hình | Giống lần 1 |
|---|---|---|---|---|---|---|
| 1 | `3.1_eda_run01.ipynb` | 13:56:49 | 4.4 s | 0 | 7 | ✓ |
| 2 | `3.1_eda_run02.ipynb` | 13:56:53 | 4.0 s | 0 | 7 | ✓ |
| 3 | `3.1_eda_run03.ipynb` | 13:56:57 | 3.9 s | 0 | 7 | ✓ |
| 4 | `3.1_eda_run04.ipynb` | 13:57:01 | 4.0 s | 0 | 7 | ✓ |
| 5 | `3.1_eda_run05.ipynb` | 13:57:06 | 4.0 s | 0 | 7 | ✓ |
| 6 | `3.1_eda_run06.ipynb` | 13:57:10 | 3.9 s | 0 | 7 | ✓ |
| 7 | `3.1_eda_run07.ipynb` | 13:57:14 | 4.1 s | 0 | 7 | ✓ |
| 8 | `3.1_eda_run08.ipynb` | 13:57:19 | 4.0 s | 0 | 7 | ✓ |
| 9 | `3.1_eda_run09.ipynb` | 13:57:23 | 3.9 s | 0 | 7 | ✓ |
| 10 | `3.1_eda_run10.ipynb` | 13:57:27 | 3.7 s | 0 | 7 | ✓ |

**Vì sao 10 lần giống hệt nhau:** notebook chỉ đọc dữ liệu, tính thống kê/tương quan và vẽ hình,
**không có bước ngẫu nhiên**. Kết quả trùng nhau là đúng kỳ vọng, notebook ổn định và tái lập được.
6 hình `02_*.png` cũ sinh lại **giống từng byte** với bản đã có trên repo.

**Chốt:** output của lần 10 được ghi vào file gốc `3.1_eda.ipynb`.
Hình mới `02_corr_with_target.png` được lưu vào `reports/figures/`.

---

## Việc còn tồn đọng (sau lần kiểm tra 15/11)

- [ ] 🔴 **Gỡ xung đột merge trong `src/features.py`** (xem Lần thử 2). Đang làm hỏng mọi chỗ import file này.
- [ ] **Sửa lỗi đường dẫn dữ liệu** (tồn từ 14/11). Sau khi sửa cả 2 lỗi, chạy lại *Run All* để xác nhận
  notebook chạy được mà không cần khắc phục tạm.
- [ ] Tiêu đề trong notebook ghi **"02 – Khám phá dữ liệu (EDA)"** nhưng tên file là `3.1_eda.ipynb`. Hình cũng lưu với tiền tố `02_`.
  Nên thống nhất cách đánh số.
- [ ] `3.1_eda.ipynb` lặp lại toàn bộ Phần 1–2 của `02_eda .ipynb` (chỉ thêm Phần 3). Nên gộp vào một notebook.
- [ ] Sửa thứ tự trong nhận xét Phần 3.1 (`T2` đứng trước `lights`; có `hour_cos` sau `RH_out`).
- [ ] Ô tóm tắt ghi "Tóm tắt EDA (phần 1–2)" và đặt **trước** Phần 3. Nên bổ sung tóm tắt cho Phần 3.
- [ ] File gốc thiếu `id` cho các ô (nbformat 4.5 cảnh báo `MissingIDFieldWarning`). Bản chốt đã được chuẩn hóa thêm `id`.
