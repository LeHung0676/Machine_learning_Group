# Nhật ký chạy notebook EDA

> **Quy ước tên file EDA (đổi tên ngày 18/11/2026).** Các mục nhật ký bên dưới giữ nguyên tên file tại thời điểm
> ghi để đúng lịch sử. Bảng đối chiếu tên cũ → tên mới:
>
> | Phiên bản | Tên cũ | Nội dung | Notebook hiện tại | Bản chạy kiểm thử |
> |---|---|---|---|---|
> | v1 | `01_eda.ipynb` | Phần 1 | `archive/02_eda_v1_phan1.ipynb` | `test_runs/02_eda_v1_runXX` (cũ: `01_eda_runXX`) |
> | v2 | `02_eda .ipynb` (14/11) | Phần 1–2 | `archive/02_eda_v2_phan1-2.ipynb` | `test_runs/02_eda_v2_runXX` (cũ: `02_eda_runXX`) |
> | v3 | `3.1_eda.ipynb` | Phần 1–2, 3.1 | `archive/02_eda_v3_phan1-3.1.ipynb` | `test_runs/02_eda_v3_runXX` (cũ: `3.1_eda_runXX`) |
> | v4 | `02_edact.ipynb` | Phần 1–3 | `archive/02_eda_v4_phan1-3.ipynb` | `test_runs/02_eda_v4_runXX` (cũ: `02_edact_runXX`) |
> | v5 | `02_eda.ipynb` tải lên 18/11 | Phần 1–4.2 | – | `test_runs/02_eda_v5_runXX` (cũ: `02_eda_v2_runXX`) |
> | v6 | `02_eda (1).ipynb` | Phần 1–4.3 | – | `test_runs/02_eda_v6_runXX` (cũ: `02_eda_v3_runXX`) |
> | v7 | bản đã hoàn thiện | Phần 1–4.3 | **`02_eda.ipynb`** | `test_runs/02_eda_v7_runXX` (cũ: `02_eda_v4_runXX`) |

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

---
---

# Nhật ký chạy notebook `02_edact.ipynb`

**Ngày:** 15/11/2026
**Notebook kiểm tra:** `02_edact.ipynb` (EDA phần 1–3; so với `3.1_eda.ipynb` có thêm **3.2 Đa cộng tuyến** và **3.3 VIF**)
**Môi trường:** macOS, Python của Anaconda (`~/anaconda3/bin/python`, Python 3.13, pandas 2.3.3), kernel `python3`, chạy bằng `nbclient`
**Code trên repo lúc kiểm tra:** commit `9b386d2` (đã pull 9 commit mới của nhóm trước khi chạy; các commit này không đụng tới `DoAnCuoiKy/`)

---

## Lần thử 1: chạy nguyên bản (20:15:03). LỖI

- **Hiện tượng:** dừng ở ô code đầu tiên (`df = load_raw_data()`):
  ```
  FileNotFoundError: Không tìm thấy .../DoAnCuoiKy/data/raw/energydata_complete.csv.
  ```
- **Nguyên nhân:** lỗi đường dẫn dữ liệu, tồn từ 14/11, **vẫn chưa sửa**.
- **Khắc phục tạm:** trỏ `load_raw_data()` tới `data/energydata_complete.csv`, chỉ trong kernel, không sửa code.

## Lần thử 2: chạy sau khi trỏ lại đường dẫn (20:15:06). LỖI

- **Hiện tượng:** Phần 1 chạy xong (2 hình), dừng ở `from src.features import add_time_features`:
  ```
  File .../DoAnCuoiKy/src/features.py:10
      >>>>>>> 4d1608ea44602e52ed03a5da141d94c452ca596c
  SyntaxError: invalid decimal literal
  ```
- **Nguyên nhân:** `src/features.py` còn sót dấu xung đột merge từ commit `4a578ac`. Lỗi này đã ghi ở nhật ký `3.1_eda`
  sáng 15/11, **vẫn chưa sửa** sau 9 commit mới của nhóm.
- **Khắc phục tạm:** nạp bản `src/features.py` của commit `a8f37ae`, chỉ trong kernel, không sửa file trên repo.

## Lần thử 3: chạy với cả 2 cách khắc phục tạm (20:15:10). OK

- Tất cả 41 ô (16 ô code) chạy xong, không lỗi, tạo 8 hình.
- **Phần 1 – 3.1:** output giống hệt `3.1_eda.ipynb` (đã đối chiếu ở nhật ký trên), kể cả lệch nhẹ về thứ tự
  `T2`/`lights` trong nhận xét 3.1.
- **Đối chiếu nhận xét Phần 3.2–3.3 với kết quả thực tế:**

  | Nhận xét viết trong notebook | Kết quả thực tế | Khớp |
  |---|---|---|
  | 22 cặp đặc trưng có \|r\| > 0,85 | 22 cặp | ✓ |
  | Chủ yếu nhiệt độ–nhiệt độ và độ ẩm–độ ẩm | 15 cặp nhiệt độ, 7 cặp độ ẩm, 0 cặp khác | ✓ |
  | Mạnh nhất `T6` – `T_out`, r ≈ 0,975 | `T6` – `T_out` = 0.975 | ✓ |
  | `T9` gần trùng `T3`, `T5`, `T7` (r > 0,9) | 0.901; 0.911; 0.945 | ✓ |
  | 14/25 đặc trưng có VIF > 10 | 14 / 25 | ✓ |
  | `T_out` ~147, `Tdewpoint` ~86 | 146.8; 86.1 | ✓ |

- Đã xem hình mới `02_corr_heatmap_inputs.png`: hiển thị đúng ma trận tam giác 25×25, tiếng Việt không lỗi font.

## Lần thử 4: chạy lặp 10 lần (20:15:43–20:16:24). OK

Mỗi lần chạy lại từ đầu với kernel mới (cùng 2 cách khắc phục tạm), lưu thành file riêng, so **toàn bộ output
chữ/bảng** và **hash từng hình** với lần 1.

| Lần | File | Thời điểm | Thời gian chạy | Lỗi | Hình | Giống lần 1 |
|---|---|---|---|---|---|---|
| 1 | `02_edact_run01.ipynb` | 20:15:43 | 4.4 s | 0 | 8 | ✓ |
| 2 | `02_edact_run02.ipynb` | 20:15:48 | 4.4 s | 0 | 8 | ✓ |
| 3 | `02_edact_run03.ipynb` | 20:15:52 | 4.4 s | 0 | 8 | ✓ |
| 4 | `02_edact_run04.ipynb` | 20:15:57 | 4.4 s | 0 | 8 | ✓ |
| 5 | `02_edact_run05.ipynb` | 20:16:01 | 4.3 s | 0 | 8 | ✓ |
| 6 | `02_edact_run06.ipynb` | 20:16:06 | 4.3 s | 0 | 8 | ✓ |
| 7 | `02_edact_run07.ipynb` | 20:16:11 | 4.4 s | 0 | 8 | ✓ |
| 8 | `02_edact_run08.ipynb` | 20:16:15 | 4.3 s | 0 | 8 | ✓ |
| 9 | `02_edact_run09.ipynb` | 20:16:20 | 4.4 s | 0 | 8 | ✓ |
| 10 | `02_edact_run10.ipynb` | 20:16:24 | 4.4 s | 0 | 8 | ✓ |

**Vì sao 10 lần giống hệt nhau:** notebook chỉ đọc dữ liệu, tính thống kê, tương quan, VIF (nghịch đảo ma trận,
có tính xác định) và vẽ hình. **Không có bước ngẫu nhiên**, nên notebook ổn định và tái lập được.
7 hình `02_*.png` cũ sinh lại **giống từng byte** với bản đã có trên repo.

**Chốt:** output của lần 10 được ghi vào file gốc `02_edact.ipynb`.
Hình mới `02_corr_heatmap_inputs.png` được lưu vào `reports/figures/`.

---

## Việc còn tồn đọng (sau lần kiểm tra `02_edact`)

- [ ] 🔴 **Gỡ xung đột merge trong `src/features.py`**: vẫn còn, làm hỏng mọi chỗ import file này.
- [ ] **Sửa lỗi đường dẫn dữ liệu** (`data/raw/` vẫn trống).
- [ ] `02_edact.ipynb` là bản mở rộng của `3.1_eda.ipynb` (Phần 1 → 3.1 giống hệt), và cả hai trùng Phần 1–2 với `02_eda .ipynb`.
  Hiện có **3 notebook EDA chồng nội dung**. Nên giữ `02_edact` làm bản chính và bỏ hoặc gộp hai bản kia.
- [ ] Tiêu đề trong notebook vẫn ghi "02 – Khám phá dữ liệu (EDA)". Ô "Tóm tắt EDA (phần 1–2)" đặt trước Phần 3, chưa có tóm tắt cho Phần 3.
- [ ] Bảng top-10 cặp tương quan (ô 3.2) có tên cột `level_0`, `level_1`. Nên đặt tên rõ, ví dụ `feature_1`, `feature_2`.
- [ ] Hình heatmap 3.2 bị đường lưới của style `whitegrid` vẽ đè lên các ô (chỉ ảnh hưởng thẩm mỹ).
- [ ] Bảng VIF chỉ hiển thị 12 dòng đầu trong khi nhận xét nói 14 đặc trưng VIF > 10. Nên hiện `head(14)` để người đọc tự kiểm.

---
---

# Sửa lỗi và dọn dẹp thư mục `DoAnCuoiKy` (15/11/2026)

Kiểm tra toàn bộ thư mục, phát hiện và sửa các lỗi sau:

| # | Vấn đề | Cách sửa | Trạng thái |
|---|---|---|---|
| 1 | `src/features.py` còn dấu xung đột merge (commit `4a578ac`) → `SyntaxError` | Gỡ xung đột, giữ bản đầy đủ (giống `a8f37ae`) | ✅ |
| 2 | `01_eda.ipynb` còn dấu xung đột → file JSON hỏng, không mở được | Khôi phục bản đã chốt 14/11 (`0c1257d`, cùng nội dung, có output) | ✅ |
| 3 | `02_eda .ipynb` bị xóa trắng (30 ô → 0 ô) | Khôi phục từ `9fc3cb3` | ✅ |
| 4 | `README.md` còn dấu xung đột ở mục 7 | Gỡ xung đột, giữ nguyên nội dung ghi nhận | ✅ |
| 5 | Sai đường dẫn dữ liệu (tồn từ 14/11) | Chuyển `energydata_complete.csv` vào `data/raw/` | ✅ |
| 6 | `pytest` không chạy được; docstring ghi `tests/` | Nhờ #1 test chạy lại; sửa thành `test/`; thêm `pytest` vào `requirements.txt` | ✅ |
| 7 | Tên `02_eda .ipynb` có dấu cách thừa | Đổi thành `02_eda.ipynb` | ✅ |
| 8 | 40 file `*_runXX.ipynb` nằm lẫn với notebook chính | Chuyển vào `notebooks/test_runs/` | ✅ |
| 9 | `01_data_overview.ipynb` chưa chạy | Chạy, lưu output và hình `01_appliances_timeseries.png` | ✅ |
| 10 | README trỏ tới `data/README.md` không tồn tại; hướng dẫn chạy còn thiếu | Tạo `data/README.md` (mô tả cột); bổ sung lệnh chạy, bảng notebook, thư mục `test/` | ✅ |

**Kiểm tra sau khi sửa (không dùng cách khắc phục tạm nào):**

- `python -m pytest test/ -v`: **3/3 passed**
- `python -m src.data_loader`, `python -m src.split`, `python -m src.features`: chạy đúng
- *Run All* 5 notebook: `01_data_overview` (1 hình), `01_eda` (2), `02_eda` (6), `3.1_eda` (7), `02_edact` (8): **0 lỗi**
- Các hình `02_*.png` sinh lại giống từng byte bản trên repo

**Ghi chú:** các mục nhật ký phía trên giữ nguyên tên file cũ (`02_eda .ipynb`, `*_runXX.ipynb` ở `notebooks/`) để đúng lịch sử.

## Việc còn tồn đọng (cần nhóm quyết định)

- [ ] Gộp các notebook EDA trùng nhau: `02_edact.ipynb` đã chứa toàn bộ nội dung `01_eda`, `02_eda`, `3.1_eda`. Đề xuất giữ `02_edact` làm bản chính.
- [ ] Thống nhất đánh số notebook (tiêu đề `02_edact`/`3.1_eda` vẫn ghi "02"; có 2 file `01_`).
- [ ] Bổ sung tóm tắt Phần 3 trong `02_edact`, đặt tên cột rõ cho bảng cặp tương quan (`level_0`/`level_1`), hiện `head(14)` cho bảng VIF.
- [ ] `app/` chưa có demo Streamlit như README mô tả.

---
---

# Kiểm thử và gộp bản EDA mới (18/11/2026)

**Notebook kiểm tra:** 2 bản EDA mới tải lên
- **v2** = `02_eda.ipynb` (ghi đè bản cũ): Phần 1–4 (thêm **Phần 4. Ngoại lệ ở các biến đầu vào**).
- **v3** = `02_eda (1).ipynb`: giống v2 + mục **4.3 Kết luận cho bước tiền xử lý**.

**Môi trường:** macOS, Anaconda Python 3.13.9, kernel `python3`, nbconvert 7.16.6.

## So sánh nội dung các bản EDA

| Bản | Nội dung | Quan hệ |
|---|---|---|
| `01_eda` | Phần 1–2 | tập con của `02_edact` |
| `3.1_eda` | Phần 1–2, 3.1 | tập con của `02_edact` |
| `02_edact` | Phần 1–3 | tập con của v2 |
| v2 | Phần 1–4 (4.1, 4.2) | tập con của v3 |
| **v3** | Phần 1–4 + 4.3 Kết luận tiền xử lý | **đầy đủ nhất** |

## Chạy lặp 10 lần mỗi bản

Mỗi lần chạy với kernel mới, lưu thành file riêng, so **từng output** (chữ, bảng, hash từng hình) với lần 1.

| Lần | File v2 | Thời điểm | Lỗi | Hình | Giống lần 1 | File v3 | Thời điểm | Lỗi | Hình | Giống lần 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `02_eda_v2_run01.ipynb` | 21:12:45 | 0 | 10 | ✓ | `02_eda_v3_run01.ipynb` | 21:13:43 | 0 | 10 | ✓ |
| 2 | `02_eda_v2_run02.ipynb` | 21:12:52 | 0 | 10 | ✓ | `02_eda_v3_run02.ipynb` | 21:13:49 | 0 | 10 | ✓ |
| 3 | `02_eda_v2_run03.ipynb` | 21:12:58 | 0 | 10 | ✓ | `02_eda_v3_run03.ipynb` | 21:13:54 | 0 | 10 | ✓ |
| 4 | `02_eda_v2_run04.ipynb` | 21:13:03 | 0 | 10 | ✓ | `02_eda_v3_run04.ipynb` | 21:14:00 | 0 | 10 | ✓ |
| 5 | `02_eda_v2_run05.ipynb` | 21:13:09 | 0 | 10 | ✓ | `02_eda_v3_run05.ipynb` | 21:14:05 | 0 | 10 | ✓ |
| 6 | `02_eda_v2_run06.ipynb` | 21:13:14 | 0 | 10 | ✓ | `02_eda_v3_run06.ipynb` | 21:14:10 | 0 | 10 | ✓ |
| 7 | `02_eda_v2_run07.ipynb` | 21:13:19 | 0 | 10 | ✓ | `02_eda_v3_run07.ipynb` | 21:14:16 | 0 | 10 | ✓ |
| 8 | `02_eda_v2_run08.ipynb` | 21:13:25 | 0 | 10 | ✓ | `02_eda_v3_run08.ipynb` | 21:14:21 | 0 | 10 | ✓ |
| 9 | `02_eda_v2_run09.ipynb` | 21:13:31 | 0 | 10 | ✓ | `02_eda_v3_run09.ipynb` | 21:14:26 | 0 | 10 | ✓ |
| 10 | `02_eda_v2_run10.ipynb` | 21:13:37 | 0 | 10 | ✓ | `02_eda_v3_run10.ipynb` | 21:14:31 | 0 | 10 | ✓ |

- 20/20 lần chạy: **0 lỗi**, mỗi lần 10 hình, output **giống hệt lần 1**. EDA không có bước ngẫu nhiên nên đây là kết quả mong đợi.
- Output 20 ô code của v2 và v3 **giống hệt nhau**: v3 chỉ thêm một ô chữ (4.3).
- 8 hình `02_*.png` đã có trên repo sinh lại **giống từng byte**. Có thêm 2 hình mới của Phần 4:
  `02_input_outliers_boxplot.png`, `02_input_outliers_detail.png`.

## Đối chiếu nhận xét trong notebook với số liệu thực tế

Các số in trực tiếp trong output (skewness, phân vị, IQR, `lights`, cuối tuần/ngày thường, tương quan, VIF,
bảng ngoại lệ) đều khớp nhận xét. Các nhận xét không in số trực tiếp được tính lại riêng:

| Nhận xét | Thực tế | Khớp |
|---|---|---|
| Trung bình ~98, trung vị 60, max 1080; sau log1p ≈ 1.19 | 97.7; 60; 1080; 1.19 | ✓ |
| Đỉnh phụ 11h ~133, đỉnh 18h ~190, đêm ~50 | 133; 190; 48–53 | ✓ |
| T2 ~112, T3–T5 ~87–90 | 111.5; 87.1 / 89.9 / 90.4 | ✓ |
| Trung bình theo tuần 75–122 Wh | 75.5 – 122.0 | ✓ |
| 22 cặp đặc trưng có \|r\| > 0,85 | 22 | ✓ |
| 14/25 đặc trưng VIF > 10 | 14/25 | ✓ |
| Nhiệt độ trong nhà 14,9–29,9 °C; độ ẩm 1–100% | 14.89–29.86; 1.0–100.0 | ✓ |
| `RH_6` trải 1–100%; `Visibility` phần lớn 40 km | 1.0–99.9; mode = 40 | ✓ |
| `rv2` = `rv1` | trùng hoàn toàn | ✓ |
| Số lưu dạng chuỗi có khoảng trắng (4.3) | ghi ở docstring `src/data_loader.py`, không phải trong notebook 01 | ✓ (nguồn ghi "Notebook 01" chưa chính xác) |

## Chốt và sắp xếp file

- **`02_eda.ipynb` = v3 + output lần chạy 10**: đây là bản EDA chính, đầy đủ nhất.
- Bỏ file `02_eda (1).ipynb` (tên có dấu cách và ngoặc); nội dung đã nằm trong `02_eda.ipynb`.
- Chuyển 3 bản cũ trùng nội dung `01_eda`, `3.1_eda`, `02_edact` vào `notebooks/archive/`.
- 20 bản chạy lưu ở `test_runs/02_eda_v2_runXX.ipynb`, `test_runs/02_eda_v3_runXX.ipynb`.
  (Các file `test_runs/02_eda_runXX.ipynb` cũ là bản v1 ngày 14/11.)
- Cập nhật bảng notebook và cây thư mục trong `README.md`.

## Việc còn tồn đọng

- [x] Gộp các notebook EDA trùng nhau → `02_eda.ipynb` là bản chính.
- [ ] Ô "Tóm tắt EDA (phần 1–2)" vẫn đặt trước Phần 3; mục 4.3 đã đóng vai trò kết luận chung.
- [ ] Bảng cặp tương quan (3.2) vẫn có tên cột `level_0`, `level_1`; bảng VIF hiện 12 dòng trong khi nhận xét nói 14.
- [ ] Mục 4.3 ghi nguồn "Notebook 01" cho lỗi định dạng số, thực tế nằm ở `src/data_loader.py`.
- [ ] `app/` chưa có demo Streamlit như README mô tả.

---
---

# Kiểm thử `06_log_target.ipynb` (18/11/2026)

**Nội dung:** thí nghiệm huấn luyện trên `log1p(Appliances)` rồi đổi ngược bằng `expm1` (`TransformedTargetRegressor`),
so sánh Ridge và Random Forest điều chuẩn, có và không có log.
**Môi trường:** macOS, Anaconda Python 3.13.9, kernel `python3`, nbconvert 7.16.6.
**Cách chạy:** chạy tự động bằng nbconvert, mỗi lần một kernel mới, các lần nối tiếp nhau.
Mốc thời gian dưới đây là thời điểm thực tế bắt đầu từng lần chạy (mỗi lần mất khoảng 10 giây).

## Kiểm tra trước khi chạy

- File tải lên **chưa có output** (4 ô code chưa chạy).
- Quét lại **64 notebook đã chạy trước đó** (`notebooks/*.ipynb` và `test_runs/*.ipynb`): **không có ô nào lỗi**, không ô nào bị bỏ sót.
- Chạy thử 1 lần (23:12:25): 0 lỗi.

## Chạy lặp 10 lần

| Lần | File | Bắt đầu | Thời gian chạy | Lỗi | Hình | Giống lần 1 |
|---|---|---|---|---|---|---|
| 1 | `06_log_target_run01.ipynb` | 23:12:52 | 9.9 s | 0 | 1 | ✓ |
| 2 | `06_log_target_run02.ipynb` | 23:13:02 | 9.8 s | 0 | 1 | ✓ |
| 3 | `06_log_target_run03.ipynb` | 23:13:12 | 9.5 s | 0 | 1 | ✓ |
| 4 | `06_log_target_run04.ipynb` | 23:13:21 | 9.6 s | 0 | 1 | ✓ |
| 5 | `06_log_target_run05.ipynb` | 23:13:31 | 9.8 s | 0 | 1 | ✓ |
| 6 | `06_log_target_run06.ipynb` | 23:13:40 | 9.5 s | 0 | 1 | ✓ |
| 7 | `06_log_target_run07.ipynb` | 23:13:50 | 9.7 s | 0 | 1 | ✓ |
| 8 | `06_log_target_run08.ipynb` | 23:14:00 | 9.6 s | 0 | 1 | ✓ |
| 9 | `06_log_target_run09.ipynb` | 23:14:09 | 9.8 s | 0 | 1 | ✓ |
| 10 | `06_log_target_run10.ipynb` | 23:14:19 | 9.6 s | 0 | 1 | ✓ |

**Vì sao 10 lần giống hệt nhau:** Random Forest cố định `random_state = RANDOM_STATE`, chia dữ liệu theo thời gian
(`time_split`, `TimeSeriesSplit`) không xáo trộn, Ridge là nghiệm đóng → không có nguồn ngẫu nhiên nào thay đổi giữa các lần.

## Kết quả (lần 10)

| Mô hình | cv_RMSE | cv_MAE | val_RMSE | val_MAE | val_R² | val_MAPE % |
|---|---|---|---|---|---|---|
| Ridge | 96.663 | 57.619 | 85.597 | 51.050 | 0.139 | 62.761 |
| Ridge + log | 96.363 | 46.596 | 85.336 | 40.193 | 0.144 | 38.744 |
| RF điều chuẩn | **94.042** | 57.127 | **84.675** | 46.631 | **0.157** | 51.897 |
| RF điều chuẩn + log | 95.436 | **46.323** | 85.966 | **37.388** | 0.131 | **32.996** |

MAE trên validation theo mức tiêu thụ thực tế (RF):

| Mức | Không log | Có log | Số mẫu |
|---|---|---|---|
| ≤ 60 Wh | 27.3 | 15.7 | 1544 |
| 60–100 Wh | 37.3 | 19.2 | 826 |
| 100–200 Wh | 42.4 | 31.2 | 360 |
| > 200 Wh | 216.6 | 257.6 | 230 |

## Đối chiếu nhận xét trong notebook

| Nhận xét | Thực tế | Khớp |
|---|---|---|
| Log giảm MAE: RF ~46,7 → ~37,4; Ridge ~51 → ~40 | 46.63 → 37.39; 51.05 → 40.19 | ✓ |
| Log giảm MAPE | RF 51.9 → 33.0; Ridge 62.8 → 38.7 | ✓ |
| RMSE gần như không đổi hoặc tăng nhẹ | Ridge giảm nhẹ (−0,3); RF tăng (+1,4 cv, +1,3 val) | ✓ |
| Có log dự đoán tốt mức nền ≤ 100 Wh | MAE giảm gần một nửa ở 2 nhóm thấp | ✓ |
| Có log dự đoán **thấp** các đỉnh > 200 Wh | Sai số có dấu trung bình: không log −215,3; có log −257,6 Wh (kiểm tra riêng) | ✓ |

- Hình `06_log_error_by_level.png` mới được lưu vào `reports/figures/`; 15 hình cũ không thay đổi.

**Chốt:** output của lần 10 được ghi vào file gốc `06_log_target.ipynb`. Thêm 03, 04, 06 vào bảng notebook trong `README.md`.

## Việc còn tồn đọng

- [ ] Số thứ tự notebook nhảy từ 04 sang 06 (chưa có `05_*.ipynb`).
- [ ] `RF điều chuẩn + log` có val_R² thấp hơn bản không log (0.131 so với 0.157); nên nêu rõ trong báo cáo cùng với nhận xét về RMSE.

---
---

# Hoàn thiện `02_eda.ipynb` (18/11/2026)

Xử lý các mục còn tồn đọng của notebook EDA:

| # | Vấn đề | Cách sửa |
|---|---|---|
| 1 | Bảng top-10 cặp tương quan (3.2) có tên cột `level_0`, `level_1` | Đặt tên `đặc trưng 1`, `đặc trưng 2`, đánh lại chỉ số 0–9 |
| 2 | Bảng VIF chỉ hiện 12 dòng trong khi nhận xét nói 14 đặc trưng VIF > 10 | Hiện đúng các đặc trưng có VIF > 10 (14 dòng, thêm `RH_3` 10.8, `T5` 10.5) |
| 3 | Heatmap tương quan (3.2) bị đường lưới của style `whitegrid` vẽ đè lên ô | Tắt lưới (`ax.grid(False)`); thêm tương tự cho heatmap giờ × thứ (2.3) để thống nhất |
| 4 | Mục 4.3 ghi nguồn "Notebook 01" cho lỗi số lưu dạng chuỗi | Sửa thành `src/data_loader.py` |

## Chạy lặp 10 lần sau khi sửa

Chạy tự động bằng nbconvert, mỗi lần một kernel mới, nối tiếp nhau; mốc thời gian là thời điểm thực tế bắt đầu.

| Lần | File | Bắt đầu | Thời gian chạy | Lỗi | Hình | Giống lần 1 |
|---|---|---|---|---|---|---|
| 1 | `02_eda_v4_run01.ipynb` | 23:17:54 | 5.8 s | 0 | 10 | ✓ |
| 2 | `02_eda_v4_run02.ipynb` | 23:18:00 | 5.3 s | 0 | 10 | ✓ |
| 3 | `02_eda_v4_run03.ipynb` | 23:18:05 | 5.3 s | 0 | 10 | ✓ |
| 4 | `02_eda_v4_run04.ipynb` | 23:18:10 | 5.6 s | 0 | 10 | ✓ |
| 5 | `02_eda_v4_run05.ipynb` | 23:18:16 | 5.4 s | 0 | 10 | ✓ |
| 6 | `02_eda_v4_run06.ipynb` | 23:18:21 | 5.5 s | 0 | 10 | ✓ (*) |
| 7 | `02_eda_v4_run07.ipynb` | 23:18:27 | 5.6 s | 0 | 10 | ✓ |
| 8 | `02_eda_v4_run08.ipynb` | 23:18:32 | 5.2 s | 0 | 10 | ✓ |
| 9 | `02_eda_v4_run09.ipynb` | 23:18:38 | 5.4 s | 0 | 10 | ✓ |
| 10 | `02_eda_v4_run10.ipynb` | 23:18:43 | 5.4 s | 0 | 10 | ✓ |

(*) Lần 6, bộ so sánh tự động báo khác ở ô VIF (3.3). Kiểm tra lại: dòng `print` bị Jupyter ghi thành **2 đoạn**
(`"Số đặc trưng có VIF > 10:"` + `" 14 / 25"`) thay vì 1. Ghép lại thì giống hệt lần 1, bảng VIF cũng giống hệt.
Đây là cách Jupyter chia luồng in ra, **không phải lỗi** và không ảnh hưởng kết quả.

- Mọi con số không đổi so với lần kiểm tra trước (skewness 3.39, IQR 175 Wh / 10.83%, 14/25 VIF > 10, ...).
- Hình: chỉ `02_corr_heatmap_inputs.png` thay đổi (đã bỏ lưới). Heatmap giờ × thứ sinh lại giống từng byte
  (seaborn vốn không vẽ lưới lên hình này). Các hình khác không đổi.

**Chốt:** output của lần 10 được ghi vào `02_eda.ipynb`.

---
---

# Đặt lại tên file EDA và hoàn thiện cấu trúc `02_eda.ipynb` (18/11/2026)

- Đổi tên toàn bộ file EDA theo một quy ước `02_eda_vN` đánh số theo thứ tự thời gian (bảng đối chiếu ở đầu nhật ký):
  - 4 bản cũ trong `archive/` → `02_eda_v1_phan1`, `02_eda_v3_phan1-3.1`, `02_eda_v4_phan1-3`; bổ sung lại
    `02_eda_v2_phan1-2` (bản 14/11, lấy từ commit `a879026`, trước khi bị ghi đè).
  - 70 file chạy kiểm thử EDA trong `test_runs/` → `02_eda_v1_runXX` … `02_eda_v7_runXX` (dùng `git mv`, nội dung không đổi).
- Trong `02_eda.ipynb`: ô "Tóm tắt EDA (phần 1–2)" là mục cấp 2 đứng lẻ giữa Phần 2 và Phần 3 → đổi thành
  **2.5 Tóm tắt Phần 1–2** (chỉ sửa tiêu đề, không đổi code). Phần kết luận chung là mục 4.3.
- Kiểm tra sau khi sửa: chạy lại 1 lần (23:27:11, 5.9 s): 0 lỗi, 10 hình, **output giống hệt** `02_eda_v7_run10`.
  Notebook giữ output của lần chạy v7 số 10.
- Cập nhật tên file trong `README.md`.

## Việc còn tồn đọng (EDA)

- Không còn. Các mục tồn đọng của EDA ở các lần kiểm tra trước đều đã xử lý.


---
---

# Notebook 07–09, mô hình cuối cùng và demo (20/11/2026)

## Thêm mới

| File | Nội dung |
|---|---|
| `notebooks/05_tree_models.ipynb` | Lấy từ nhánh `Minh` (commit `f85bf4f`) vào `main`, nội dung giữ nguyên |
| `notebooks/07_gradient_boosting_mlp.ipynb` | HGB: lưới 18 cấu hình, chọn theo `cv_RMSE`; MLP 4 cấu hình; so sánh các họ mô hình (RQ2) |
| `notebooks/08_lag_features.ipynb` | Đặc trưng lag, kiểm tra không rò rỉ, baseline persistence, so sánh có/không lag, ablation (RQ4) |
| `notebooks/09_final_evaluation.ipynb` | Huấn luyện lại trên train + val, đánh giá một lần trên test, phân tích trôi dạt theo mùa, permutation importance (RQ3), lưu mô hình |
| `src/features.py` | Thêm `add_lag_features`, `LAG_FEATURES`, bộ đặc trưng `full_lag` và `lag_time`; `make_xy` bỏ dòng NaN của lag |
| `src/config.py` | Thêm `TABLES_DIR = reports/tables` |
| `src/predict.py` | Nạp mô hình, tạo đặc trưng từ 60 phút lịch sử, dự báo 10 phút tới |
| `app/app.py` | Demo Streamlit: xem lại một ngày (thực tế – dự báo – persistence), tự nhập số liệu, thông tin mô hình |
| `models/best_model.joblib`, `models/best_model_info.json` | HGB, bộ `lag_time`, 12 đặc trưng, huấn luyện trên train + val |
| `tests/test_lag_features.py` | 3 test: lag chỉ dùng quá khứ, NaN chỉ ở đầu chuỗi, báo lỗi rõ khi thiếu cột lag |
| `tests/test_predict.py` | 3 test: dự báo 1 dòng khớp với cả chuỗi, báo lỗi khi thiếu lịch sử, mô hình đã lưu dự báo hợp lý |

## Phân công đẩy lên GitHub (theo vai trò ở README)

| Lượt | Người | Nội dung |
|---|---|---|
| 1 | Phạm Anh Minh | `src/config.py`, notebook 05 (từ nhánh `Minh`), notebook 07 + hình/bảng 07 |
| 2 | Tất Võ Tiến Nam | `src/features.py` (đặc trưng lag), gộp `test/` → `tests/`, `tests/test_lag_features.py`, notebook 08 + hình/bảng 08 |
| 3 | Lê Đức Anh | notebook 09 + hình/bảng 09, `models/best_model.*` |
| 4 | Phạm Anh Minh | `src/predict.py`, `app/app.py`, `tests/test_predict.py` |
| 5 | Lê Long Hưng | README gốc, `.gitignore` gốc, README đồ án, nhật ký này, README Challenge 2 |

Thứ tự bắt buộc: notebook 08, 09 đọc `reports/tables/07_best_hgb_params.json` (lượt 1); notebook 08, 09 và `src/predict.py`
cần đặc trưng lag trong `src/features.py` (lượt 2); demo cần mô hình do notebook 09 lưu (lượt 3).

## Dọn dẹp

- Gộp 2 thư mục kiểm thử: `test/test_features.py` → `tests/test_features.py`; xóa `test/requirements.txt`
  (trùng `requirements.txt`). Lệnh chạy thống nhất: `python -m pytest tests/ -v`.

## Kiểm tra

- `python -m pytest tests/ -v`: **15/15 passed**.
- Chạy notebook 07 → 08 → 09 theo thứ tự (Python 3.13, scikit-learn 1.9.1), mỗi notebook 1 lần: **0 lỗi**.
  Thời gian: 07 ≈ 2 phút 10 giây, 08 ≈ 47 giây, 09 ≈ 18 giây.
- Kết quả cố định nhờ `random_state = 42` (HGB, Random Forest, MLP, permutation importance).
- `app/app.py` chưa chạy được bằng Streamlit thật trong môi trường kiểm tra (không cài được thư viện); logic đã chạy
  thử bằng bộ giả lập: biểu đồ 144 điểm/ngày, không có giá trị thiếu, chỉ số khớp với notebook 09.

## Việc cần nhóm làm

- [ ] Chạy lại notebook 07 → 08 → 09 trên máy nhóm. Nếu nạp `best_model.joblib` báo lỗi phiên bản scikit-learn,
      chạy lại notebook 09 để sinh lại mô hình.
- [ ] Chạy `streamlit run app/app.py`, chụp màn hình cho slide/poster.
- [ ] Đối chiếu cách bài báo gốc chọn tập test (ngẫu nhiên hay theo thời gian) trước khi so sánh R² trong báo cáo
      (notebook 07, mục 3).


---

## Chạy lại notebook 07 → 08 → 09 (24/11/2026, Windows, scikit-learn 1.7.2)

- Bổ sung notebook 09, `models/best_model.*`, hình/bảng 09 từ gói lượt 3 (Lê Đức Anh) vào repo; `pytest`: 15 passed.
- Mô hình đã lưu được tạo bằng scikit-learn 1.9.1 nên chạy lại 07, 08, 09 trên máy này để mô hình khớp phiên bản.
- Sửa 2 lỗi môi trường: notebook 08 (`.max().round(3)` lỗi vì pandas trả về `float`), notebook 09 (`permutation_importance(n_jobs=-1)` lỗi dọn thư mục tạm joblib trên Windows → `n_jobs=1`).
- Kết quả test gần như không đổi (HGB B: RMSE 58.914, MAE 25.257, R² 0.58). Số đặc trưng không hơn `rv1` đổi từ 13 → 8 (A) / 16 (B) do nhiễu của permutation importance.
