# Bài nộp cho lớp

Thư mục này chứa sản phẩm cuối cùng để nộp, theo quy ước của môn học.

## Cần có

- [ ] `Paper_HousePricing.pdf` — báo cáo dạng bài báo
- [ ] `Slide_HousePricing.pdf` — slide thuyết trình
- [ ] `Poster_HousePricing.pdf` — poster
- [ ] `code/` — notebook và file nộp Kaggle
- [ ] `Challenge1_HousePrices_<Ten>_<MSSV>.zip` — nén toàn bộ

Hãy xác nhận lại với giảng viên xem có cần đủ cả ba file PDF không.

## Nội dung nên có trong báo cáo

Dựa trên những gì nhóm đã làm, các phần này đều đã có số liệu sẵn:

1. **Mô tả dữ liệu** — 1460 dòng train, 1459 dòng test, 79 đặc trưng.
   Lấy bảng và biểu đồ từ `notebooks/global_eda.ipynb`.
2. **Làm sạch dữ liệu** — 19 cột thiếu, phần lớn NA mang nghĩa "không có"
   theo `data_description.txt`. Bỏ 2 ngoại lai. Lấy số từ `docs/`.
3. **Tạo đặc trưng** — mã hóa thứ bậc 20 cột, thêm 28 đặc trưng mới.
4. **Mô hình và đánh giá** — 5-fold OOF, bảng so sánh ở `reports/results_summary.md`.
5. **Kết hợp mô hình** — so sánh Voting, Blending, Stacking.
6. **Kết luận** — điểm Kaggle, hạn chế, hướng cải thiện.

## Biểu đồ dùng được ngay

Trong `reports/figures/`:
- `target_distribution.png`, `correlation_heatmap.png`, `missing_by_column.png`
- `pred_vs_actual_<model>.png`, `residuals_<model>.png`, `importance_<model>.png`
