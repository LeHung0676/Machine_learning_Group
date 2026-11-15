# Tổng hợp kết quả — Music Genre Classification

Đánh giá bằng 5-fold phân tầng, dự đoán out-of-fold, `seed = 42`, 14 396 dòng train.
Chỉ số chính: **accuracy**. Kèm F1 macro (vì lệch lớp) và log loss.

> Cần xác nhận chỉ số chấm điểm trên trang Kaggle của lớp. File mẫu nộp nhãn cứng nên
> nhóm tạm lấy accuracy làm chính.

## Đóng góp của từng bước đặc trưng (model HistGradientBoosting)

| Bước | Accuracy | F1 macro | Log loss |
|------|----------|----------|----------|
| A. Số thô, để nguyên NaN | 0.5419 | 0.5788 | 1.2175 |
| B. + xử lý ngữ nghĩa, cờ đơn vị | 0.5439 | 0.5816 | 1.2185 |
| C. + target encoding nghệ sĩ | 0.6296 | 0.6780 | 0.9812 |
| D. + target encoding (nghệ sĩ, tên bài) | 0.6841 | 0.7263 | 0.8717 |
| E. + đặc trưng tên bài | 0.6861 | 0.7308 | 0.8557 |

## Từng model (đủ đặc trưng)

| Ngày | Model | Accuracy | ± fold | F1 macro | Log loss | Kaggle |
|------|-------|----------|--------|----------|----------|--------|
| 2026-11-15 | LightGBM | 0.6895 | 0.0031 | 0.7378 | 1.0265 | |
| 2026-11-15 | HistGradientBoosting | 0.6868 | 0.0059 | 0.7313 | 0.8560 | |
| 2026-11-15 | Random Forest | 0.6772 | 0.0068 | 0.7177 | 0.9345 | |
| 2026-11-15 | Extra Trees | 0.6755 | 0.0058 | 0.7134 | 0.9330 | |
| 2026-11-15 | Logistic Regression | 0.6600 | 0.0072 | 0.6972 | 0.9352 | |
| 2026-11-15 | Text (TF-IDF) | 0.4451 | 0.0050 | 0.4601 | 1.5790 | |
| | XGBoost / CatBoost | | | | | không có trong `run.models` của config |

## Kết hợp model

| Phương pháp | Accuracy | Log loss | Ghi chú |
|-------------|----------|----------|---------|
| **Stacking** | **0.7021** | 0.8489 | meta model cũng được cross-validate |
| Voting | 0.6999 | 0.8907 | trung bình đều 6 model |
| Blending | 0.6944 | 0.8344 | trọng số tối ưu log loss |
| LightGBM đơn lẻ | 0.6895 | 1.0265 | model đơn tốt nhất |
| HistGB đơn lẻ | 0.6868 | 0.8560 | mốc |

Trọng số Blending: `histgb 0.500, lightgbm 0.248, logreg 0.142, extratrees 0.108, text 0.002, randomforest 0.000`.

## Theo từng lớp (Stacking)

| Lớp | Thể loại | Precision | Recall | F1 | Số bài |
|-----|----------|-----------|--------|----|--------|
| 7 | Instrumental | 0.974 | 0.959 | 0.966 | 461 |
| 3 | Bollywood | 0.951 | 0.910 | 0.930 | 322 |
| 0 | Acoustic/Folk | 0.839 | 0.894 | 0.865 | 500 |
| 4 | Country | 0.902 | 0.829 | 0.864 | 310 |
| 5 | HipHop | 0.816 | 0.800 | 0.808 | 1157 |
| 8 | Metal | 0.790 | 0.728 | 0.758 | 1483 |
| 2 | Blues | 0.775 | 0.689 | 0.729 | 1018 |
| 10 | Rock | 0.652 | 0.756 | 0.700 | 3959 |
| 9 | Pop | 0.719 | 0.673 | 0.695 | 2019 |
| 6 | Indie Alt | 0.557 | 0.617 | 0.586 | 2069 |
| 1 | Alt Music | 0.525 | 0.305 | 0.386 | 1098 |

Tên thể loại theo bảng mã của bộ dữ liệu gốc, cần đối chiếu với đề bài.

## Nhận xét

- **Biết nghệ sĩ là ai quan trọng hơn mọi thuộc tính âm thanh.** Hai bước target encoding
  nâng accuracy từ 0.544 lên 0.684, trong khi xử lý ngữ nghĩa chỉ thêm 0.002.
- **Voting hơn Blending về accuracy nhưng kém về log loss.** Blending tối ưu log loss nên
  gần như loại model văn bản (trọng số 0.002); Voting giữ nó và được lợi về accuracy. Chỉ số quyết định lựa chọn.
- **Trần độ chính xác thấp hơn 100%.** 1043 cặp (nghệ sĩ, tên bài) xuất hiện với nhiều nhãn
  khác nhau, nên không model nào đúng được hết.
- **Lớp 1 và 6 khó nhất**, nhầm chủ yếu sang lớp 10.

## Model tốt nhất hiện tại

- Phương pháp: Stacking 6 model (thêm LightGBM)
- Accuracy OOF: 0.7021 · F1 macro: 0.7534
- Điểm Kaggle (nộp 2026-11-15, late submission): **Public 0.73571** · Private 0.71111
