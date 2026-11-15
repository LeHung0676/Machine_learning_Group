# Tổng hợp kết quả — House Prices Prediction

Metric: **RMSE trên log1p(SalePrice)**, đánh giá bằng 5-fold OOF (out-of-fold).
Số thấp hơn là tốt hơn. Cột *Kaggle* điền sau khi nộp `submission.csv`.

## Mốc cơ sở (5-fold OOF, seed 42, 1458 dòng train sau khi bỏ 2 ngoại lai)

| Ngày | Người | Model | RMSE OOF | CV mean ± std | Kaggle | Ghi chú |
|------|-------|-------|----------|---------------|--------|---------|
| 2026-11-15 | team | Lasso (alpha=0.0005) | 0.11000 | 0.10978 ± 0.00685 | | mốc cơ sở |
| 2026-11-15 | team | KernelRidge (poly d=2) | 0.11052 | 0.11025 ± 0.00767 | | |
| 2026-11-15 | team | ElasticNet (l1=0.5) | 0.11100 | 0.11077 ± 0.00702 | | |
| 2026-11-15 | team | Ridge (alpha=15) | 0.11155 | 0.11127 ± 0.00789 | | không dùng, trùng Lasso |
| 2026-11-15 | team | HistGradientBoosting | 0.12335 | 0.12317 ± 0.00652 | | thay tạm cho LightGBM |
| 2026-11-15 | team | RandomForest | 0.13327 | 0.13293 ± 0.00956 | | yếu nhất nhưng vẫn góp khi trộn |
| | | LightGBM | | | | **chưa chạy, cần cài lightgbm** |
| | | XGBoost | | | | chưa chạy |
| | | CatBoost | | | | chưa chạy |

## Kết hợp model

| Ngày | Người | Phương pháp | Model thành phần | RMSE OOF | Kaggle |
|------|-------|-------------|------------------|----------|--------|
| 2026-11-15 | team | Blending | lasso + elasticnet + kernelridge + randomforest | 0.10896 | |
| 2026-11-15 | team | Stacking | 4 model như trên | 0.10867 | |
| 2026-11-15 | team | Voting | 4 model như trên | 0.10950 | |
| 2026-11-15 | team | Blending | thêm cả Ridge + HistGB (6 model) | 0.10841 | |
| 2026-11-15 | team | Stacking | 6 model | **0.10839** | |

Trọng số Blending tự tìm được cho 4 model:
`lasso 0.295, elasticnet 0.291, kernelridge 0.287, randomforest 0.126`

Ridge bị gán trọng số **0.0** khi có mặt Lasso và ElasticNet, nghĩa là nó không
đóng góp gì thêm. Đây là bằng chứng để bỏ Ridge khỏi danh sách.

## Nhận xét

**Trộn model có lợi nhưng không nhiều.** Từ 0.11000 xuống 0.10839, tức khoảng 1.5%.
Nguyên nhân: bốn model tuyến tính (Lasso, ElasticNet, Ridge, KernelRidge) sai theo
cùng một kiểu, nên trộn chúng gần như không thêm thông tin.

**Model cây mới là phần còn thiếu.** RandomForest và HistGB riêng lẻ tệ hơn rõ rệt
(0.133 và 0.123) nhưng vẫn được gán trọng số dương, vì chúng sai khác kiểu.
LightGBM mạnh hơn hai cái này nhiều, nên đây là việc đáng làm nhất tiếp theo.

**Độ lệch chuẩn giữa các fold khoảng 0.007.** Nghĩa là mọi cải thiện nhỏ hơn 0.007
chưa thể kết luận là thật. Muốn chắc thì chạy lại với seed khác.

## Model tốt nhất hiện tại

- Phương pháp: Stacking 6 model
- RMSE OOF: 0.10839
- Điểm Kaggle: *chưa nộp*
- File nộp: `SUBMIT-summarize_report/submission.csv`

## Cách ghi thêm dòng mới

Sau mỗi lần chạy, lấy số từ bảng `Evaluator.compare` mà `main.py` in ra,
hoặc từ `reports/experiments/<run_id>/model_comparison.csv`.
