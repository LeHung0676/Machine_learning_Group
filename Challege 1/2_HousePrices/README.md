# 🏠 House Prices Prediction — ML Group (4 thành viên)

Dự đoán giá nhà (Kaggle: *House Prices - Advanced Regression Techniques*).
Metric chính: **RMSE trên log(SalePrice)**.

## Cấu trúc thư mục

```
2_HousePrices/
├── config/global_config.yaml      # Cấu hình toàn hệ thống
├── data/{raw,interim,processed}   # Dữ liệu gốc → tạm → hoàn chỉnh
├── docs/                          # Tài liệu, note kỹ thuật
├── models/                        # Model đã train (.joblib)
├── notebooks/
│   ├── global_eda.ipynb           # EDA tổng quan
│   └── experiments/               # Notebook riêng từng thành viên
├── reports/{experiments,figures}  # Log/metrics, biểu đồ
│   └── results_summary.md         # Bảng tổng hợp kết quả
├── src/
│   ├── eda/eda.py
│   ├── preprocess/{Clean,Preprocess,Feature}.py
│   ├── model/{LightGBM,Lasso}.py
│   ├── evaluation/Evaluation.py
│   ├── ensemble/Ensemble.py
│   ├── pipeline/Pipeline.py
│   ├── tracking/tracking.py
│   └── utils/data_loader.py
├── tests/                         # Unit tests
├── SUBMIT-summarize_report/       # File nộp Kaggle
└── main.py
```

## Bắt đầu

```bash
pip install -r requirements.txt
# Tải train.csv, test.csv từ Kaggle vào data/raw/
python main.py --member <ten_ban>
pytest tests/
```

## Quy ước làm việc nhóm
- Notebook cá nhân: `notebooks/experiments/<ten>_<chu_de>.ipynb`
- Code dùng chung đưa vào `src/`, không copy-paste giữa các notebook.
- Mỗi lần chạy thử nghiệm ghi một dòng vào `reports/results_summary.md`.
- Không commit dữ liệu và model (đã có `.gitignore`).

## Phân công (điền tên)
| Thành viên | Phụ trách |
|-----------|-----------|
|  | EDA + Clean |
|  | Feature Engineering + Preprocess |
|  | Model (Lasso, LightGBM) + tuning |
|  | Ensemble + Evaluation + Submission |
