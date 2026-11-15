# 🏠 House Prices Prediction — ML Group (4 thành viên)

Kaggle: *House Prices - Advanced Regression Techniques*.
Metric: **RMSE trên log1p(SalePrice)**, đánh giá bằng 5-fold out-of-fold (OOF).

## Kết quả hiện tại

| Model | RMSE OOF | CV mean ± std |
|---|---|---|
| Lasso | 0.11000 | 0.10978 ± 0.00685 |
| KernelRidge | 0.11052 | 0.11025 ± 0.00767 |
| ElasticNet | 0.11100 | 0.11077 ± 0.00702 |
| RandomForest | 0.13327 | 0.13293 ± 0.00956 |
| **Stacking (6 model)** | **0.10839** | |

Bảng đầy đủ ở `reports/results_summary.md`. LightGBM, XGBoost, CatBoost chưa chạy.

## Bắt đầu

```bash
# 1. Môi trường ảo (nên làm, tránh xung đột giữa Anaconda và Python hệ thống)
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt

# 2. Cài dự án thành package -> import src.* chạy được từ bất cứ đâu
pip install -e .

# 3. Dữ liệu: train.csv và test.csv từ Kaggle đặt vào data/raw/

# 4. Chạy — chọn một trong hai cách
#    a) Notebook: mở main.ipynb, sửa MEMBER ở ô tham số rồi Run All
#    b) Dòng lệnh:
python main.py --member <ten_ban>
python main.py --config config/config_<ten_ban>.yaml --member <ten_ban>

# 5. Kiểm thử
pytest tests/ -q                    # phải ra 46 passed
```

Hai cách cho **cùng một kết quả**, vì đều gọi `HousePricePipeline`.
Dùng `main.ipynb` khi cần xem bảng số và biểu đồ ngay tại chỗ, hoặc khi làm báo cáo.
Dùng `main.py` khi chỉ cần file nộp, hoặc khi chạy lại nhiều lần cho nhanh.

File nộp Kaggle: `SUBMIT-summarize_report/submission.csv`

## Cấu trúc thư mục

```
2_HousePrices/
├── config/
│   ├── global_config.yaml        # Cấu hình chung
│   └── config_<ten>.yaml         # Config riêng từng người, dùng `extends:`
├── data/
│   ├── raw/                      # train.csv, test.csv từ Kaggle
│   ├── interim/                  # Sau khi làm sạch
│   └── processed/                # Sau tiền xử lý hoàn chỉnh
├── docs/
│   └── data_cleaning_note.md     # Note kỹ thuật
├── models/                       # Model đã train (.joblib)
├── notebooks/
│   ├── global_eda.ipynb          # EDA tổng quan
│   └── experiments/              # Notebook riêng từng thành viên
├── reports/
│   ├── experiments/              # Log, metrics, bảng so sánh từng lần chạy
│   ├── figures/                  # Biểu đồ
│   └── results_summary.md        # Tổng hợp kết quả
├── src/
│   ├── eda/eda.py                # Class EDA
│   ├── preprocess/
│   │   ├── Clean.py              # Missing, duplicate, outlier
│   │   ├── Preprocess.py         # Scale, lệch phân phối, one-hot
│   │   └── Feature.py            # Mã hóa thứ bậc, tạo đặc trưng
│   ├── model/
│   │   ├── base.py               # BaseModel: giao diện chung
│   │   ├── Lasso.py  ElasticNet.py  KernelRidge.py
│   │   ├── RandomForest.py
│   │   └── LightGBM.py  XGBoost.py  CatBoost.py    (cần thư viện ngoài)
│   ├── evaluation/Evaluation.py  # Metrics, OOF cross-validation, biểu đồ
│   ├── ensemble/Ensemble.py      # Voting, Blending, Stacking, Bagging
│   ├── pipeline/Pipeline.py      # Luồng end-to-end
│   ├── tracking/tracking.py      # Log, metrics, lưu kết quả
│   └── utils/data_loader.py      # Load dữ liệu và config
├── tests/                        # 46 unit test
├── Nop_Classroom/                # Bài nộp cho lớp (Paper, Slide, Poster)
├── SUBMIT-summarize_report/      # submission.csv cho Kaggle
├── main.ipynb                    # Điểm vào dạng notebook
├── main.py                       # Điểm vào dạng dòng lệnh
├── setup.py
└── requirements.txt
```

## Luồng xử lý

```
main.py  ->  đọc config  ->  Pipeline
                                 |
     Clean  ->  Feature  ->  Preprocess
     (missing,   (thứ bậc,   (lệch phân phối,
      outlier)   đặc trưng)   scale, one-hot)
                                 |
                   5-fold OOF cho từng model
                                 |
                 Ensemble (tự tìm trọng số trên OOF)
                                 |
              submission.csv  +  log  +  biểu đồ
```

## Quy ước làm việc nhóm

**File sở hữu riêng.** Mỗi người chỉ sửa file của mình, tránh xung đột khi gộp:

| Thành viên | Phụ trách | File sở hữu |
|---|---|---|
|  | EDA + Clean | `src/eda/eda.py`, `src/preprocess/Clean.py`, `notebooks/global_eda.ipynb`, `docs/` |
|  | Feature + Preprocess | `src/preprocess/Feature.py`, `src/preprocess/Preprocess.py` |
|  | Model + tuning | `src/model/*.py` |
|  | Ensemble + Evaluation + Submission | `src/ensemble/`, `src/evaluation/`, `src/pipeline/`, `src/tracking/`, `reports/results_summary.md`, `main.ipynb` |

**Nhánh riêng, không commit thẳng vào `main`:**

```bash
git checkout -b feat/<ten>-<viec>
git add <file của mình>
git commit -m "feat(...): ..."
git push -u origin feat/<ten>-<viec>
```

Rồi mở Pull Request để người khác xem trước khi gộp.

**Config riêng.** Đừng sửa `global_config.yaml`. Tạo `config/config_<ten>.yaml`:

```yaml
extends: global_config.yaml
run:
  models: [lasso, elasticnet]
models:
  lasso:
    alpha: 0.0008
```

**Notebook cá nhân** đặt trong `notebooks/experiments/<ten>_<chu_de>.ipynb`.
Thứ gì thử thấy tốt thì chuyển vào `src/` rồi commit, đừng để code hay nằm mãi
trong notebook.

**Ghi kết quả.** Mỗi lần chạy, thêm một dòng vào `reports/results_summary.md`.
Lưu ý: độ lệch chuẩn giữa các fold khoảng 0.007, nên cải thiện nhỏ hơn con số đó
chưa thể kết luận là thật.

**Không commit dữ liệu sinh ra.** `.gitignore` đã chặn `models/`, `data/interim/`,
`data/processed/`, `mlruns/`, `catboost_info/`.
