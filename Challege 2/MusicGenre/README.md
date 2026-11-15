# 🎵 Music Genre Classification — Nhóm 14

Challenge 2, môn Nhập môn Học máy. Phân loại bài hát vào 11 thể loại từ thuộc tính âm thanh,
tên nghệ sĩ và tên bài. GVHD: TS. Đỗ Như Tài.

## Kết quả hiện tại (5-fold OOF, seed 42, 14 396 dòng train)

| Model | Accuracy | ± | F1 macro | Log loss |
|---|---|---|---|---|
| HistGradientBoosting | 0.6861 | 0.0042 | 0.7308 | 0.8557 |
| Random Forest | 0.6754 | 0.0047 | 0.7158 | 0.9366 |
| Extra Trees | 0.6752 | 0.0047 | 0.7148 | 0.9330 |
| Logistic Regression | 0.6598 | 0.0074 | 0.6968 | 0.9352 |
| Text (TF-IDF tên bài) | 0.4451 | 0.0050 | 0.4603 | 1.5789 |
| **Stacking 5 model** | **0.7003** | | **0.7508** | 0.8500 |

Bảng đầy đủ ở `reports/results_summary.md`. LightGBM, XGBoost, CatBoost đã có mã nhưng chưa chạy.

## Bắt đầu

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
pip install -e .

# đặt train.csv, test.csv vào data/raw/

pytest tests/ -q                    # phải ra 42 passed
python main.py --member <ten_ban>   # hoặc mở main.ipynb rồi Run All
```

File nộp Kaggle: `SUBMIT-summarize_report/submission.csv`

## Cấu trúc thư mục

```
MusicGenre/
├── config/
│   ├── global_config.yaml        # Cấu hình chung
│   └── config_<ten>.yaml         # Config riêng, kế thừa qua `extends:`
├── data/{raw,interim,processed}
├── docs/data_cleaning_note.md    # Note kỹ thuật
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
│   │   ├── Clean.py              # Đơn vị thời lượng, giá trị thiếu theo ngữ nghĩa
│   │   ├── Feature.py            # Khóa pair, đặc trưng tên bài
│   │   └── Preprocess.py         # Nhóm cột, bộ mã hóa target encoding
│   ├── model/
│   │   ├── base.py               # BaseModel: Pipeline(bộ mã hóa, bộ phân loại)
│   │   ├── HistGB.py  RandomForest.py  ExtraTrees.py  LogReg.py  TextModel.py
│   │   └── LightGBM.py  XGBoost.py  CatBoost.py     (cần thư viện ngoài)
│   ├── evaluation/Evaluation.py  # Metrics, OOF phân tầng, ma trận nhầm lẫn
│   ├── ensemble/Ensemble.py      # Voting, Blending, Stacking trên xác suất
│   ├── pipeline/Pipeline.py      # Luồng end-to-end
│   ├── tracking/tracking.py      # Log, metrics, lưu kết quả
│   └── utils/data_loader.py      # Load dữ liệu và config
├── tests/                        # 42 unit test
├── Nop_Classroom/                # Bài nộp cho lớp
├── SUBMIT-summarize_report/      # submission.csv cho Kaggle
├── main.ipynb                    # Điểm vào dạng notebook
├── main.py                       # Điểm vào dạng dòng lệnh
├── setup.py
└── requirements.txt
```

## Ba điểm cần nhớ về dữ liệu

1. **`duration_in min/ms` trộn hai đơn vị.** 14,4% dòng ghi bằng phút, còn lại bằng mili-giây.
   Bốn lớp 0, 3, 4, 7 ghi 100% bằng phút.
2. **`key` thiếu nghĩa là 0.** Thang là 0..11 nhưng dữ liệu không có giá trị 0 nào.
3. **`instrumentalness` thiếu nghĩa là 0.** Giá trị nhỏ nhất là 1e-6, tỉ lệ thiếu từ 0% đến 61% tùy lớp.

## Nguyên tắc chống rò rỉ

Target encoding cho nghệ sĩ và cho cặp (nghệ sĩ, tên bài) nằm **bên trong** pipeline của từng
model. Trong mỗi fold, nó chỉ học từ phần huấn luyện của fold đó. Test
`test_target_encoding_chi_hoc_tu_du_lieu_duoc_fit` canh nguyên tắc này.

## Quy ước làm việc nhóm

| MSSV | Thành viên | Phụ trách | File sở hữu |
|---|---|---|---|
| 3124411174 | Tất Võ Tiến Nam | EDA + Clean | `src/eda/`, `src/preprocess/Clean.py`, `notebooks/global_eda.ipynb`, `docs/` |
| 3124411166 | Phạm Anh Minh | Feature + Preprocess | `src/preprocess/Feature.py`, `src/preprocess/Preprocess.py` |
| 3124411116 | Lê Long Hưng | Model + tuning | `src/model/*.py` |
| 3124411009 | Lê Đức Anh | Ensemble + Evaluation + Submission | `src/ensemble/`, `src/evaluation/`, `src/pipeline/`, `main.ipynb` |

Mỗi người làm trên nhánh riêng, dùng `config/config_<ten>.yaml` riêng, notebook riêng trong
`notebooks/experiments/`. Không commit dữ liệu sinh ra (`.gitignore` đã chặn).
