LAB 02 - HOUSE PRICES: ADVANCED REGRESSION TECHNIQUES
========================================================
Nhom: 
- 3124411174 Tat Vo Tien Nam
- 3124411166 Pham Anh Minh
- 3124411116 Le Long Hung

Cau truc thu muc:
- report/                 : Bao cao Word (CRISP-DM) - BaoCao_HousePrices_Lab02.docx
- source_code/            : Ma nguon
    - house_price_analysis.py : Script chinh (EDA, tien xu ly, huan luyen & danh gia 5 mo hinh)
    - build_report.js         : Script tao bao cao Word tu ket qua (docx-js)
- data/                   : train.csv, test.csv (Ames Housing dataset)
- output_results/         : Ket qua sinh ra sau khi chay script
    - results.json            : Toan bo so lieu, chi so danh gia
    - submission.csv           : File du doan nop Kaggle (Id, SalePrice)
    - *.png                    : Cac bieu do (phan phoi, missing values, correlation,
                                  so sanh mo hinh, feature importance, actual vs predicted)

Cach chay lai:
1. cd source_code
2. pip install pandas numpy scikit-learn matplotlib seaborn
3. python3 house_price_analysis.py   (dat train.csv/test.csv cung thu muc hoac sua duong dan)
4. (tuy chon) node build_report.js   (can cai dat npm package "docx")
