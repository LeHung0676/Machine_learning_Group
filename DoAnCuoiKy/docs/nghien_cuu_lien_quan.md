# Tổng quan nghiên cứu liên quan và hướng tiếp cận

## 1. Nghiên cứu gốc của bộ dữ liệu

**Candanedo, Feldheim & Deramaix (2017)** – *Data driven prediction models of energy use of appliances in a
low-energy house*, Energy and Buildings 140, 81–97.

- **Dữ liệu:** cảm biến nhiệt độ/độ ẩm không dây (ZigBee) trong nhà, dữ liệu thời tiết từ trạm sân bay Chièvres
  (Bỉ), điện năng của đèn; tần suất 10 phút trong ~4,5 tháng.
- **Lọc đặc trưng:** tác giả thêm hai biến ngẫu nhiên `rv1`, `rv2` để loại các biến không có khả năng dự đoán –
  biến nào xếp hạng thấp hơn biến ngẫu nhiên thì bị xem là không hữu ích.
- **Mô hình:** hồi quy tuyến tính bội, SVM (kernel RBF), Random Forest và Gradient Boosting Machine (GBM),
  huấn luyện bằng *repeated cross-validation*, đánh giá trên tập test riêng.
- **Kết quả chính:** GBM tốt nhất, giải thích khoảng **97% phương sai (R²) trên tập train nhưng chỉ ~57% trên tập
  test** khi dùng tất cả biến đầu vào.
- **Đặc trưng quan trọng:** trong mạng cảm biến, dữ liệu từ nhà bếp, phòng giặt và phòng khách được xếp hạng cao
  nhất; trong dữ liệu thời tiết, áp suất khí quyển là biến quan trọng nhất.

**Điểm nhóm rút ra:**
1. Khoảng cách R² train (97%) – test (57%) rất lớn → **overfitting** là vấn đề trọng tâm cần phân tích.
2. R² ≈ 0,57 là mốc tham khảo hợp lý; nhóm không kỳ vọng vượt xa con số này khi chỉ dùng dữ liệu môi trường.
3. Tái sử dụng ý tưởng biến ngẫu nhiên `rv1` để kiểm tra độ tin cậy của feature importance.

## 2. Các hướng tiếp cận phổ biến cho dự báo năng lượng tòa nhà

| Hướng | Ý tưởng | Ưu điểm | Hạn chế |
|---|---|---|---|
| Mô hình vật lý (EnergyPlus, TRNSYS) | Mô phỏng nhiệt động học của tòa nhà | Giải thích được bằng vật lý | Cần thông số chi tiết, không mô hình hóa được hành vi người dùng |
| Thống kê/tuyến tính | Hồi quy tuyến tính, ARIMA | Đơn giản, dễ diễn giải | Không bắt được quan hệ phi tuyến |
| Học máy cổ điển | Random Forest, Gradient Boosting, SVR, KNN | Bắt quan hệ phi tuyến, ít cần chuẩn hóa (cây) | Dễ overfit, khó diễn giải hơn |
| Học sâu | MLP, LSTM, CNN cho chuỗi thời gian | Mạnh với dữ liệu lớn, chuỗi dài | Cần nhiều dữ liệu, tốn tài nguyên, khó diễn giải |

## 3. Hướng tiếp cận của nhóm

- Bài toán **hồi quy có giám sát**, dữ liệu dạng bảng có thứ tự thời gian.
- **Chia dữ liệu theo thời gian** (không xáo trộn) để mô phỏng đúng tình huống thực tế: học từ quá khứ, dự đoán
  tương lai, đồng thời tránh rò rỉ thông tin.
- So sánh có hệ thống: baseline → tuyến tính (Linear/Ridge) → ensemble cây (Random Forest, Gradient Boosting)
  → MLP cơ bản.
- Tập trung **phân tích overfitting** (so với bài báo gốc) và **diễn giải đặc trưng quan trọng**.
- Mở rộng: thêm đặc trưng trễ (giá trị tiêu thụ 10–60 phút trước) để đánh giá giá trị của thông tin quá khứ gần.
