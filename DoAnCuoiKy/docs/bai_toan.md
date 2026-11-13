# Xác định bài toán

## 1. Bối cảnh

Nhà ở tiêu thụ khoảng 20–30% tổng năng lượng tại nhiều quốc gia, trong đó thiết bị gia dụng (tủ lạnh, máy giặt,
TV, lò vi sóng, ...) chiếm tỷ trọng đáng kể. Dự đoán được mức tiêu thụ của thiết bị giúp:

- Quản lý phụ tải và lập lịch sử dụng điện (demand response),
- Phát hiện tiêu thụ bất thường,
- Hỗ trợ thiết kế hệ thống quản lý năng lượng trong nhà thông minh.

Các ngôi nhà hiện đại thường đã có sẵn cảm biến nhiệt độ/độ ẩm. Câu hỏi đặt ra là liệu dữ liệu môi trường trong
nhà (vốn rẻ và sẵn có) cùng với thời tiết có đủ thông tin để ước lượng điện năng của thiết bị hay không.

## 2. Phát biểu bài toán

| Thành phần | Mô tả |
|---|---|
| **Input** | 18 biến nhiệt độ/độ ẩm trong nhà (T1–T9, RH_1–RH_9), 6 biến thời tiết, `lights`, và các đặc trưng thời gian trích xuất từ `date` |
| **Output** | `Appliances` – điện năng tiêu thụ của thiết bị trong khoảng 10 phút (Wh), số thực dương |
| **Loại bài toán** | Hồi quy có giám sát (supervised regression) trên dữ liệu chuỗi thời gian |
| **Độ đo chính** | RMSE, MAE, R² (trên thang Wh gốc) |

## 3. Câu hỏi nghiên cứu

1. **RQ1** – Dữ liệu cảm biến môi trường + thời tiết + thời gian dự đoán được điện năng thiết bị tốt đến mức nào
   so với một mô hình baseline đơn giản?
2. **RQ2** – Mô hình phi tuyến (Random Forest, Gradient Boosting) có vượt trội rõ rệt so với mô hình tuyến tính
   hay không, và vì sao?
3. **RQ3** – Những đặc trưng nào ảnh hưởng nhiều nhất đến điện năng tiêu thụ? Các biến ngẫu nhiên `rv1`, `rv2`
   có bị mô hình đánh giá thấp đúng như kỳ vọng không?
4. **RQ4 (mở rộng)** – Thêm thông tin quá khứ gần (giá trị tiêu thụ 10–60 phút trước) cải thiện dự đoán bao nhiêu?

## 4. Thách thức dự kiến

- Biến mục tiêu **lệch phải mạnh**: trung vị 60 Wh nhưng có những đỉnh tới 1080 Wh.
- Dữ liệu **chuỗi thời gian** → không được chia ngẫu nhiên, phải chia theo thời gian để tránh data leakage.
- Nhiều biến nhiệt độ/độ ẩm **tương quan mạnh** với nhau (đa cộng tuyến) → ảnh hưởng mô hình tuyến tính.
- Hành vi con người (bật/tắt thiết bị) gần như không quan sát được trực tiếp → giới hạn độ chính xác có thể đạt được.
