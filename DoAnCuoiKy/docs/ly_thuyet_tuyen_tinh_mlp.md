# Cơ sở lý thuyết – Mô hình tuyến tính, mạng nơ-ron MLP và độ đo đánh giá

(Tài liệu cho Chương 2 của báo cáo)

## 1. Hồi quy tuyến tính (Linear Regression)

### 1.1 Mô hình

Dự đoán là tổ hợp tuyến tính của các đặc trưng:

$$\hat{y} = w_0 + w_1 x_1 + \dots + w_p x_p = w_0 + \mathbf{w}^\top \mathbf{x}$$

Tham số được ước lượng bằng **bình phương tối thiểu (OLS)** – cực tiểu tổng bình phương phần dư:

$$\min_{\mathbf{w}} \; \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \lVert \mathbf{y} - X\mathbf{w} \rVert^2
\quad\Rightarrow\quad \hat{\mathbf{w}} = (X^\top X)^{-1} X^\top \mathbf{y}$$

### 1.2 Diễn giải hệ số

Khi các đặc trưng đã được chuẩn hóa (trung bình 0, độ lệch chuẩn 1), hệ số $w_j$ cho biết dự đoán thay đổi bao
nhiêu khi $x_j$ tăng một độ lệch chuẩn và các đặc trưng khác giữ nguyên.

### 1.3 Vấn đề đa cộng tuyến

Khi các đặc trưng tương quan mạnh với nhau, ma trận $X^\top X$ gần suy biến → nghịch đảo không ổn định:

- Hệ số có **phương sai lớn**, đổi dấu thất thường khi dữ liệu thay đổi nhẹ.
- Hai biến tương quan có thể nhận hệ số lớn và **trái dấu** để "bù trừ" nhau → không diễn giải được.

Đo bằng **VIF** (Variance Inflation Factor): $VIF_j = 1 / (1 - R_j^2)$, với $R_j^2$ là R² khi hồi quy $x_j$ theo các
đặc trưng còn lại. VIF > 10 được xem là đa cộng tuyến nghiêm trọng. Trong bộ dữ liệu này, 14/25 đặc trưng có VIF > 10.

## 2. Hồi quy có điều chuẩn: Ridge và Lasso

Thêm một số hạng **phạt độ lớn của hệ số** vào hàm mục tiêu để giảm phương sai (đánh đổi lấy một chút độ chệch).

### 2.1 Ridge (phạt L2) – Hoerl & Kennard (1970)

$$\min_{\mathbf{w}} \; \lVert \mathbf{y} - X\mathbf{w} \rVert^2 + \alpha \lVert \mathbf{w} \rVert_2^2
\quad\Rightarrow\quad \hat{\mathbf{w}} = (X^\top X + \alpha I)^{-1} X^\top \mathbf{y}$$

- Cộng $\alpha I$ vào đường chéo làm ma trận luôn khả nghịch → **xử lý trực tiếp đa cộng tuyến**.
- Co **mọi** hệ số về gần 0 nhưng không bằng 0; các biến tương quan chia sẻ hệ số với nhau.

### 2.2 Lasso (phạt L1) – Tibshirani (1996)

$$\min_{\mathbf{w}} \; \frac{1}{2n}\lVert \mathbf{y} - X\mathbf{w} \rVert^2 + \alpha \lVert \mathbf{w} \rVert_1$$

- Phạt L1 có "góc nhọn" tại 0 → nhiều hệ số **bằng đúng 0** → **tự động chọn đặc trưng**.
- Trong nhóm biến tương quan, Lasso thường giữ một biến và loại các biến còn lại.
- Không có nghiệm dạng đóng; giải bằng thuật toán coordinate descent.

### 2.3 Vai trò của alpha và chuẩn hóa

| alpha | Ảnh hưởng |
|---|---|
| alpha → 0 | Trở về hồi quy tuyến tính thường (phương sai cao) |
| alpha lớn | Hệ số co mạnh, mô hình đơn giản hơn (độ chệch tăng, có thể underfit) |

Phạt dựa trên độ lớn hệ số, mà độ lớn hệ số phụ thuộc thang đo đặc trưng (áp suất ~750 so với độ ẩm ~50) → **bắt buộc
chuẩn hóa** dữ liệu trước khi dùng Ridge/Lasso, để phạt công bằng giữa các đặc trưng.

### 2.4 Ưu điểm, hạn chế và lý do lựa chọn

- **Ưu điểm:** nhanh, dễ diễn giải, ít overfit (khi có điều chuẩn), là mốc tham chiếu tốt.
- **Hạn chế:** chỉ biểu diễn được quan hệ tuyến tính (trừ khi tự tạo đặc trưng phi tuyến), nhạy với ngoại lệ.
- **Lý do chọn:** làm mô hình tham chiếu "đơn giản nhất có học"; Ridge/Lasso xử lý đa cộng tuyến mạnh của bộ dữ
  liệu; hệ số giúp trả lời câu hỏi đặc trưng nào ảnh hưởng (RQ3).
