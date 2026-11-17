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

## 3. Mạng nơ-ron truyền thẳng (MLP – Multi-Layer Perceptron)

### 3.1 Cấu trúc

MLP gồm lớp đầu vào, một hoặc nhiều **lớp ẩn** và lớp đầu ra. Mỗi nơ-ron tính tổng có trọng số của đầu vào rồi
đi qua **hàm kích hoạt phi tuyến**:

$$\mathbf{h}^{(l)} = \sigma\big(W^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\big), \qquad \hat{y} = \mathbf{w}^{\top}\mathbf{h}^{(L)} + b$$

Với bài toán hồi quy, lớp đầu ra là một nơ-ron tuyến tính (không có hàm kích hoạt).

Hàm kích hoạt phổ biến: **ReLU** $\sigma(z) = \max(0, z)$ – đơn giản, tránh hiện tượng gradient biến mất của
sigmoid/tanh. Nhờ các hàm phi tuyến, MLP có thể xấp xỉ quan hệ phi tuyến phức tạp (định lý xấp xỉ phổ quát).

### 3.2 Huấn luyện

- **Hàm mất mát:** sai số bình phương trung bình (MSE) + phạt L2 trên trọng số (tham số `alpha` trong scikit-learn).
- **Lan truyền ngược (backpropagation):** tính gradient của hàm mất mát theo từng trọng số bằng quy tắc chuỗi.
- **Tối ưu:** Adam – gradient descent với tốc độ học thích nghi cho từng tham số.
- **Early stopping:** tách một phần dữ liệu train làm validation nội bộ, dừng khi sai số trên đó không giảm nữa.

### 3.3 Siêu tham số chính

| Siêu tham số | Ý nghĩa |
|---|---|
| `hidden_layer_sizes` | Số lớp ẩn và số nơ-ron mỗi lớp, vd. (64, 32) |
| `alpha` | Hệ số phạt L2 – chống overfit |
| `learning_rate_init` | Tốc độ học ban đầu |
| `early_stopping`, `max_iter` | Điều kiện dừng |

### 3.4 Ưu điểm, hạn chế và lý do lựa chọn

- **Ưu điểm:** linh hoạt, mô hình hóa được quan hệ phi tuyến và tương tác giữa đặc trưng.
- **Hạn chế:** cần **chuẩn hóa dữ liệu**, nhạy với siêu tham số và khởi tạo ngẫu nhiên, khó diễn giải, thường cần
  nhiều dữ liệu hơn mô hình cây để vượt trội trên dữ liệu dạng bảng.
- **Lý do chọn:** đại diện cho hướng tiếp cận học sâu cơ bản, để so sánh với ensemble cây trên cùng bộ dữ liệu.

## 4. Độ đo đánh giá mô hình hồi quy

Với $n$ mẫu, giá trị thật $y_i$, dự đoán $\hat{y}_i$, trung bình $\bar{y}$:

| Độ đo | Công thức | Ý nghĩa |
|---|---|---|
| MAE | $\frac{1}{n}\sum \lvert y_i - \hat{y}_i \rvert$ | Sai lệch trung bình (Wh). Ít nhạy với ngoại lệ |
| RMSE | $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ | Phạt nặng lỗi lớn (các đỉnh tiêu thụ). Cùng đơn vị Wh |
| R² | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Tỷ lệ phương sai giải thích được; 0 = ngang dự đoán trung bình, có thể âm |
| MAPE | $\frac{100\%}{n}\sum \left\lvert \frac{y_i - \hat{y}_i}{y_i} \right\rvert$ | Sai số tương đối; dùng được vì $y \ge 10$ Wh |

**Lựa chọn của nhóm:** RMSE là độ đo chính (bài toán quan tâm cả các đỉnh tiêu thụ; so sánh được với bài báo gốc);
MAE và R² báo cáo kèm để diễn giải. Mọi độ đo tính trên **đơn vị Wh gốc**, kể cả khi mô hình huấn luyện trên `log1p(y)`.

## 5. Đánh giá trên dữ liệu chuỗi thời gian

- **Chia theo thời gian:** train là quá khứ, validation/test là tương lai. Chia ngẫu nhiên gây **rò rỉ dữ liệu** vì
  các mẫu liền kề (cách 10 phút) gần như giống nhau (Bergmeir & Benítez, 2012).
- **TimeSeriesSplit:** cross-validation kiểu "cửa sổ mở rộng" – fold $k$ huấn luyện trên $k$ đoạn đầu và đánh giá
  trên đoạn kế tiếp, nên không bao giờ dùng tương lai để dự đoán quá khứ.
