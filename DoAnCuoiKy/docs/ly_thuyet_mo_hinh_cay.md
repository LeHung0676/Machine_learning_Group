# Cơ sở lý thuyết – Các mô hình dựa trên cây quyết định

(Tài liệu cho Chương 2 của báo cáo)

## 1. Cây quyết định hồi quy (Decision Tree Regressor)

### 1.1 Nguyên lý

Cây quyết định chia không gian đặc trưng thành các vùng hình hộp bằng một chuỗi câu hỏi dạng
*"đặc trưng $x_j \le t$?"*. Mỗi lá của cây ứng với một vùng $R_m$ và dự đoán **giá trị trung bình** của các mẫu
huấn luyện rơi vào vùng đó:

$$\hat{y}(x) = \frac{1}{|R_m|} \sum_{i:\, x_i \in R_m} y_i \quad \text{với } x \in R_m$$

Tại mỗi nút, thuật toán CART chọn cặp (đặc trưng $j$, ngưỡng $t$) sao cho tổng sai số bình phương (SSE) của hai
nút con là nhỏ nhất:

$$\min_{j,\,t}\; \Big[\sum_{x_i \in R_L(j,t)} (y_i - \bar{y}_L)^2 + \sum_{x_i \in R_R(j,t)} (y_i - \bar{y}_R)^2\Big]$$

Quá trình lặp lại đệ quy cho tới khi đạt điều kiện dừng (độ sâu tối đa, số mẫu tối thiểu ở lá, ...).

### 1.2 Ưu điểm và hạn chế

| Ưu điểm | Hạn chế |
|---|---|
| Bắt được quan hệ phi tuyến và tương tác giữa các đặc trưng | **Phương sai cao**: thay đổi nhỏ trong dữ liệu có thể tạo ra cây rất khác |
| Không cần chuẩn hóa dữ liệu | Cây sâu dễ **overfitting** (học thuộc dữ liệu huấn luyện) |
| Dễ diễn giải khi cây nông | Dự đoán dạng bậc thang, không ngoại suy ngoài khoảng giá trị đã thấy |

Một cây đơn lẻ thường không đủ tốt → dùng **phương pháp tổ hợp (ensemble)** nhiều cây.

## 2. Random Forest

### 2.1 Nguyên lý

Random Forest (Breiman, 2001) huấn luyện $B$ cây quyết định độc lập và lấy **trung bình** dự đoán:

$$\hat{y}_{RF}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$$

Hai nguồn ngẫu nhiên giúp các cây khác nhau (giảm tương quan giữa các cây):

1. **Bagging (Bootstrap Aggregating):** mỗi cây học trên một mẫu bootstrap – lấy ngẫu nhiên có hoàn lại $n$ mẫu
   từ tập huấn luyện.
2. **Chọn ngẫu nhiên đặc trưng:** tại mỗi nút chỉ xét một tập con ngẫu nhiên gồm `max_features` đặc trưng.

**Vì sao hiệu quả?** Nếu mỗi cây có phương sai $\sigma^2$ và tương quan giữa hai cây là $\rho$, phương sai của
trung bình $B$ cây là $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. Tăng $B$ làm giảm số hạng thứ hai; chọn ngẫu nhiên
đặc trưng làm giảm $\rho$. Kết quả: **giảm phương sai mà gần như không tăng độ chệch (bias)**.

### 2.2 Siêu tham số chính

| Siêu tham số | Ý nghĩa | Ảnh hưởng |
|---|---|---|
| `n_estimators` | Số cây | Càng nhiều càng ổn định, tốn thời gian hơn; ít khi gây overfitting |
| `max_depth` | Độ sâu tối đa mỗi cây | Sâu → bias thấp nhưng dễ overfit |
| `min_samples_leaf` | Số mẫu tối thiểu ở lá | Lớn hơn → mô hình trơn hơn, giảm overfit |
| `max_features` | Số đặc trưng xét tại mỗi nút | Nhỏ → cây đa dạng hơn, giảm tương quan giữa cây |

### 2.3 Độ quan trọng đặc trưng

- **Impurity-based (MDI):** tổng mức giảm SSE nhờ các lần tách bằng đặc trưng đó. Nhanh nhưng **thiên vị** các
  đặc trưng liên tục có nhiều giá trị, và bị **chia sẻ** giữa các đặc trưng tương quan.
- **Permutation importance:** xáo trộn ngẫu nhiên một cột trên tập validation và đo mức tăng sai số. Đáng tin cậy
  hơn, nhóm sẽ dùng để kiểm tra lại cùng với biến đối chứng `rv1`.

### 2.4 Ưu điểm, hạn chế và lý do lựa chọn

- **Ưu điểm:** mạnh với dữ liệu bảng, chịu được ngoại lệ và đa cộng tuyến, ít cần tinh chỉnh, cho biết độ quan
  trọng đặc trưng.
- **Hạn chế:** mô hình lớn, dự đoán chậm hơn mô hình tuyến tính; không ngoại suy; khó diễn giải từng dự đoán.
- **Lý do chọn:** EDA cho thấy quan hệ giữa đặc trưng và `Appliances` là **phi tuyến** (Spearman > Pearson) và
  có **đa cộng tuyến mạnh** (14/25 đặc trưng VIF > 10) – Random Forest xử lý tốt cả hai. Bài báo gốc cũng dùng
  Random Forest nên kết quả có thể so sánh.
