# Note kỹ thuật: làm sạch dữ liệu

> Người 1 cập nhật file này sau khi chạy `notebooks/global_eda.ipynb`.
> Lấy số liệu từ ô gọi `cleaner.summary()`.

## Kết quả (điền sau khi chạy)

- Dòng trước khi làm sạch:
- Dòng sau khi làm sạch:
- Trùng lặp đã bỏ:
- Id ngoại lai đã bỏ:
- Cột phải dùng lưới an toàn:

## Nguyên tắc xử lý dữ liệu thiếu

Bộ này có 19 cột thiếu dữ liệu trong train. Phần lớn **không phải lỗi**,
mà mang nghĩa "căn nhà không có hạng mục đó", theo `data_description.txt`:

| Nhóm cột | Số NA | Ý nghĩa | Cách xử lý |
|---|---|---|---|
| PoolQC | 1453 | không có hồ bơi | điền `"None"` |
| MiscFeature | 1406 | không có tiện ích phụ | điền `"None"` |
| Alley | 1369 | không có hẻm | điền `"None"` |
| Fence | 1179 | không có hàng rào | điền `"None"` |
| FireplaceQu | 690 | không có lò sưởi | điền `"None"` |
| Garage* (5 cột) | 81 | không có garage | `"None"` hoặc 0 |
| Bsmt* (5 cột) | 37-38 | không có tầng hầm | `"None"` hoặc 0 |
| MasVnrType/Area | 8 | không ốp tường | `"None"` / 0 |
| **LotFrontage** | 259 | **thiếu thật** | median theo Neighborhood |
| **Electrical** | 1 | **thiếu thật** | mode học từ train |

Điền mean hoặc median cho các cột nhóm trên là **sai nghĩa dữ liệu**.

## Hai quyết định cần giải thích trong báo cáo

**`GarageYrBlt` điền bằng `YearBuilt`, không điền 0.** Nhà không có garage
mà điền 0 thì cột tuổi garage thành hơn 2000 năm, phá hỏng mọi model tuyến tính.
Ngoài ra dữ liệu gốc có một dòng ghi `GarageYrBlt = 2207`, đã sửa về `YearBuilt`.

**Mọi giá trị điền đều học từ tập train rồi áp cho test.** Nếu tính median trên
cả hai tập gộp lại thì thông tin từ test rò rỉ vào huấn luyện, điểm
cross-validation sẽ đẹp hơn thực tế.

## Ngoại lai

Luật áp dụng: `GrLivArea > 4000` và `SalePrice < 300000`, **chỉ trên tập train**.
Không bao giờ bỏ dòng của test, vì file nộp Kaggle phải đủ 1459 dòng.

Kiểm tra cột `SaleCondition` của các dòng bị bỏ: nếu là `Partial` thì đó là nhà
bán khi chưa hoàn thiện, nên giá thấp dù diện tích lớn là hợp lý. Đây là căn cứ
để loại, thay vì chỉ nói "bỏ vì là ngoại lai".
