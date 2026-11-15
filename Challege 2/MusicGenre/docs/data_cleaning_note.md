# Note kỹ thuật: làm sạch dữ liệu Music Genre

Train 14 396 dòng, test 3 600 dòng, không có dòng trùng hoàn toàn.

| Cột | Số thiếu (train) | Bản chất | Cách xử lý |
|---|---|---|---|
| `instrumentalness` | 3541 (24,6%) | NA nghĩa là 0 | điền 0, thêm cờ `instr_missing` |
| `key` | 1609 (11,2%) | NA nghĩa là key = 0 | điền 0 |
| `Popularity` | 333 (2,3%) | thiếu thật | trung vị học từ train, thêm cờ `pop_missing` |

## Căn cứ

**`key`.** Thang của Spotify là 0..11. Dữ liệu có đủ giá trị 1..11 nhưng không có giá trị 0 nào,
còn số dòng thiếu (1609) nằm đúng trong khoảng tần suất của một cung nhạc (411 đến 1650).

**`instrumentalness`.** Giá trị nhỏ nhất là 1e-6, không có giá trị 0 nào. Tỉ lệ thiếu theo lớp:
lớp 5 là 61%, lớp 4 là 54%, lớp 9 là 50%, trong khi lớp 7 (nhạc không lời) chỉ 0,2%.
Thiếu ngẫu nhiên không thể tạo ra quy luật này.

**`duration_in min/ms`.** 2075 dòng (14,4%) có giá trị dưới 30, lớn nhất là 29,886; phần còn lại
nhỏ nhất là 55 775. Không có giá trị nào ở giữa. Nhóm quy về mili-giây và giữ cờ `duration_is_min`.
Tỉ lệ ghi bằng phút theo lớp: lớp 0, 3, 4, 7 là 100%; lớp 9 là 16,8%; lớp 10 là 3,6%; còn lại 0%.

## Lưu ý về cờ đơn vị

Cờ `duration_is_min` là dấu vết của cách thu thập dữ liệu, không phải đặc tính âm nhạc.
Nó hữu ích trong cuộc thi này vì tập test cũng có 14% dòng ghi bằng phút, nhưng một hệ thống
thật không nên dựa vào nó.

## Bài hát lặp lại

1103 cặp (nghệ sĩ, tên bài) xuất hiện nhiều hơn một lần, chiếm 2323 dòng; 1043 cặp trong đó
mang nhiều nhãn khác nhau. Nhóm không bỏ các dòng này vì không biết nhãn nào đúng.
