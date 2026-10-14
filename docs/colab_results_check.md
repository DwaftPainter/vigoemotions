# Kiểm tra kết quả Colab — 14/10/2026

Nguồn: `vigoemotions_results.zip` do người dùng cung cấp, gồm 56 file trong `reports/` và `splits/`.
Đã giữ bản kết quả local trước đó trong ZIP ở `backups/` trước khi nhập dữ liệu mới.

## Kết quả đã xác minh

| Mô hình | Epoch đã chạy / tốt nhất | Macro-F1 | Micro-F1 | Hamming Loss | LRAP | Thời gian train |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + OvR LR | Không áp dụng | 0,2388 | 0,2712 | 0,0606 | 0,6407 | 2,52 giây |
| BiGRU | 10 / 10 | 0,4060 | 0,4526 | 0,0604 | 0,6322 | 37,50 giây |
| PhoBERT | 4 / 4 | 0,4393 | 0,5160 | 0,0533 | 0,7158 | 3.047,11 giây (~50 phút 47 giây) |

Tất cả metric dùng cùng validation 2.066 mẫu, 27 cảm xúc, threshold 0,5. Neural chạy trên Tesla T4.
Thời gian neural gồm training, validation mỗi epoch và ghi checkpoint, không gồm tải mô hình/tách từ.
Thời gian TF-IDF gồm fit vectorizer + LR trên CPU, nên không so tốc độ tuyệt đối giữa các kiến trúc.

Đã kiểm tra checksum ba CSV khớp dữ liệu local; config khớp `configs/midterm.json`;
thứ tự ID/nhãn và ground truth trong NPZ khớp dữ liệu. Mỗi model có xác suất hữu hạn trong [0,1],
kích thước (2066,27). Tính lại Macro/Micro-F1, Hamming và LRAP từ xác suất đều khớp bảng trong ZIP.
History neural có đủ epoch; checkpoint tốt nhất theo validation là epoch cuối ở cả hai mô hình.
Điều này chưa chứng minh mô hình đã hội tụ; chưa thử thêm epoch hoặc seed.

## Q4 và sửa khác biệt giữa môi trường

PhoBERT cao hơn BiGRU về Macro-F1 toàn validation 3,33 điểm phần trăm.
Trong ba nhóm style khảo sát, chênh lệch lớn nhất ở **teencode**: 0,4111 so với 0,3170,
tăng 9,41 điểm phần trăm trên 307 mẫu. Đây là chênh lệch mô tả, chưa kiểm định ý nghĩa thống kê.

Bản local trước đó đếm teencode 726 mẫu do backend chuỗi/regex khác nhau: cùng biểu thức word boundary,
Python Unicode regex đếm 307 nhưng backend mặc định của pandas local đếm 726.
Đã tái hiện bằng các từ tiếng Việt: không được nhận chữ `k` trong `không` hoặc `kệ` là token teencode.
Đã sửa `style_masks` để dùng Python `re` trực tiếp và thêm test chống tái phát với các kiểu chuỗi.
Định nghĩa sau sửa khớp kết quả Colab, không cần huấn luyện lại; các metric toàn validation không đổi.

## Việc còn lại

- Đọc và diễn giải ca lỗi neural; danh sách 20 FN/20 FP tự sinh không thay thế việc audit thủ công.
- Xác minh mapping nhãn và giao thức split theo yêu cầu đề.
- Lưu riêng checkpoint và vocabulary từ Colab: ZIP này không chứa chúng, nên chưa kiểm tra được
  trọng số checkpoint hoặc suy luận lại từ trọng số.
- Bổ sung thông tin nhóm, rà soát báo cáo/slide; chưa đánh giá test hoặc thực hiện RQ Final.
