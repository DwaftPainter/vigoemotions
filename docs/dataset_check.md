### ViGoEmotions: kiểm tra dữ liệu và quy ước midterm

Đã kiểm tra trực tiếp ba CSV, tổng 20.664 mẫu: train 16.531, validation 2.066, test 2.067.
Không thiếu dữ liệu, không trùng ID. Một cặp văn bản trùng train/test: train `1698`, test
`red000094`; cùng nhãn `[11]`. Pipeline bỏ mẫu train, giữ nguyên CSV và tập test.
Train dùng mô hình còn 16.530 mẫu. So sánh sau NFC và gom khoảng trắng cũng chỉ tìm thấy cặp này.

Có **306** mẫu chứa neutral cùng cảm xúc (train 251, val 19, test 36), **725** mẫu chỉ neutral,
và **1** mẫu test lặp nhãn trong danh sách. Không có nhãn ngoài 0–27 hoặc danh sách nhãn rỗng.
Notebook cũ dùng `and print(labels)` làm kết quả kiểm tra mixed-neutral thành `None`, dẫn đến đếm 0.
Đã sửa biểu thức kiểm tra thành boolean và bỏ in comment thô.

Theo quy ước đánh giá 27 cảm xúc của đề: ID 27 không là đầu ra riêng; neutral-only thành vector 0;
neutral + emotion giữ emotion, bỏ ID 27; các ID cảm xúc lặp được gộp. Đây là quyết định xử lý của
project, không khẳng định là sửa đúng ý định người gán nhãn. Chưa tái gán nhãn thủ công.

Ngày 22/10/2026 đã đối chiếu toàn bộ mapping trong ảnh dataset card Hugging Face do người dùng cung cấp: ID 0–26 khớp `scripts/common.py`, ID 27 là neutral. Không đổi mapping hay chạy lại mô hình vì mapping. Xem [nguồn và bảng đầy đủ](label_mapping.md).

Giữ split được cung cấp, xuất `splits/*.csv` và audit tỷ lệ từng nhãn. Paper gốc mục 5.1, trang PDF 6 (2810), xác nhận tỷ lệ 8:1:1 nhưng không mô tả thuật toán chia tập; xem [đối chiếu nguồn](label_mapping.md#paper-xác-nhận-gì-về-split).
Chưa có bằng chứng thuật toán tạo split gốc là iterative multilabel stratification;
không gọi split này là đã được chứng minh stratified. Không dùng nhãn test để xây ma trận đồng xuất hiện,
chọn siêu tham số hay ngưỡng. EDA phân bố biên theo split chỉ phục vụ kiểm tra mô tả.

Chạy `python scripts/analyze_data.py` để tái tạo các bảng, hình và ID split.
