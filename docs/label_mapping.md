# Nguồn mapping nhãn và split ViGoEmotions

Đối chiếu ngày **22/10/2026**.

## Mapping ID–nhãn đã đối chiếu

Nguồn: ảnh bảng **Label mapping** trên [dataset card Hugging Face](https://huggingface.co/datasets/uitnlp/vigoemotions) do người dùng cung cấp. Đây là đối chiếu với ảnh nguồn, không phải xác thực đăng nhập hoặc tải metadata trực tiếp từ Hugging Face.

Đã so từng vị trí với `scripts/common.py::LABELS`: toàn bộ ID 0–26 khớp; `configs/midterm.json` đặt `neutral_id = 27`. Không cần đổi thứ tự đầu ra, đổi ground truth hay huấn luyện lại vì mapping.

| ID | Nhãn |
| --- | --- |
| 0 | amusement |
| 1 | excitement |
| 2 | joy |
| 3 | love |
| 4 | desire |
| 5 | optimism |
| 6 | caring |
| 7 | pride |
| 8 | admiration |
| 9 | gratitude |
| 10 | relief |
| 11 | approval |
| 12 | realization |
| 13 | surprise |
| 14 | curiosity |
| 15 | confusion |
| 16 | fear |
| 17 | nervousness |
| 18 | remorse |
| 19 | embarrassment |
| 20 | disappointment |
| 21 | sadness |
| 22 | grief |
| 23 | disgust |
| 24 | anger |
| 25 | annoyance |
| 26 | disapproval |
| 27 | neutral |

Dataset có 27 cảm xúc và neutral. Project dùng 27 đầu ra theo đề: neutral-only → vector 0; mixed-neutral → giữ emotion, bỏ neutral. Mapping đúng không tự xác minh quyết định mã hóa neutral là giao thức của paper; đây vẫn là quy ước của project.

## Paper xác nhận gì về split?

Nguồn: [Hung et al., EACL 2026](https://aclanthology.org/2026.eacl-long.129.pdf), mục **5.1 — Corpus Preparation**, trang PDF **6**, trang in **2810**. Đã đọc bản PDF người dùng cung cấp `2026.eacl-long.129.pdf`.

Paper nêu train/development/test theo tỷ lệ **8:1:1**. Không tìm thấy mô tả thuật toán chia tập, iterative multilabel stratification hoặc seed chia tập trong bản paper đã kiểm tra. Không suy ra split là random hay stratified từ tỷ lệ này; thiếu mô tả cũng không chứng minh tác giả không dùng stratification.

Các CSV được cấp có 16.531 / 2.066 / 2.067 mẫu, phù hợp xấp xỉ tỷ lệ công bố; tỷ lệ khớp không tự chứng minh đây là đúng cùng các ID split trong thí nghiệm paper. Project giữ các CSV/split ID được cấp, bỏ một mẫu train trùng text với test nên train thực nghiệm còn 16.530; giữ nguyên validation/test.

**Cách trình bày:** “Nhóm giữ split được cung cấp. Paper báo cáo tỷ lệ 8:1:1 nhưng không mô tả thuật toán chia tập; nhóm đã audit phân bố nhãn và lưu ID, chưa thể xác nhận multilabel stratification.”

Đề Midterm có yêu cầu stratified multilabel partition protocol. Việc paper không mô tả stratification chưa chứng minh split hiện tại đáp ứng yêu cầu đó. Cần làm rõ với giảng viên liệu chấp nhận split được cấp hay cần protocol bổ sung. Không tự chia lại giữa study đang chạy hoặc trộn điểm giữa các split.
