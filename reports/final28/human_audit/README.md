# Human error audit — PhoBERT 28

**Đã chuẩn bị 62 dòng cho 62 bình luận; chưa có dòng nào được người xác nhận.** Audit này dùng per-label decoder trên test của model 28, khác nghiên cứu Q1 Gratitude–Disapproval từ model 27. Mục tiêu là đọc lỗi và mô tả giới hạn, giữ nguyên scores, thresholds và metric Final.

Theo yêu cầu ngày 17/11, đã có [AI-assisted review đủ 62 ca](ai_assisted_review.md), [nhận xét từng ca](ai_assisted_review.csv) và [summary riêng](ai_assisted_summary.json). Worksheet có cả raw text và notes AI nằm private ở `data/final28/human_audit/ai_review_private.csv`. Bản human worksheet gốc không đổi; notes AI không tự cộng vào số human reviews. Khi người đọc muốn xác nhận, nên đọc comment trước notes AI để giảm định hướng.

Nhóm đã chốt dùng bản AI-assisted cho paper; [paper tiếng Anh](../../final_report.pdf) công bố rõ chưa có independent human adjudication. Hướng dẫn worksheet của người bên dưới được giữ cho phần kiểm chứng bổ sung; trạng thái human review không được đổi thành hoàn tất chỉ vì paper đã có audit AI.

## 1. Mở file để làm

Worksheet chứa bình luận gốc nằm **private, đã Git ignore**:

```text
data/final28/human_audit/review_private.csv
```

Mở bằng Excel/Numbers hoặc công cụ đọc CSV có UTF-8. File có BOM để giữ tiếng Việt. Giữ nguyên các cột và sample/review IDs; có thể sort/filter hàng. Cột `text` chứa comment cần đọc. Có thể ẩn hai cột vector dài để dễ xem; giữ chúng trong file khi lưu. Những ô text bắt đầu giống công thức spreadsheet được thêm dấu nháy đơn để hiển thị literal.

Khi nhập CSV vào spreadsheet, chọn **Text cho các cột** để giữ nguyên IDs và chuỗi xác suất, rồi lưu lại CSV UTF-8 với dấu phẩy. Tự chuyển kiểu số hoặc làm tròn probabilities có thể làm validator báo thay đổi cột readonly, dù bạn chỉ định sửa phần review.

[cases.csv](cases.csv) chỉ có IDs, labels, scores và thông tin dự đoán, không chứa raw text. [sampling_design.json](sampling_design.json) ghi quy tắc chọn, số lượng và source hashes. [summary.json](summary.json) ghi tiến độ được kiểm tra gần nhất.

## 2. Những dòng cần đọc

| `selection_group` | Số dòng | Việc cần xem |
| --- | ---: | --- |
| `core_FN` | 20 | Nhãn thật bị per-label bỏ sót; đối chiếu fixed/global |
| `core_FP` | 20 | Nhãn per-label dự đoán nhưng không có trong annotation |
| `neutral_FN`, `neutral_FP` | 2 | Một ca mỗi loại cho Neutral; bổ sung ngoài quota chính |
| `high_cardinality` | 5 | Mẫu có ít nhất 4 ground-truth labels, xem toàn bộ tập nhãn |
| `pair_20_21` | 5 | Disappointment–sadness |
| `pair_24_25` | 5 | Anger–annoyance |
| `pair_21_22` | 5 | Sadness–grief |

40 cặp chính được chọn bằng SHA-256 theo protocol, ưu tiên bình luận khác nhau trước khi lặp cặp trên cùng bình luận. Tổng population per-label có 1.467 FN và 1.907 FP; mỗi nhóm chính chọn 20, không thiếu quota. Hai ca Neutral bổ sung giữ nguyên 40 cặp chính. Packet có ba dòng mixed Neutral, nên không cần thêm mẫu mixed ngoài packet.

Ba cặp cảm xúc được chọn **chỉ từ retained train 27 emotions**, lần lượt có 904, 699 và 620 co-occurrences. Mỗi cặp lấy năm test cases theo hash quy định, đủ cả hai true labels hoặc đủ cả hai per-label predicted labels; loại lặp comment giữa các cặp. Đây là mẫu phục vụ đọc, không phải mẫu để ước lượng tỷ lệ lỗi toàn bộ test.

## 3. Cách đọc từng dòng

1. Đọc `text` và `true_labels`; xem `target_label` nếu có. Annotation là reference dùng để tính metric; có thể ghi nhận reference còn gây tranh luận mà không sửa annotation hoặc tính lại điểm.
2. Đối chiếu `pred_fixed_labels`, `pred_global_labels`, `pred_per_label_labels` và outcome TP/FP/FN/TN của target. Các dòng context không có target đơn; xem toàn bộ nhãn/cardinality hoặc cặp ghi ở `pair_labels`.
3. Xem score/cutoff để xác định quyết định đổi thế nào. `probability_vector` và `per_label_threshold_vector` dùng thứ tự 28 nhãn trong `configs/final_thresholds.json`; không tự chọn threshold mới.
4. Viết **quan sát** từ nội dung đã đọc và **giả thuyết** riêng. Nếu chưa rõ, ghi rõ chưa đủ cơ sở; không bắt buộc quy lỗi cho một cơ chế ngôn ngữ.
5. Điền các trường dưới đây và đổi `status` thành `complete` sau khi thực sự đọc.

| Trường được sửa | Cách điền |
| --- | --- |
| `reviewer` | Tên hoặc mã của thành viên thực sự đọc dòng đó |
| `status` | `pending` hoặc `complete` |
| `read_private_text` | `yes` sau khi đọc bình luận gốc |
| `reviewed_on` | Ngày thực tế, dạng `YYYY-MM-DD` |
| `error_category` | Một mã trong danh sách bên dưới |
| `reference_assessment` | `consistent`, `questionable`, hoặc `insufficient_context` |
| `observation` | Nhận xét cụ thể, diễn đạt trừu tượng từ comment; không chỉ chép lại “đây là FN/FP” |
| `hypothesis` | Giải thích có thể có, tách với quan sát; có thể ghi “chưa đủ cơ sở xác định” |

Mã `error_category`: `threshold_tradeoff`, `negation`, `implicit_emotion`, `sarcasm`, `slang_emoji`, `missing_context`, `label_overlap`, `annotation_query`, `other`, `unclear`. Các mã mô tả nhận định của reviewer; chúng không xác lập nguyên nhân bằng thực nghiệm. Dòng context có thể dùng `other` hoặc `unclear` nếu không thuộc các loại còn lại.

Neutral là output độc lập; mixed Neutral được giữ nguyên theo thiết kế. Không coi việc cùng có Neutral và emotion là lỗi annotation chỉ vì chúng cùng xuất hiện. Với một comment có nhiều cảm xúc, xem từng nhãn và cặp được giữ/bỏ giữa ba decoder.

**Chỉ sửa tám trường review.** Không sửa `text`, targets, probabilities hoặc các cột dự đoán. Không điền reviewer thay cho người chưa đọc, không dùng nhận xét AI làm bằng chứng human audit. Người nhận phần việc điền đúng tên/mã của mình; không có yêu cầu hai annotators độc lập trong protocol này.

## 4. Kiểm tra tiến độ sau khi lưu

Chạy từ project root:

```bash
.venv/bin/python scripts/final_human_audit.py check
```

Không cần GPU, không chạy inference. Lệnh kiểm IDs, các cột readonly, source text, reviewer/date/categories và các trường nhận xét của dòng `complete`. Đọc số `completed_core_fn`, `completed_core_fp` và `pending_rows` trong summary. Muốn hoàn tất packet hiện tại, cần đủ **20 core FN, 20 core FP và toàn bộ 22 context/supplement rows** đã đọc.

Validator kiểm tra khai báo và cấu trúc, không tự xác minh con người thực sự đã đọc hoặc nhận xét đúng. Nhóm vẫn chịu trách nhiệm đọc và xác nhận nội dung. Chỉ xuất public tiến độ và số lượng category; tên người, raw text, observations và hypotheses ở worksheet private không được tự động xuất vào repo.

Nếu cần chuẩn bị lại cùng packet:

```bash
.venv/bin/python scripts/final_human_audit.py prepare
```

Lệnh giữ nguyên review fields đã điền, từ chối ghi đè nếu sampling/metadata/source text thay đổi. Chỉ chạy lại khi cần khôi phục packet; không xóa worksheet có công sức review. Nếu sửa code sampling sau khi chuẩn bị, phải điều tra khác biệt thay vì thay mẫu để đủ số lượng.

## 5. Sau khi nhóm hoàn tất

Chạy `check`, gửi lại summary hoặc báo đã lưu worksheet. Khi viết Results/Discussion, tổng hợp các mẫu lỗi từ nhận xét đã xác nhận, phân biệt quan sát/giả thuyết và ghi số cặp/bình luận thực sự đọc. Không đưa nguyên văn bình luận restricted vào paper/repo public, không suy tỷ lệ population từ 62 mẫu đã chọn, và không dùng audit để tune lại bằng test.
