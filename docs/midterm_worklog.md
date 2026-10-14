# ViGoEmotions — Các công việc từ bắt đầu đến midterm

Cập nhật trạng thái: **14/10/2026**. Tài liệu này tổng hợp công việc theo thứ tự thực hiện hợp lý,
dựa trên mã, notebook và kết quả đang có trong project; không phải nhật ký có thời điểm của từng thao tác.
Mục đích là giúp thành viên mới hiểu **đã làm gì, vì sao làm, xem bằng chứng ở đâu và còn thiếu gì**.

Đã chạy đủ TF-IDF, BiGRU và PhoBERT; kết quả Colab đã được nhập và kiểm chứng ngày 14/10/2026.
BiGRU: 10 epoch, 37,50 giây; PhoBERT: 4 epoch, khoảng 50 phút 47 giây trên Tesla T4.
Xem [kiểm tra kết quả Colab](colab_results_check.md). Còn xác minh mapping/split và audit lỗi neural trước khi nộp.

## 1. Đọc đề và xác định sản phẩm phải nộp

Đọc [đề ViGoEmotions](../02-vigoemotions-vietnamese-emotion.pdf) và
[student guide](../student-guide.pdf) để xác định bài toán, giới hạn dữ liệu và tiêu chí giữa kỳ.
Midterm là checkpoint thuyết trình, chiếm 30% điểm; cần có bằng chứng thực nghiệm tái lập được.

Bài toán là **phân loại đa nhãn**: một bình luận có thể có nhiều cảm xúc cùng lúc. Nhóm cần khảo sát
27 cảm xúc, xây ba baseline (TF-IDF + Logistic Regression, recurrent model, PhoBERT), đánh giá đúng
metric, phân tích lỗi và đề xuất câu hỏi nghiên cứu cho Final.

Sản phẩm cần chuẩn bị gồm báo cáo PDF, slide, code/notebook, README, bảng/hình EDA, kết quả baseline,
cấu hình thực nghiệm và các file giúp tái lập. Không đưa bình luận nguồn bị hạn chế phân phối vào repo công khai.

**Đã làm:** chuyển yêu cầu thành các phần trong [báo cáo](../reports/midterm_report.md).
Mục 8 của báo cáo đối chiếu từng nhóm yêu cầu với bằng chứng và trạng thái thực hiện.

## 2. Chuẩn bị dữ liệu và môi trường chạy

Dữ liệu hiện có gồm `data/train.csv`, `data/val.csv`, `data/test.csv`, với ba cột:

| Cột | Ý nghĩa |
| --- | --- |
| `id` | Định danh mẫu, dùng để đối chiếu split và kết quả |
| `text` | Bình luận tiếng Việt |
| `labels` | Danh sách ID cảm xúc được lưu dạng chuỗi |

Dữ liệu nguồn nằm trong `data/`, được loại khỏi Git bằng `.gitignore`. Các báo cáo công khai chỉ dùng
thống kê tổng hợp, ID và mô tả lỗi trừu tượng. Quy trình xin quyền dữ liệu nằm trong [README](../README.md).

Môi trường local của project là `ViGoEmotions/.venv`; các phiên bản thư viện được ghi ở
[requirements.txt](../requirements.txt). Môi trường Colab có file dependency riêng vì cần chạy neural model trên GPU.

**Sự cố đã xử lý:** notebook từng chọn nhầm `ATI/.venv`, có NumPy 1.26.4 nhưng SciPy 1.18.1 yêu cầu
NumPy từ 2.0, gây lỗi `numpy.long`. Đã đăng ký kernel **Python (ViGoEmotions)** và thêm cell kiểm tra
môi trường vào notebook 01–03. Ba notebook đã chạy lại thành công bằng kernel này.
Khi mở notebook, chọn đúng kernel rồi **Restart Kernel → Run All**; chi tiết ở README.

## 3. Kiểm tra chất lượng dữ liệu trước khi huấn luyện

Chạy [notebook 01](../notebooks/01_dataset_validation.ipynb) để kiểm tra số mẫu, schema,
giá trị thiếu, ID trùng, văn bản trùng và tính hợp lệ của nhãn. Nếu bỏ qua bước này, mô hình có thể
được đánh giá trên mẫu đã xuất hiện trong train hoặc sử dụng vector nhãn sai.

| Kết quả kiểm tra | Cách xử lý / ý nghĩa |
| --- | --- |
| Tổng 20.664 mẫu: train 16.531, val 2.066, test 2.067 | Khớp số mẫu trong đề |
| Không thiếu trường, không trùng ID | Có thể đối chiếu mẫu bằng ID |
| Train `1698` trùng text với test `red000094` | Loại mẫu train khỏi thực nghiệm; không sửa CSV gốc |
| 306 mẫu có neutral cùng cảm xúc | Giữ các cảm xúc, bỏ ID neutral khi mã hóa |
| 725 mẫu chỉ neutral | Mã hóa bằng vector 27 phần tử đều bằng 0 |
| 1 mẫu test lặp ID nhãn | Gộp nhãn lặp khi mã hóa |

Đã sửa lỗi notebook cũ dùng `and print(labels)` khi đếm mixed-neutral, làm kết quả đếm sai thành 0.
Quy ước xử lý chung nằm ở [common.py](../scripts/common.py), giải thích chi tiết ở
[dataset_check.md](dataset_check.md).

**Còn cần xác nhận:** mapping ID → tên nhãn được kế thừa từ notebook gốc; người có quyền truy cập
dataset cần đối chiếu với metadata nguồn. Không tự đổi thứ tự nhãn theo alphabet.

## 4. Thống nhất nhãn, split và tiền xử lý

Mỗi mẫu được chuyển thành vector **multi-hot 27 chiều**: vị trí của cảm xúc có mặt bằng 1, còn lại bằng 0.
ID 27 là neutral, không phải đầu ra thứ 28 trong giao thức project hiện tại. Ví dụ nhãn `[2, 11]`
đặt hai vị trí tương ứng bằng 1; đây chỉ là ví dụ mã hóa, không phải nội dung bình luận.

Pipeline chuẩn hóa Unicode NFC và gom khoảng trắng, giữ dấu tiếng Việt và thông tin cảm xúc ở bước chung.
Sau khi loại train trùng held-out, còn **16.530 mẫu train**. Vocabulary và IDF chỉ được học trên train.

Giữ ba split đã được cấp và lưu danh sách ID ở [splits](../splits). Vai trò từng tập:

| Tập | Dùng để làm gì trong giao thức hiện tại? |
| --- | --- |
| Train | Học tham số, vocabulary/IDF; tính quan hệ đồng xuất hiện giữa nhãn |
| Validation | Đánh giá giữa kỳ, chọn checkpoint neural, khảo sát ngưỡng |
| Test | Audit dữ liệu cơ bản; giữ lại để đánh giá mô hình ở Final |

**Giới hạn:** đã kiểm tra tỷ lệ nhãn giữa các split nhưng chưa có bằng chứng thuật toán tạo split gốc
là iterative multi-label stratification. Cần xác nhận nguồn giao thức split với bên cấp dữ liệu/giảng viên.
Nếu bắt buộc tạo split mới, phải khóa split trước rồi chạy lại tất cả mô hình, không trộn các kết quả cũ và mới.

## 5. Phân tích khám phá dữ liệu — EDA

Chạy [notebook 02](../notebooks/02_exploratory_analysis.ipynb) hoặc
[analyze_data.py](../scripts/analyze_data.py). Mục đích là hiểu đặc điểm dữ liệu trước khi giải thích điểm mô hình.

| Phân tích | Câu hỏi cần trả lời | File kết quả trong `reports/tables/` |
| --- | --- | --- |
| Tần suất 27 nhãn | Cảm xúc nào phổ biến/ít gặp? | `label_frequency.csv` |
| Cardinality | Một mẫu thường có bao nhiêu cảm xúc? | `cardinality.csv` |
| Độ dài | Bình luận ngắn/dài ra sao? | `text_lengths.csv` |
| Đồng xuất hiện | Những cảm xúc nào thường đi cùng nhau? | `train_cooccurrence.csv`, `train_top_pairs.csv` |
| Tương quan nhãn | Các biến nhãn liên hệ mạnh/yếu thế nào? | `train_correlation.csv` |
| Phong cách | Có bao nhiêu mẫu ngắn, nhiều emoji, teencode theo heuristic? | `text_styles.csv` |

Các hình nằm ở [reports/figures](../reports/figures). Ma trận đồng xuất hiện và tương quan chỉ dùng train.
Chia head/mid/tail bằng thứ hạng tần suất train, mỗi nhóm 9 nhãn; nếu hòa thì xét ID.

Một số phát hiện: train trung bình khoảng **1,859 cảm xúc/mẫu**, trung vị **11 token khoảng trắng**
và **47 ký tự**. Cặp disappointment–sadness có **904** mẫu train. Token khoảng trắng không đồng nghĩa
với từ đã tách của tiếng Việt hoặc BPE token. Các nhóm phong cách là heuristic và có thể chồng lấn.

**Trạng thái:** đã chạy và lưu đủ bảng/hình phục vụ phần EDA giữa kỳ.

## 6. Chạy baseline đầu tiên: TF-IDF + OvR Logistic Regression

Chạy [notebook 03](../notebooks/03_baseline_comparison.ipynb) hoặc
[train_tfidf.py](../scripts/train_tfidf.py). TF-IDF biến văn bản thành đặc trưng từ vựng;
One-vs-Rest huấn luyện một bộ phân loại nhị phân cho mỗi cảm xúc.

Cấu hình nằm ở [midterm.json](../configs/midterm.json): unigram/bigram, tối đa 40.000 features,
min_df=2, sublinear TF, Logistic Regression C=1, solver liblinear, seed 42.
Regex word-token của TF-IDF không biến emoji thành feature; đây là một hạn chế cần xét khi phân tích lỗi.

Kết quả validation tại ngưỡng 0,5:

| Metric | Giá trị |
| --- | --- |
| Macro-F1 | 0,2388 |
| Micro-F1 | 0,2712 |
| Hamming Loss | 0,0606 |
| LRAP | 0,6407 |
| Số hệ số và bias | 940.626 |

Macro-F1 coi các nhãn ngang nhau; Micro-F1 gộp các quyết định; Hamming đo tỷ lệ quyết định nhãn sai;
LRAP đo thứ hạng xác suất. Vì bài toán đa nhãn, không dùng subset accuracy làm metric chính.

**Bằng chứng:** [bảng metric](../reports/tables/tfidf_lr_metrics.csv),
[per-label F1](../reports/tables/tfidf_lr_per_label.csv),
[run metadata](../reports/tfidf_lr_run.json) và xác suất validation trong `reports/predictions/`.
Run metadata ghi cấu hình, môi trường, checksum dữ liệu và thời gian đo thực tế.

## 7. Giải thích kết quả và phân tích lỗi

Không dừng ở một điểm F1 tổng hợp. Đã đối chiếu điểm theo nhãn, tần suất, cardinality, phong cách văn bản
và khả năng dự đoán đồng thời các cặp cảm xúc.

| Câu hỏi giữa kỳ | Bằng chứng hiện có |
| --- | --- |
| Nhãn nào tốt/kém nhất? | TF-IDF: gratitude F1 khoảng 0,6797; realization bằng 0 tại ngưỡng 0,5 |
| Tần suất liên hệ với F1 ra sao? | Spearman khoảng 0,242; chưa có bằng chứng mạnh về quan hệ đơn điệu |
| Baseline giữ được đồng xuất hiện không? | Disappointment–sadness: đúng đủ hai nhãn ở 2/122 ca validation có cặp này |
| PhoBERT vượt recurrent model ở phong cách nào? | Trong các nhóm khảo sát, teencode có delta Macro-F1 lớn nhất: +9,41 điểm phần trăm (307 mẫu) |
| Ngưỡng 0,5 có ảnh hưởng mạnh không? | Có; khảo sát validation tốt nhất trong lưới ở 0,15, Macro-F1 khoảng 0,4844 |

Hạ ngưỡng làm F1 tăng nhưng Hamming Loss cũng tăng. Đây là kết quả thăm dò trên validation đã dùng chọn
ngưỡng, không phải bằng chứng cải thiện trên test.

Đã xuất **20 false negatives** (bỏ sót cảm xúc) và **20 false positives** (dự đoán thêm cảm xúc)
trong [bảng ca lỗi](../reports/tables/tfidf_lr_error_cases_val.csv), chọn theo mức tự tin sai.
Đã đọc thủ công **10 bình luận khác nhau**, ghi diễn giải ở mục 6 của báo cáo. Việc xuất 40 cặp lỗi
không có nghĩa đã đọc thủ công 40 mẫu; đây cũng không phải mẫu lỗi ngẫu nhiên.

## 8. Chuẩn bị BiGRU và PhoBERT trên Colab

Đã viết [train_neural.py](../scripts/train_neural.py) và
[notebook 04](../notebooks/04_colab_neural_baselines.ipynb). Luồng chuẩn bị gồm nạp dữ liệu,
huấn luyện, chọn checkpoint bằng validation Macro-F1, lưu xác suất, đánh giá và xuất lại báo cáo.

| Mô hình | Vai trò và thiết kế đã chuẩn bị | Trạng thái |
| --- | --- | --- |
| BiGRU | Học chuỗi token theo hai chiều; embedding 128, hidden 128 mỗi chiều | Đã chạy trên Tesla T4; có run JSON và xác suất validation |
| PhoBERT | Fine-tune mô hình tiếng Việt pretrained, tách từ VnCoreNLP trước BPE | Đã chạy trên Tesla T4; có run JSON và xác suất validation |

Cả hai có 27 logits, dùng BCEWithLogitsLoss và sigmoid; mỗi nhãn có thể được dự đoán độc lập.
Tên/revision checkpoint PhoBERT, seed và siêu tham số được lưu trong config.

Đã nhận và kiểm chứng kết quả chạy end-to-end của notebook Colab trên GPU Tesla T4.
Theo lựa chọn của người dùng, phần này được chuẩn bị để chạy trên Colab. Hướng dẫn upload gói code,
nạp CSV riêng và tải kết quả nằm trong README. Không điền số giả vào bảng khi chưa chạy.

## 9. Viết báo cáo, slide và hướng nghiên cứu Final

Đã tạo [báo cáo PDF](../reports/midterm_report.pdf),
[slide PDF](../reports/midterm_slides.pdf), các bản Markdown có thể sửa và README hướng dẫn tái lập.
[Nội dung slide](../reports/midterm_slides.md) có ghi chú thuyết trình cho từng trang.

[build_report.py](../scripts/build_report.py) đọc các kết quả hiện có để sinh báo cáo/slide.
Khi đã có output neural, script bổ sung dòng so sánh và bảng chênh lệch theo phong cách; người thực hiện
vẫn phải đọc các ca lỗi và chỉnh diễn giải, không coi việc tự sinh bảng là hoàn thành phân tích.

RQ Final được đề xuất là: **một ngưỡng chung 0,5 có tối ưu cho đủ 27 cảm xúc không?**
Thiết kế dự kiến chọn neural backbone tốt nhất từ midterm, giữ nguyên trọng số rồi so sánh ngưỡng fixed,
global tuned và per-label tuned. Tune trên validation, khóa ngưỡng, sau đó đánh giá test một lần.
Đây mới là thiết kế nghiên cứu; chưa thực hiện thí nghiệm Final.

## 10. Kiểm chứng và phần còn phải hoàn tất

Đã chạy notebook local 01–03 từ đầu đến cuối sau khi sửa kernel. Các kiểm tra tự động về nhãn hợp lệ,
neutral/mixed-neutral, rò rỉ văn bản train–held-out và metric trên ví dụ biết trước đều qua.
Đã kiểm tra ID khớp với xác suất validation, file PDF đọc được và gói Colab không chứa CSV nguồn.

Danh sách còn lại trước khi nộp:

- [ ] Đối chiếu mapping ID–nhãn với metadata dataset được cấp quyền.
- [ ] Xác nhận việc giữ split gốc có đáp ứng yêu cầu phân tầng đa nhãn trong đề hay không.
- [x] Chạy BiGRU/PhoBERT; nhập và kiểm chứng run metadata, probabilities.
- [ ] Lưu riêng checkpoint/vocab từ Colab (không có trong ZIP kết quả).
- [x] Cập nhật bảng đủ ba mô hình, per-label metric, số tham số và thời gian thực tế.
- [x] Trả lời Q4 bằng chênh lệch neural trên cùng nhóm validation; chưa kiểm định ý nghĩa thống kê.
- [ ] Đọc ca lỗi neural và cập nhật phần thảo luận/giới hạn, báo cáo và slide.
- [ ] Bổ sung tên, MSSV và phân công nhóm; kiểm tra lại toàn bộ gói trước nộp.

Khi cần tự đi qua project, đọc tài liệu này rồi mở notebook **01 → 02 → 03 → 04**.
README dành cho cách chạy; báo cáo dành cho giải thích phương pháp và kết quả chi tiết.
Các trạng thái và số liệu trong tài liệu này là bản tổng hợp tại ngày cập nhật; sau lượt chạy mới,
đối chiếu lại `reports/tables/` và `reports/*_run.json` trước khi trình bày.
